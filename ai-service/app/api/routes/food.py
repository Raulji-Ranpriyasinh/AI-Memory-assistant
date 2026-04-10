"""
Food logging and recognition endpoints (Phase 3/6).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_chatbot, get_current_user
from app.api.schemas.health import FoodLogRequest, FoodRecognizeRequest, FoodRecognizeResponse
from app.config import settings
from app.models.schemas import MemoryCandidate
from app.security.auth import CurrentUser

router = APIRouter()


@router.post("/food/log")
async def log_food(
    entry: FoodLogRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Log a food entry.
    Creates a MemoryCandidate and injects it into the health memory pipeline.
    Returns status and id.
    """
    try:
        chatbot = get_chatbot(current_user.user_id)

        # Build memory candidate text
        items_text = ", ".join(entry.items)
        cal_text = f". {entry.estimated_calories} cal" if entry.estimated_calories else ""
        gl_text = f", GL: {entry.glycemic_load}" if entry.glycemic_load else ""
        text = f"{entry.meal_type}: {items_text}{cal_text}{gl_text}"

        candidate = MemoryCandidate(
            text=text,
            category="dietary",
            salience=0.5 if entry.glycemic_load == "high" else 0.4,
        )

        # Inject health memory
        from langgraph.store.postgres import PostgresStore
        from app.config.settings import DB_URI

        with PostgresStore.from_conn_string(DB_URI) as store:
            chatbot.controller.inject_health_memory(
                store,
                current_user.user_id,
                f"thread-{current_user.user_id}",
                candidate,
            )

        return {
            "status": "success",
            "id": text[:50] + "...",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/food/recognize", response_model=FoodRecognizeResponse)
async def recognize_food(
    request: FoodRecognizeRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Recognize food from image (base64-encoded).
    Phase 6 — requires ENABLE_FOOD_RECOGNITION=true and external API.
    """
    if not settings.ENABLE_FOOD_RECOGNITION:
        raise HTTPException(status_code=501, detail="Food recognition is not enabled")

    if not settings.NUTRITIONIX_API_KEY and not settings.GOOGLE_SPEECH_API_KEY:
        raise HTTPException(
            status_code=501,
            detail="No food recognition API configured (Nutritionix or Google Vision)",
        )

    try:
        from app.health.food_recognition import recognize_food_image
        result = recognize_food_image(request.image_base64)
        return FoodRecognizeResponse(**result)
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Food recognition integration not yet implemented",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
