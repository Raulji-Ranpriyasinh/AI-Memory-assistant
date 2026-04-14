"""
Food logging and recognition endpoints (Phase 3/6).
"""

from __future__ import annotations

import base64
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

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
    current_user: CurrentUser = Depends(get_current_user),
    image: UploadFile | None = File(None),
    image_path: str | None = Form(None),
    image_base64: str | None = Form(None),
):
    """
    Recognize food from image.
    Accepts image via file upload, file path, or base64 string.
    Uses Gemini Vision API for food recognition.
    Phase 6 — requires ENABLE_FOOD_RECOGNITION=true and GEMINI_API_KEY.
    """
    if not settings.ENABLE_FOOD_RECOGNITION:
        raise HTTPException(status_code=501, detail="Food recognition is not enabled")

    import os
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
    if not gemini_api_key:
        raise HTTPException(
            status_code=501,
            detail="GEMINI_API_KEY or GENAI_API_KEY not configured",
        )

    # Get image data from one of the input methods
    try:
        if image:
            # File upload
            content = await image.read()
            image_b64 = base64.b64encode(content).decode("utf-8")
        elif image_path:
            # File path
            path = Path(image_path)
            if not path.exists():
                raise HTTPException(status_code=400, detail=f"File not found: {image_path}")
            content = path.read_bytes()
            image_b64 = base64.b64encode(content).decode("utf-8")
        elif image_base64:
            # Base64 string
            image_b64 = image_base64
        else:
            raise HTTPException(
                status_code=400,
                detail="No image provided. Provide image via file upload, image_path, or image_base64",
            )

        from app.health.food_recognition import recognize_food_image
        result = recognize_food_image(image_b64)
        return FoodRecognizeResponse(**result)
    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Food recognition integration not yet implemented",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
