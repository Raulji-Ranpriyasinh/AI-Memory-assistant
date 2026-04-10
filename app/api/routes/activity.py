"""
Activity logging endpoints (Phase 3).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_chatbot, get_current_user
from app.api.schemas.health import ActivityLogRequest
from app.models.schemas import MemoryCandidate
from app.security.auth import CurrentUser

router = APIRouter()


@router.post("/activity")
async def log_activity(
    entry: ActivityLogRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Log an activity.
    Creates a MemoryCandidate and injects it into the health memory pipeline.
    Returns status and id.
    """
    try:
        chatbot = get_chatbot(current_user.user_id)

        # Build memory candidate text
        text = (
            f"{entry.activity_type} {entry.duration_minutes}min ({entry.intensity})"
        )

        salience = 0.5 if entry.intensity == "high" else 0.3
        candidate = MemoryCandidate(
            text=text,
            category="activity",
            salience=salience,
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
            "id": text,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
