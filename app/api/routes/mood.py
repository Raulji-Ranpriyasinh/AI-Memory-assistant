"""
Mood logging endpoints (Phase 3).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_chatbot, get_current_user
from app.api.schemas.health import MoodEntryRequest
from app.models.schemas import MemoryCandidate
from app.security.auth import CurrentUser

router = APIRouter()


@router.post("/mood")
async def log_mood(
    entry: MoodEntryRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Log a mood entry.
    Creates a MemoryCandidate and injects it into the health memory pipeline.
    Returns status and optional correlation hint from LTM search.
    """
    try:
        chatbot = get_chatbot(current_user.user_id)

        # Build memory candidate text
        sleep_text = f", slept {entry.sleep_hours}h" if entry.sleep_hours else ""
        notes_text = f". Notes: {entry.notes}" if entry.notes else ""
        text = (
            f"Mood: {entry.emotion}, stress {entry.stress_level}/10"
            f"{sleep_text}{notes_text}"
        )

        candidate = MemoryCandidate(
            text=text,
            category="mood_pattern",
            salience=0.5 if entry.stress_level >= 7 else 0.3,
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

        # Search for mood-glucose correlation hints
        correlation_hint = None
        try:
            memories = chatbot.search_memories(
                f"glucose response to {entry.emotion} mood", top_k=3
            )
            if memories:
                correlation_hint = f"Found {len(memories)} related mood-glucose patterns in your history."
        except Exception:
            pass

        return {
            "status": "success",
            "id": candidate.text[:50] + "...",
            "correlation_hint": correlation_hint,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
