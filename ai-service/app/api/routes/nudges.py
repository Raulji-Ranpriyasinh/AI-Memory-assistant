"""
Nudge endpoints (Phase 5).
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/nudges")
async def get_nudges():
    """Get pending nudges. Phase 5 — not yet implemented."""
    return {"status": "not_implemented", "message": "Nudge endpoints are coming in Phase 5"}
