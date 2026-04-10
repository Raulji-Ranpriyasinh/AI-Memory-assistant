"""
Program guidance endpoints (Phase 6).
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/programs")
async def list_programs():
    """List enrolled programs. Phase 6 — not yet implemented."""
    return {"status": "not_implemented", "message": "Program endpoints are coming in Phase 6"}


@router.post("/programs/{program_id}/progress")
async def update_program_progress(program_id: str):
    """Update program progress. Phase 6 — not yet implemented."""
    return {"status": "not_implemented", "message": "Program progress is coming in Phase 6"}
