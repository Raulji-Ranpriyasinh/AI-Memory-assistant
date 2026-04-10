"""
Program guidance Pydantic models.
"""

from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel


class ProgramSummary(BaseModel):
    id: str
    name: str
    progress_pct: float
    current_day: int
    total_days: int


class ProgramListResponse(BaseModel):
    programs: List[ProgramSummary]


class ProgramProgressRequest(BaseModel):
    task_id: str
    completed: bool
    notes: Optional[str] = None
