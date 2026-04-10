"""
All Pydantic models used across the memory system.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Candidates ────────────────────────────────────────────────────────────────

class MemoryCandidate(BaseModel):
    text: str             = Field(description="The memory content")
    category: str         = Field(description="identity | preferences | projects | facts")
    context: Optional[str] = Field(default=None, description="Additional context")
    timestamp: str        = Field(
        default_factory=lambda: datetime.now().isoformat()
    )


class ScoredCandidate(BaseModel):
    candidate: MemoryCandidate
    salience_score: float  = Field(ge=0.0, le=1.0)
    is_duplicate: bool
    reasoning: str


class CandidateExtractionResult(BaseModel):
    candidates: List[MemoryCandidate] = Field(default_factory=list)


class ScoringResult(BaseModel):
    scored_candidates: List[ScoredCandidate] = Field(default_factory=list)


# ── Summaries ─────────────────────────────────────────────────────────────────

class ConversationSummary(BaseModel):
    key_topics: List[str]      = Field(default_factory=list)
    decisions_made: List[str]  = Field(default_factory=list)
    action_items: List[str]    = Field(default_factory=list)
    important_context: str     = ""
    turn_range: str            = ""
    timestamp: str             = Field(
        default_factory=lambda: datetime.now().isoformat()
    )


class MergedSummary(BaseModel):
    key_topics: List[str]       = Field(default_factory=list)
    decisions_made: List[str]   = Field(default_factory=list)
    action_items: List[str]     = Field(default_factory=list)
    important_context: str      = ""
    chronological_flow: str     = ""


# ── Conflict resolution ───────────────────────────────────────────────────────

class ConflictRecord(BaseModel):
    new_candidate_idx: int
    existing_memory_id: str
    action: str   # "supersede" | "keep_both" | "ignore_new"
    reason: str   = ""


class ConflictResolutionResult(BaseModel):
    conflicts: List[ConflictRecord] = Field(default_factory=list)
