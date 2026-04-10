"""
Memory CRUD Pydantic models.
"""

from __future__ import annotations

from typing import Optional, List, Dict

from pydantic import BaseModel


class MemoriesResponse(BaseModel):
    ltm: List[Dict]
    summaries: List[Dict]


class SearchResponse(BaseModel):
    query: str
    results: List[Dict]
    count: int


class HistoryMessage(BaseModel):
    role: str
    content: str


class DeleteResponse(BaseModel):
    status: str
    memory_id: Optional[str] = None
