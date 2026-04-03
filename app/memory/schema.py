# app/memory/schema.py

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone


def now_utc():
    return datetime.now(timezone.utc)


class ShortTermMemoryEntry(BaseModel):
    session_id: str
    key: str
    value: str
    timestamp: datetime = Field(default_factory=now_utc)


class LongTermMemoryEntry(BaseModel):
    id: str
    user_id: str
    text: str
    metadata: Optional[dict] = {}
    embedding: Optional[List[float]] = None
    timestamp: datetime = Field(default_factory=now_utc)


class MemoryEntry(BaseModel):
    id: Optional[str] = None  # For LTM; STM won't have it
    user_id: Optional[str] = None
    text: str
    metadata: Optional[dict] = {}
    timestamp: datetime
    source: str  # Either 'short_term' or 'long_term'
    importance: float = 0.5
