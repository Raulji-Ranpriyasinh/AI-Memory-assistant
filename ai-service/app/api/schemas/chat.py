"""
Chat request/response Pydantic models.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    response: str
    user_id: str


class VoiceChatRequest(BaseModel):
    audio_base64: str
    format: str = "wav"


class VoiceChatResponse(BaseModel):
    response: str
    user_id: str
    transcription: str
