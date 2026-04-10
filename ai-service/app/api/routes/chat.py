"""
Chat endpoints: POST /chat, POST /chat/voice (Phase 7).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_chatbot, get_current_user
from app.api.schemas.chat import ChatRequest, ChatResponse, VoiceChatRequest, VoiceChatResponse
from app.config import settings
from app.security.auth import CurrentUser

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Send a text message and get an AI response.
    Requires authenticated user.
    """
    try:
        chatbot = get_chatbot(current_user.user_id)
        response_text = chatbot.chat(request.message)
        return ChatResponse(response=response_text, user_id=current_user.user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat/voice", response_model=VoiceChatResponse)
async def voice_chat(
    request: VoiceChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Send a voice message (base64-encoded audio) and get an AI response.
    Phase 7 — requires ENABLE_VOICE=true and Google Speech API key.
    """
    if not settings.ENABLE_VOICE:
        raise HTTPException(status_code=501, detail="Voice interaction is not enabled")

    if not settings.GOOGLE_SPEECH_API_KEY:
        raise HTTPException(status_code=501, detail="Google Speech API key not configured")

    try:
        from app.integrations.speech import speech_to_text
        transcription = speech_to_text(request.audio_base64, request.format)
        chatbot = get_chatbot(current_user.user_id)
        response_text = chatbot.chat(transcription)
        return VoiceChatResponse(
            response=response_text,
            user_id=current_user.user_id,
            transcription=transcription,
        )
    except ImportError:
        raise HTTPException(status_code=501, detail="Speech-to-text integration not implemented")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
