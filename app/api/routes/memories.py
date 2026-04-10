"""
Memory CRUD endpoints: GET/DELETE /memories, GET /memories/search, GET /history.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_chatbot, get_current_user
from app.api.schemas.memories import MemoriesResponse, SearchResponse, HistoryMessage, DeleteResponse
from app.security.auth import CurrentUser

router = APIRouter()


@router.get("/memories", response_model=MemoriesResponse)
async def get_memories(
    current_user: CurrentUser = Depends(get_current_user),
):
    """View all LTM memories + summaries."""
    try:
        chatbot = get_chatbot(current_user.user_id)
        return chatbot.get_memories()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/memories/search", response_model=SearchResponse)
async def search_memories(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results"),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Semantic search over LTM memories."""
    try:
        chatbot = get_chatbot(current_user.user_id)
        results = chatbot.search_memories(q, top_k=top_k)
        return SearchResponse(query=q, results=results, count=len(results))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/history")
async def get_history(
    current_user: CurrentUser = Depends(get_current_user),
):
    """View recent conversation history (STM-A)."""
    try:
        chatbot = get_chatbot(current_user.user_id)
        return chatbot.get_conversation_history()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/memories/{memory_id}", response_model=DeleteResponse)
async def delete_memory(
    memory_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Delete a specific memory by ID."""
    try:
        chatbot = get_chatbot(current_user.user_id)
        chatbot.delete_memory(memory_id)
        return DeleteResponse(status="deleted", memory_id=memory_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/memories", response_model=DeleteResponse)
async def delete_all_memories(
    current_user: CurrentUser = Depends(get_current_user),
):
    """Delete ALL memories for a user. Irreversible!"""
    try:
        chatbot = get_chatbot(current_user.user_id)
        chatbot.delete_all_memories()
        return DeleteResponse(status="all_deleted", memory_id=None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
