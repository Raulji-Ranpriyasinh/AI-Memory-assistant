"""
FastAPI endpoints — exposes all CLI chatbot functionality as REST endpoints.
"""

from __future__ import annotations

from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.chatbot import MultiLayerChatbot

router = APIRouter()


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"


class ChatResponse(BaseModel):
    response: str
    user_id: str


class PruneResponse(BaseModel):
    pruned_count: int
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# Health-related schemas
class CGMRequest(BaseModel):
    user_id: str = "default_user"
    readings: List[dict] = []


class MoodRequest(BaseModel):
    user_id: str = "default_user"
    emotion: str = ""
    stressLevel: int = 5
    notes: str = ""
    timestamp: Optional[str] = None


class FoodLogRequest(BaseModel):
    user_id: str = "default_user"
    mealType: str = "lunch"
    items: List[str] = []
    timestamp: Optional[str] = None


# ── Helper ─────────────────────────────────────────────────────────────────────

def _get_chatbot(user_id: str) -> MultiLayerChatbot:
    """Create or retrieve a chatbot instance."""
    try:
        return MultiLayerChatbot(user_id=user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize chatbot: {e}")


# ── Chat endpoint ───────────────────────────────────────────────────────────────

@router.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get an AI response.
    Replaces the main CLI chat loop.
    """
    try:
        chatbot = _get_chatbot(request.user_id)
        response_text = chatbot.chat(request.message)
        return ChatResponse(response=response_text, user_id=request.user_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Memory endpoints ────────────────────────────────────────────────────────────

@router.get("/api/v1/memories")
async def get_memories(user_id: str = Query("default_user", description="User ID")):
    """
    View all LTM memories + summaries.
    Replaces CLI: /memories
    """
    try:
        chatbot = _get_chatbot(user_id)
        return chatbot.get_memories()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/history")
async def get_history(user_id: str = Query("default_user", description="User ID")):
    """
    View recent conversation history (STM-A).
    Replaces CLI: /history
    """
    try:
        chatbot = _get_chatbot(user_id)
        return chatbot.get_conversation_history()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/summaries")
async def get_summaries(user_id: str = Query("default_user", description="User ID")):
    """
    View detailed summaries (STM-B).
    Replaces CLI: /summaries
    """
    try:
        chatbot = _get_chatbot(user_id)
        memories = chatbot.get_memories()
        return memories.get("summaries", [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/memories/search")
async def search_memories(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results"),
    category: Optional[str] = Query(None, description="Filter by category"),
    user_id: str = Query("default_user", description="User ID"),
):
    """
    Semantic search over LTM memories.
    Replaces CLI: /search <query>
    """
    try:
        chatbot = _get_chatbot(user_id)
        results = chatbot.search_memories(query, top_k=top_k, category=category)
        return {"query": query, "results": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Maintenance endpoints ───────────────────────────────────────────────────────

@router.get("/api/v1/metrics")
async def get_metrics(user_id: str = Query("default_user", description="User ID")):
    """
    View observability metrics.
    Replaces CLI: /metrics
    """
    try:
        chatbot = _get_chatbot(user_id)
        return chatbot.get_metrics()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/prune", response_model=PruneResponse)
async def prune_memories(user_id: str = Query("default_user", description="User ID")):
    """
    Run memory decay & pruning.
    Replaces CLI: /prune
    """
    try:
        chatbot = _get_chatbot(user_id)
        pruned_count = chatbot.prune_stale_memories()
        return PruneResponse(
            pruned_count=pruned_count,
            message=f"Successfully pruned {pruned_count} stale memories",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Memory deletion endpoints ───────────────────────────────────────────────────

@router.delete("/api/v1/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    user_id: str = Query("default_user", description="User ID"),
):
    """Delete a specific memory by ID."""
    try:
        chatbot = _get_chatbot(user_id)
        chatbot.delete_memory(memory_id)
        return {"message": f"Memory {memory_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/v1/memories")
async def delete_all_memories(
    user_id: str = Query("default_user", description="User ID"),
):
    """
    Delete ALL memories for a user. Irreversible!
    """
    try:
        chatbot = _get_chatbot(user_id)
        chatbot.delete_all_memories()
        return {"message": f"All memories for user '{user_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health integration endpoints ────────────────────────────────────────────────

@router.post("/api/v1/cgm")
async def process_cgm(request: CGMRequest):
    """Process CGM readings and provide AI analysis."""
    try:
        chatbot = _get_chatbot(request.user_id)
        # Store readings as a memory for future context
        if request.readings:
            latest = request.readings[-1]
            glucose_val = latest.get("glucoseMgDl", 0)
            trend = latest.get("trend", "stable")
            analysis = (
                f"CGM reading: {glucose_val} mg/dL, trend: {trend}. "
                f"{'Glucose is in range.' if 70 <= glucose_val <= 140 else 'Glucose needs attention.'}"
            )
            chatbot.chat(f"My latest glucose is {glucose_val} mg/dL, trend is {trend}")
        return {
            "success": True,
            "message": "CGM data processed",
            "readings_count": len(request.readings),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/mood")
async def process_mood(request: MoodRequest):
    """Process mood entry and provide AI insight."""
    try:
        chatbot = _get_chatbot(request.user_id)
        chatbot.chat(
            f"I'm feeling {request.emotion}, stress level {request.stressLevel}/10. "
            f"Notes: {request.notes}"
        )
        return {
            "success": True,
            "message": "Mood logged",
            "aiHint": f"Remembered your {request.emotion} mood entry",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/food/log")
async def process_food(request: FoodLogRequest):
    """Process food log and provide AI analysis."""
    try:
        chatbot = _get_chatbot(request.user_id)
        items_str = ", ".join(request.items) if request.items else "no items"
        chatbot.chat(
            f"I ate {items_str} for {request.mealType}."
        )
        return {
            "success": True,
            "message": "Food logged",
            "meal_type": request.mealType,
            "items_count": len(request.items),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
