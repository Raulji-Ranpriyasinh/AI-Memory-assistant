"""
System endpoints: GET /health, POST /prune, GET /metrics.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_chatbot, get_current_user
from app.api.schemas.system import HealthResponse, PruneResponse, MetricsResponse
from app.security.auth import CurrentUser

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint — NO AUTH REQUIRED.
    Used by load balancers and monitoring systems.
    """
    return HealthResponse(status="ok", version="1.0.0")


@router.post("/prune", response_model=PruneResponse)
async def prune_memories(
    current_user: CurrentUser = Depends(get_current_user),
):
    """Run memory decay & pruning on stale memories."""
    try:
        chatbot = get_chatbot(current_user.user_id)
        pruned_count = chatbot.prune_stale_memories()
        return PruneResponse(
            pruned_count=pruned_count,
            message=f"Successfully pruned {pruned_count} stale memories",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    current_user: CurrentUser = Depends(get_current_user),
):
    """View observability metrics for the user."""
    try:
        chatbot = get_chatbot(current_user.user_id)
        metrics = chatbot.get_metrics()
        return MetricsResponse(
            counts=metrics.get("counts", {}),
            sums=metrics.get("sums", {}),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
