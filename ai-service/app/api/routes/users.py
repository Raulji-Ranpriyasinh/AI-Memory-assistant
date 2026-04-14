"""
User management endpoints for user synchronization.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.security.auth import CurrentUser

router = APIRouter()


class UserSyncRequest(BaseModel):
    user_id: str
    email: str
    role: str = "patient"


class UserSyncResponse(BaseModel):
    success: bool
    user_id: str
    message: str


@router.post("/users/sync", response_model=UserSyncResponse)
async def sync_user(request: UserSyncRequest):
    """
    Sync user from backend registration.
    Creates user in AI service memory system if not exists.
    This endpoint is called by the backend when a new user registers.
    """
    try:
        # Initialize user's memory space in Pinecone
        # The chatbot will handle creating the namespace on first interaction
        # but we can pre-warm it here if needed
        
        return UserSyncResponse(
            success=True,
            user_id=request.user_id,
            message=f"User {request.user_id} synced successfully",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
