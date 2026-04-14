"""
Auth endpoints for login and token generation (testing/development).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException
from jose import jwt
from pydantic import BaseModel

from app.config import settings

router = APIRouter()


class LoginRequest(BaseModel):
    user_id: str
    role: str = "patient"


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    role: str


@router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Generate a JWT token for testing.
    In production, this would be handled by your main backend authentication.
    """
    try:
        expire = datetime.utcnow() + timedelta(days=7)
        payload = {
            "user_id": request.user_id,
            "role": request.role,
            "exp": expire,
        }
        token = jwt.encode(
            payload,
            settings.JWT_SECRET,
            algorithm=settings.JWT_ALGORITHM,
        )
        return LoginResponse(
            access_token=token,
            user_id=request.user_id,
            role=request.role,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
