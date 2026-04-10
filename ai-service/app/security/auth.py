"""
JWT decoding and RBAC role management.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, List

from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt
from pydantic import BaseModel

from app.config import settings


# ── Role enumeration ───────────────────────────────────────────────────────────

class Role(str, Enum):
    patient = "patient"
    physician = "physician"
    admin = "admin"


# ── Custom exceptions ──────────────────────────────────────────────────────────

class AuthError(Exception):
    """Raised when JWT is invalid or expired."""

    def __init__(self, detail: str = "Authentication failed"):
        self.detail = detail
        super().__init__(detail)


# ── Pydantic models ────────────────────────────────────────────────────────────

class CurrentUser(BaseModel):
    user_id: str
    role: str


# ── JWT helpers ─────────────────────────────────────────────────────────────────

def decode_jwt(token: str) -> dict:
    """
    Decode and validate a JWT token.

    Uses JWT_SECRET and JWT_ALGORITHM from settings.
    Raises AuthError if the token is expired or has an invalid signature.
    Returns the decoded claims dict (user_id, role, exp, etc.).
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except JWTError as exc:
        raise AuthError(detail=f"Invalid token: {exc}")


# ── RBAC dependency (defined inline to avoid circular imports) ──────────────────

def require_role(*roles: Role) -> Callable:
    """
    FastAPI dependency factory that enforces role-based access control.

    Usage:
        @router.get("/admin-only", dependencies=[Depends(require_role(Role.admin))])
        async def admin_endpoint():
            ...

    Accepts one or more Role values. Raises HTTPException(403) if the
    current user's role is not in the allowed set.
    """
    allowed_roles = [r.value for r in roles]

    async def _check_role(request: Request):
        user_id = getattr(request.state, "user_id", None)
        role = getattr(request.state, "role", None)
        if role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' is not authorized. Required: {allowed_roles}",
            )
        return CurrentUser(user_id=user_id, role=role)

    return _check_role
