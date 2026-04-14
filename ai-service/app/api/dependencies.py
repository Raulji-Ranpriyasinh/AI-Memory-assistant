"""
Dependency functions: get_chatbot() and get_current_user().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Header, status

from app.config import settings
from app.security.auth import CurrentUser, AuthError, decode_jwt

if TYPE_CHECKING:
    from app.chatbot import MultiLayerChatbot


# ── Chatbot cache ──────────────────────────────────────────────────────────────

_chatbot_cache: dict[str, "MultiLayerChatbot"] = {}
"""Module-level dict mapping user_id → MultiLayerChatbot instance.
Avoids re-running Postgres schema setup on every request."""


def get_chatbot(user_id: str) -> "MultiLayerChatbot":
    """
    Return a cached MultiLayerChatbot for the given user_id.
    Creates a new instance if not already cached.
    """
    # Lazy import to avoid heavy dependency load at module import time
    import logging
    from app.chatbot import MultiLayerChatbot

    logger = logging.getLogger(__name__)

    if user_id not in _chatbot_cache:
        try:
            logger.info(f"Creating new chatbot instance for user {user_id}")
            _chatbot_cache[user_id] = MultiLayerChatbot(user_id=user_id)
            logger.info(f"Chatbot created successfully for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to create chatbot for user {user_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize chatbot: {str(e)}"
            )
    return _chatbot_cache[user_id]


# ── Auth dependency ────────────────────────────────────────────────────────────

def get_current_user(
    authorization: str | None = Header(default=None),
) -> CurrentUser:
    """
    FastAPI dependency that extracts the Authorization Bearer token,
    decodes it via decode_jwt(), and returns a CurrentUser object.

    Raises HTTPException(401) if the token is missing, expired, or invalid.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    # Extract token from "Bearer <token>"
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization format. Expected 'Bearer <token>'",
        )

    try:
        claims = decode_jwt(token)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.detail,
        )

    user_id = claims.get("user_id")
    role = claims.get("role")

    if not user_id or not role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claims: user_id, role",
        )

    return CurrentUser(user_id=user_id, role=role)
