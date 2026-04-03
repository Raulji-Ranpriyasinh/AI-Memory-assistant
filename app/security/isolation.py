"""
Security & isolation helpers.

Centralises all user-ID sanitisation and Pinecone filter construction
so that a bug in one place cannot accidentally leak memories across users.
"""

from __future__ import annotations

import re
from typing import Any


# Only allow alphanumeric, hyphens, underscores, and dots in user IDs.
_SAFE_UID_RE = re.compile(r"^[A-Za-z0-9_\-\.]{1,128}$")


class SecurityError(ValueError):
    """Raised when a security constraint is violated."""


def validate_user_id(user_id: str) -> str:
    """
    Validate and return the user_id.
    Raises SecurityError if the ID contains dangerous characters or is empty.
    """
    if not user_id or not isinstance(user_id, str):
        raise SecurityError("user_id must be a non-empty string.")
    if not _SAFE_UID_RE.match(user_id):
        raise SecurityError(
            f"user_id '{user_id}' contains invalid characters. "
            "Only A-Z, a-z, 0-9, _, -, . are allowed (max 128 chars)."
        )
    return user_id


def build_pinecone_filter(user_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build a Pinecone metadata filter that is ALWAYS scoped to the validated user_id.
    Any caller-supplied extra filters are merged in, but cannot override user_id.
    """
    validated_uid = validate_user_id(user_id)
    filt: dict[str, Any] = {"user_id": {"$eq": validated_uid}}
    if extra:
        # Merge safely — never allow caller to override the user_id gate
        for k, v in extra.items():
            if k == "user_id":
                continue   # silently drop — cannot override the security gate
            filt[k] = v
    return filt


def assert_memory_owner(memory_user_id: str, requesting_user_id: str) -> None:
    """
    Assert that the memory belongs to the requesting user.
    Call this before any delete / update on a fetched vector.
    """
    if memory_user_id != requesting_user_id:
        raise SecurityError(
            f"Access denied: memory belongs to user '{memory_user_id}', "
            f"not '{requesting_user_id}'."
        )
