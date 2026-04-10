"""
JWT Bearer token validation middleware.
"""

from __future__ import annotations

import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings
from app.security.auth import decode_jwt, AuthError

logger = logging.getLogger(__name__)


class JWTMiddleware(BaseHTTPMiddleware):
    """
    Intercepts all requests to /api/v1/* except /api/v1/health.
    Extracts and validates Bearer token from Authorization header.
    Sets request.state.user_id and request.state.role.
    Returns 401 if token is missing or invalid.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip auth for health check and non-api routes
        if path == "/api/v1/health" or not path.startswith("/api/v1"):
            return await call_next(request)

        # Extract Authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing Authorization header"},
            )

        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid Authorization format. Expected 'Bearer <token>'"},
            )

        # Decode and validate JWT
        try:
            claims = decode_jwt(token)
        except AuthError as exc:
            return JSONResponse(
                status_code=401,
                content={"detail": exc.detail},
            )

        # Attach to request state
        request.state.user_id = claims.get("user_id")
        request.state.role = claims.get("role")

        if not request.state.user_id or not request.state.role:
            return JSONResponse(
                status_code=401,
                content={"detail": "Token missing required claims: user_id, role"},
            )

        return await call_next(request)
