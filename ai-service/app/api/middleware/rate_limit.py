"""
Per-user rate limiting middleware.
"""

from __future__ import annotations

import time
import logging
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Track request timestamps per user_id in a dict.
    Default: 60 requests per rolling 60-second window.
    Returns HTTP 429 with Retry-After header if limit exceeded.
    Prunes expired entries on each request.
    """

    # Class-level dict: user_id → list of timestamps
    _request_log: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip rate limiting for health check and non-api routes
        if path == "/api/v1/health" or not path.startswith("/api/v1"):
            return await call_next(request)

        user_id = getattr(request.state, "user_id", "anonymous")
        now = time.time()
        window_start = now - 60.0  # Rolling 60-second window

        # Prune expired entries for this user
        self._request_log[user_id] = [
            ts for ts in self._request_log[user_id] if ts > window_start
        ]

        # Check if limit exceeded
        if len(self._request_log[user_id]) >= settings.RATE_LIMIT_PER_MINUTE:
            oldest = self._request_log[user_id][0]
            retry_after = int(oldest + 60.0 - now) + 1
            logger.warning(
                "Rate limit exceeded for user %s on %s", user_id, path
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Too many requests.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # Record this request
        self._request_log[user_id].append(now)

        return await call_next(request)
