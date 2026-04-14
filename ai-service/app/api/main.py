"""
FastAPI app initialization, CORS setup, lifespan events, and router inclusion.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.middleware.auth import JWTMiddleware
from app.api.middleware.rate_limit import RateLimitMiddleware
from app.config import settings

# ── Logging setup ──────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ── Lifespan context manager ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info(
        "Starting Health AI Platform API v%s on %s:%s",
        "1.0.0",
        settings.API_HOST,
        settings.API_PORT,
    )
    logger.info("Feature flags — CGM: %s, Nudges: %s, Food Recognition: %s, Voice: %s",
                settings.ENABLE_CGM, settings.ENABLE_NUDGES,
                settings.ENABLE_FOOD_RECOGNITION, settings.ENABLE_VOICE)
    yield
    logger.info("Shutting down Health AI Platform API")


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(
    title="Health AI Platform API",
    description="Personalized digital health assistant with CGM intelligence, "
                "memory management, and proactive nudges.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware ────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Custom middleware ──────────────────────────────────────────────────────────

app.add_middleware(RateLimitMiddleware)
app.add_middleware(JWTMiddleware)

# ── Route includes (all under /api/v1) ─────────────────────────────────────────

from app.api.routes.chat import router as chat_router
from app.api.routes.memories import router as memories_router
from app.api.routes.system import router as system_router
from app.api.routes.cgm import router as cgm_router
from app.api.routes.mood import router as mood_router
from app.api.routes.food import router as food_router
from app.api.routes.activity import router as activity_router
from app.api.routes.nudges import router as nudges_router
from app.api.routes.programs import router as programs_router
from app.api.routes.auth import router as auth_router
from app.api.routes.users import router as users_router

app.include_router(auth_router, prefix="/api/v1", tags=["auth"])
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
app.include_router(users_router, prefix="/api/v1", tags=["users"])
app.include_router(memories_router, prefix="/api/v1", tags=["memories"])
app.include_router(system_router, prefix="/api/v1", tags=["system"])
app.include_router(cgm_router, prefix="/api/v1", tags=["cgm"])
app.include_router(mood_router, prefix="/api/v1", tags=["mood"])
app.include_router(food_router, prefix="/api/v1", tags=["food"])
app.include_router(activity_router, prefix="/api/v1", tags=["activity"])
app.include_router(nudges_router, prefix="/api/v1", tags=["nudges"])
app.include_router(programs_router, prefix="/api/v1", tags=["programs"])


# ── Global exception handlers ──────────────────────────────────────────────────

from app.security.auth import AuthError  # noqa: E402


@app.exception_handler(AuthError)
async def auth_error_handler(request, exc: AuthError):
    return JSONResponse(
        status_code=401,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
