"""
Main entry point — FastAPI server + optional CLI mode.

Usage:
  FastAPI:  uvicorn main:app --reload --port 8000
  CLI:      python main.py --cli
"""

from __future__ import annotations

import sys

from fastapi import FastAPI
from endpoints import router as endpoints_router

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Layer Memory Chatbot API",
    description="REST API exposing all CLI chatbot functionality",
    version="1.0.0",
)

# Register all endpoints
app.include_router(endpoints_router)


# ── CLI fallback ───────────────────────────────────────────────────────────────

if "--cli" in sys.argv:
    from app.cli import main as cli_main

    cli_main()
elif __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)