"""
Main entry point — FastAPI server + optional CLI mode.

Usage:
  FastAPI:  uvicorn main:app --reload --port 8000
  CLI:      python main.py --cli
"""

from __future__ import annotations

import sys

# Import the fully configured app from app.api.main
from app.api.main import app


# ── CLI fallback ───────────────────────────────────────────────────────────────

if "--cli" in sys.argv:
    from app.cli import main as cli_main

    cli_main()
elif __name__ == "__main__":
    import os
    import uvicorn
    # Only enable reload in local development, not in Docker
    reload_mode = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=reload_mode)