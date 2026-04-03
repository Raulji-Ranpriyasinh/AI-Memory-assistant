"""
Lightweight observability layer.

Metrics are written to:
  1. An in-memory ring buffer (last 10 000 events)
  2. A JSONL log file at LOG_FILE (configurable via env LTM_METRICS_LOG)

Call metrics.log(event, **kwargs) anywhere to record an event.
Call metrics.summary() for aggregated counts.
Call metrics.recent(n) for the last n raw events.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque, defaultdict
from datetime import datetime
from typing import Any, Deque


_LOG_FILE: str = os.getenv("LTM_METRICS_LOG", "ltm_metrics.jsonl")
_BUFFER_SIZE: int = 10_000
_ENABLED: bool = os.getenv("LTM_METRICS_ENABLED", "true").lower() == "true"


class _Metrics:
    """Thread-safe, zero-dependency metrics collector."""

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._events: Deque[dict[str, Any]] = deque(maxlen=_BUFFER_SIZE)
        self._counts: dict[str, int]        = defaultdict(int)
        self._sums:   dict[str, float]      = defaultdict(float)
        self._file_handle = None
        if _ENABLED and _LOG_FILE:
            try:
                self._file_handle = open(_LOG_FILE, "a", buffering=1)
            except OSError:
                pass

    # ── Public API ────────────────────────────────────────────────────────

    def log(self, event: str, **kwargs: Any) -> None:
        """Record a named event with optional key-value metadata."""
        if not _ENABLED:
            return
        record = {
            "ts":    datetime.utcnow().isoformat() + "Z",
            "event": event,
            **kwargs,
        }
        with self._lock:
            self._events.append(record)
            self._counts[event] += 1
            # Auto-sum numeric kwargs (e.g., count=5, latency_ms=123)
            for k, v in kwargs.items():
                if isinstance(v, (int, float)):
                    self._sums[f"{event}.{k}"] += v
            if self._file_handle:
                try:
                    self._file_handle.write(json.dumps(record) + "\n")
                except OSError:
                    pass

    def summary(self) -> dict[str, Any]:
        """Return aggregated event counts and sums."""
        with self._lock:
            return {
                "counts": dict(self._counts),
                "sums":   dict(self._sums),
            }

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the most recent n events."""
        with self._lock:
            events = list(self._events)
        return events[-n:]

    def reset(self) -> None:
        """Clear all in-memory state (useful in tests)."""
        with self._lock:
            self._events.clear()
            self._counts.clear()
            self._sums.clear()

    def __del__(self) -> None:
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass


# Singleton
metrics = _Metrics()
