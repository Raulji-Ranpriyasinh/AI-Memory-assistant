"""
System/health/metrics Pydantic models.
"""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str


class PruneResponse(BaseModel):
    pruned_count: int
    message: str


class MetricsResponse(BaseModel):
    counts: Dict
    sums: Dict


class ErrorResponse(BaseModel):
    detail: str
