"""
API request/response models for all health data endpoints.
"""

from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, Field


# ── CGM schemas ────────────────────────────────────────────────────────────────

class CGMReading(BaseModel):
    glucose_mg_dl: float
    timestamp: str
    trend: Optional[str] = None
    device_id: Optional[str] = None


class CGMBatchRequest(BaseModel):
    readings: List[CGMReading]


class CGMSummaryResponse(BaseModel):
    avg_glucose: float
    time_in_range: float
    spike_count: int
    hypo_count: int
    period_start: str
    period_end: str


# ── Mood schemas ───────────────────────────────────────────────────────────────

class MoodEntryRequest(BaseModel):
    emotion: str
    stress_level: int = Field(..., ge=1, le=10)
    sleep_hours: Optional[float] = None
    notes: Optional[str] = None


# ── Food schemas ───────────────────────────────────────────────────────────────

class FoodLogRequest(BaseModel):
    meal_type: str
    items: List[str]
    photo_url: Optional[str] = None
    estimated_calories: Optional[int] = None
    glycemic_load: Optional[str] = None


class FoodRecognizeRequest(BaseModel):
    image_base64: str


class FoodRecognizeResponse(BaseModel):
    items: List[str]
    estimated_calories: Optional[int] = None
    glycemic_load: Optional[str] = None
    portion_sizes: Optional[List[dict]] = None
    confidence: Optional[float] = None


# ── Activity schemas ───────────────────────────────────────────────────────────

class ActivityLogRequest(BaseModel):
    activity_type: str
    duration_minutes: int
    intensity: str
