"""
Core Pydantic models for all health data types (Phase 3).
"""

from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, Field


# ── CGM schemas ────────────────────────────────────────────────────────────────

class CGMReading(BaseModel):
    glucose_mg_dl: float
    timestamp: str
    trend: Optional[str] = None  # 'rising' | 'falling' | 'stable'
    device_id: Optional[str] = None


class CGMBatch(BaseModel):
    readings: List[CGMReading]


class CGMSummary(BaseModel):
    avg_glucose: float
    min_glucose: float
    max_glucose: float
    time_in_range_pct: float  # % of readings between 70-140
    spike_count: int           # readings > 180
    hypo_count: int            # readings < 70
    period_start: str
    period_end: str


# ── Mood schemas ───────────────────────────────────────────────────────────────

class MoodEntry(BaseModel):
    emotion: str
    stress_level: int = Field(..., ge=1, le=10)
    sleep_hours: Optional[float] = None
    notes: Optional[str] = None
    timestamp: str


# ── Food schemas ───────────────────────────────────────────────────────────────

class FoodLog(BaseModel):
    meal_type: str  # 'breakfast' | 'lunch' | 'dinner' | 'snack'
    items: List[str]
    photo_url: Optional[str] = None
    estimated_calories: Optional[int] = None
    glycemic_load: Optional[str] = None  # 'low' | 'medium' | 'high'
    timestamp: str


# ── Activity schemas ───────────────────────────────────────────────────────────

class ActivityLog(BaseModel):
    activity_type: str
    duration_minutes: int
    intensity: str  # 'low' | 'medium' | 'high'
    timestamp: str


# ── Nudge schemas ──────────────────────────────────────────────────────────────

class NudgeRecord(BaseModel):
    nudge_type: str
    message: str
    priority: str  # 'low' | 'medium' | 'high' | 'critical'
    timestamp: str
    delivered: bool = False
    acknowledged: bool = False
