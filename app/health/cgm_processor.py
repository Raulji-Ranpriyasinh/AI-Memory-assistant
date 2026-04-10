"""
CGM data normalization and pattern detection logic (Phase 3).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from app.config.settings import (
    CGM_HYPO_THRESHOLD,
    CGM_NORMAL_HIGH,
    CGM_NORMAL_LOW,
    CGM_SPIKE_THRESHOLD,
)
from app.models.health_schemas import CGMReading, CGMSummary
from app.models.schemas import MemoryCandidate


class CGMProcessor:
    """Processes CGM readings: normalization, spike/hypo detection, summaries, trends."""

    # ── Normalization ──────────────────────────────────────────────────────

    @staticmethod
    def normalize_readings(readings: List[CGMReading]) -> List[CGMReading]:
        """
        Validate: reject glucose values <= 0 or > 500.
        Sort by timestamp ascending. Return cleaned list.
        """
        cleaned = [r for r in readings if 0 < r.glucose_mg_dl <= 500]
        cleaned.sort(key=lambda r: r.timestamp)
        return cleaned

    # ── Spike detection ────────────────────────────────────────────────────

    @staticmethod
    def detect_spikes(readings: List[CGMReading]) -> List[dict]:
        """
        Find readings > CGM_SPIKE_THRESHOLD (180 mg/dL).
        Group consecutive spike readings into spike events.
        Return list of {start_time, end_time, peak_value, duration_minutes}.
        """
        spikes = [r for r in readings if r.glucose_mg_dl > CGM_SPIKE_THRESHOLD]
        if not spikes:
            return []

        events: list[dict] = []
        current_event = {
            "start_time": spikes[0].timestamp,
            "end_time": spikes[0].timestamp,
            "peak_value": spikes[0].glucose_mg_dl,
        }

        for i in range(1, len(spikes)):
            prev_dt = datetime.fromisoformat(spikes[i - 1].timestamp)
            curr_dt = datetime.fromisoformat(spikes[i].timestamp)
            gap_minutes = (curr_dt - prev_dt).total_seconds() / 60.0

            if gap_minutes <= 30:  # consecutive within 30 min
                current_event["end_time"] = spikes[i].timestamp
                current_event["peak_value"] = max(
                    current_event["peak_value"], spikes[i].glucose_mg_dl
                )
            else:
                # Close current event
                start_dt = datetime.fromisoformat(current_event["start_time"])
                end_dt = datetime.fromisoformat(current_event["end_time"])
                current_event["duration_minutes"] = round(
                    (end_dt - start_dt).total_seconds() / 60.0, 1
                )
                events.append(current_event)
                # Start new event
                current_event = {
                    "start_time": spikes[i].timestamp,
                    "end_time": spikes[i].timestamp,
                    "peak_value": spikes[i].glucose_mg_dl,
                }

        # Close last event
        start_dt = datetime.fromisoformat(current_event["start_time"])
        end_dt = datetime.fromisoformat(current_event["end_time"])
        current_event["duration_minutes"] = round(
            (end_dt - start_dt).total_seconds() / 60.0, 1
        )
        events.append(current_event)

        return events

    # ── Hypo detection ─────────────────────────────────────────────────────

    @staticmethod
    def detect_hypos(readings: List[CGMReading]) -> List[dict]:
        """
        Find readings < CGM_HYPO_THRESHOLD (70 mg/dL).
        Group into hypo events.
        Return list of {start_time, end_time, lowest_value, duration_minutes}.
        """
        hypos = [r for r in readings if r.glucose_mg_dl < CGM_HYPO_THRESHOLD]
        if not hypos:
            return []

        events: list[dict] = []
        current_event = {
            "start_time": hypos[0].timestamp,
            "end_time": hypos[0].timestamp,
            "lowest_value": hypos[0].glucose_mg_dl,
        }

        for i in range(1, len(hypos)):
            prev_dt = datetime.fromisoformat(hypos[i - 1].timestamp)
            curr_dt = datetime.fromisoformat(hypos[i].timestamp)
            gap_minutes = (curr_dt - prev_dt).total_seconds() / 60.0

            if gap_minutes <= 30:
                current_event["end_time"] = hypos[i].timestamp
                current_event["lowest_value"] = min(
                    current_event["lowest_value"], hypos[i].glucose_mg_dl
                )
            else:
                start_dt = datetime.fromisoformat(current_event["start_time"])
                end_dt = datetime.fromisoformat(current_event["end_time"])
                current_event["duration_minutes"] = round(
                    (end_dt - start_dt).total_seconds() / 60.0, 1
                )
                events.append(current_event)
                current_event = {
                    "start_time": hypos[i].timestamp,
                    "end_time": hypos[i].timestamp,
                    "lowest_value": hypos[i].glucose_mg_dl,
                }

        start_dt = datetime.fromisoformat(current_event["start_time"])
        end_dt = datetime.fromisoformat(current_event["end_time"])
        current_event["duration_minutes"] = round(
            (end_dt - start_dt).total_seconds() / 60.0, 1
        )
        events.append(current_event)

        return events

    # ── Summary computation ────────────────────────────────────────────────

    @staticmethod
    def compute_summary(readings: List[CGMReading]) -> Optional[CGMSummary]:
        """
        Calculate avg, min, max.
        Time-in-range % (readings 70-140 / total).
        Count spikes (>180) and hypos (<70).
        Return CGMSummary object.
        """
        if not readings:
            return None

        values = [r.glucose_mg_dl for r in readings]
        total = len(values)
        in_range = sum(1 for v in values if CGM_NORMAL_LOW <= v <= CGM_NORMAL_HIGH)
        spike_count = sum(1 for v in values if v > CGM_SPIKE_THRESHOLD)
        hypo_count = sum(1 for v in values if v < CGM_HYPO_THRESHOLD)

        return CGMSummary(
            avg_glucose=round(sum(values) / total, 1),
            min_glucose=min(values),
            max_glucose=max(values),
            time_in_range_pct=round((in_range / total) * 100, 1),
            spike_count=spike_count,
            hypo_count=hypo_count,
            period_start=readings[0].timestamp,
            period_end=readings[-1].timestamp,
        )

    # ── Trend detection ────────────────────────────────────────────────────

    @staticmethod
    def detect_trends(readings: List[CGMReading], window: int = 5) -> str:
        """
        Look at last `window` readings.
        If slope of linear regression > threshold: 'rising'.
        If < negative threshold: 'falling'.
        Else: 'stable'.
        """
        if len(readings) < window:
            return "stable"

        recent = readings[-window:]
        values = [r.glucose_mg_dl for r in recent]
        n = len(values)

        # Simple linear regression slope
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Thresholds for meaningful trend (mg/dL per reading interval)
        if slope > 2.0:
            return "rising"
        elif slope < -2.0:
            return "falling"
        else:
            return "stable"

    # ── Memory candidate creation ──────────────────────────────────────────

    @staticmethod
    def cgm_to_memory_candidate(readings: List[CGMReading]) -> Optional[MemoryCandidate]:
        """
        Create a MemoryCandidate from CGM readings.
        text = 'CGM session: avg {avg} mg/dL, {spikes} spikes, {hypos} hypos over {period}'
        category = 'cgm_pattern'
        salience based on severity (hypos and spikes increase importance).
        """
        if not readings:
            return None

        summary = CGMProcessor.compute_summary(readings)
        if not summary:
            return None

        period = f"{readings[0].timestamp} to {readings[-1].timestamp}"
        text = (
            f"CGM session: avg {summary.avg_glucose} mg/dL, "
            f"{summary.spike_count} spikes, "
            f"{summary.hypo_count} hypos over {period}"
        )

        # Salience: higher if there are concerning patterns
        salience = 0.3  # baseline for a single session
        if summary.hypo_count > 0:
            salience += 0.3
        if summary.spike_count > 2:
            salience += 0.3
        if summary.time_in_range_pct < 50:
            salience += 0.2

        from app.models.schemas import MemoryCandidate

        return MemoryCandidate(
            text=text,
            category="cgm_pattern",
            salience=min(salience, 1.0),
        )
