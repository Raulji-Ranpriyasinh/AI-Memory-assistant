"""
CGM data ingestion endpoints (Phase 3).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_chatbot, get_current_user
from app.api.schemas.health import CGMBatchRequest, CGMSummaryResponse
from app.health.cgm_processor import CGMProcessor
from app.models.schemas import MemoryCandidate
from app.security.auth import CurrentUser

router = APIRouter()


@router.post("/cgm/readings")
async def upload_cgm_readings(
    batch: CGMBatchRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Upload batch CGM readings.
    Normalizes, detects patterns, creates memory candidate, injects into chatbot.
    Returns status, summary, and any alerts.
    """
    if not batch.readings:
        raise HTTPException(status_code=400, detail="No readings provided")

    try:
        chatbot = get_chatbot(current_user.user_id)

        # Normalize readings
        cleaned = CGMProcessor.normalize_readings(batch.readings)
        if not cleaned:
            raise HTTPException(
                status_code=400,
                detail="No valid readings after normalization (all values must be 0-500 mg/dL)",
            )

        # Detect patterns
        spikes = CGMProcessor.detect_spikes(cleaned)
        hypos = CGMProcessor.detect_hypos(cleaned)
        summary = CGMProcessor.compute_summary(cleaned)

        # Build alerts
        alerts = []
        if hypos:
            lowest = min(h["lowest_value"] for h in hypos)
            if lowest < 54:
                alerts.append(
                    f"CRITICAL: Severe hypoglycemia detected ({lowest} mg/dL). Seek medical attention immediately."
                )
            else:
                alerts.append(
                    f"Hypoglycemia detected (lowest: {lowest} mg/dL). Consider fast-acting carbohydrates."
                )

        if spikes:
            peak = max(s["peak_value"] for s in spikes)
            if peak > 300:
                alerts.append(
                    f"Severe hyperglycemia detected ({peak} mg/dL). Contact your physician."
                )
            else:
                alerts.append(
                    f"Glucose spike detected (peak: {peak} mg/dL). Review recent food intake."
                )

        # Create and inject memory candidate
        candidate = CGMProcessor.cgm_to_memory_candidate(cleaned)
        if candidate:
            from langgraph.store.postgres import PostgresStore
            from app.config.settings import DB_URI

            with PostgresStore.from_conn_string(DB_URI) as store:
                chatbot.controller.inject_health_memory(
                    store,
                    current_user.user_id,
                    f"thread-{current_user.user_id}",
                    candidate,
                )

        return {
            "status": "success",
            "summary": summary.model_dump() if summary else None,
            "spike_events": len(spikes),
            "hypo_events": len(hypos),
            "alerts": alerts,
            "readings_processed": len(cleaned),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/cgm/summary", response_model=CGMSummaryResponse)
async def get_cgm_summary(
    period: str = Query(
        "24h", regex="^(24h|7d|30d)$", description="Summary period: 24h, 7d, or 30d"
    ),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get aggregated CGM summary for the specified period.
    Queries LTM for cgm_pattern memories.
    """
    try:
        chatbot = get_chatbot(current_user.user_id)
        memories = chatbot.get_memories()

        # Filter CGM-related memories
        ltm_memories = memories.get("ltm", [])
        cgm_memories = [
            m for m in ltm_memories if m.get("category") == "cgm_pattern"
        ]

        # Basic aggregation from available memories
        if not cgm_memories:
            return CGMSummaryResponse(
                avg_glucose=0.0,
                time_in_range=0.0,
                spike_count=0,
                hypo_count=0,
                period_start="",
                period_end="",
            )

        # Parse memory texts for summary data (basic extraction)
        total_spikes = 0
        total_hypos = 0
        for m in cgm_memories:
            text = m.get("text", "")
            if "spikes" in text:
                try:
                    # Extract numbers from text like "X spikes, Y hypos"
                    parts = text.split(",")
                    for part in parts:
                        if "spikes" in part:
                            total_spikes += int("".join(filter(str.isdigit, part.split()[0]) or "0"))
                        if "hypos" in part:
                            total_hypos += int("".join(filter(str.isdigit, part.split()[0]) or "0"))
                except (ValueError, IndexError):
                    pass

        return CGMSummaryResponse(
            avg_glucose=0.0,  # Would require actual value parsing
            time_in_range=0.0,
            spike_count=total_spikes,
            hypo_count=total_hypos,
            period_start=cgm_memories[-1].get("timestamp", ""),
            period_end=cgm_memories[0].get("timestamp", ""),
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
