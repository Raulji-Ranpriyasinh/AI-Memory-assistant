"""
Cross-signal CGM correlation engine (Phase 4).
Correlates CGM readings with food, mood, and activity to generate insights.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import (
    LTM_TOP_K,
    VALID_CATEGORIES,
)
from app.memory.controller import MemoryController
from app.models.schemas import MemoryCandidate
from app.observability.metrics import metrics
from app.prompts.health_prompts import CORRELATION_PROMPT


class CorrelationEngine:
    """
    Correlates CGM data with food, mood, and activity events
    to generate personalized health insights.
    """

    def __init__(self, controller: MemoryController):
        """
        Store MemoryController reference.
        Uses controller.chat_llm (the better Gemini model) for insight generation.
        """
        self.controller = controller
        self.llm = controller.chat_llm

    # ── Main correlation ───────────────────────────────────────────────────

    def correlate(
        self,
        user_id: str,
        trigger_event: str,
        lookback_hours: int = 4,
    ) -> Optional[str]:
        """
        1) Build timeline for user_id over lookback_hours.
        2) If timeline < 2 events, return None.
        3) Format timeline as readable string.
        4) Call Gemini with CORRELATION_PROMPT.
        5) Store insight as MemoryCandidate(category='cgm_pattern', salience=0.8).
        6) Return insight string.
        """
        timeline = self.build_timeline(user_id, lookback_hours)

        if len(timeline) < 2:
            return None

        # Format timeline as readable string
        timeline_str = "\n".join(
            f"[{evt['timestamp']}] ({evt['category']}) {evt['text']}"
            for evt in timeline
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=CORRELATION_PROMPT.format(timeline=timeline_str)),
                HumanMessage(content="Identify correlations in this timeline."),
            ])
            insight = response.content

            # Store as high-salience memory
            candidate = MemoryCandidate(
                text=f"Correlation insight: {insight[:200]}",
                category="cgm_pattern",
                salience=0.8,
            )

            # Inject into LTM via controller
            from langgraph.store.postgres import PostgresStore
            from app.config.settings import DB_URI

            with PostgresStore.from_conn_string(DB_URI) as store:
                self.controller.inject_health_memory(
                    store,
                    user_id,
                    f"thread-{user_id}",
                    candidate,
                )

            metrics.log("correlation_generated", user_id=user_id, events=len(timeline))
            return insight

        except Exception as exc:
            metrics.log("correlation_failed", user_id=user_id, error=str(exc))
            return None

    # ── Timeline builder ───────────────────────────────────────────────────

    def build_timeline(
        self,
        user_id: str,
        hours: int = 4,
    ) -> List[Dict[str, str]]:
        """
        Fetch all memories from LTM with categories in
        [cgm_pattern, dietary, mood_pattern, activity] from the last N hours.
        Sort by timestamp. Return list of {timestamp, category, text}.
        """
        relevant_categories = {
            "cgm_pattern", "dietary", "mood_pattern", "activity"
        }

        # Search LTM for each relevant category
        all_memories: List[Dict[str, Any]] = []
        for category in relevant_categories:
            try:
                results = self.controller.get_ltm_memories_semantic(
                    user_id, category, top_k=LTM_TOP_K
                )
                for m in results:
                    m["category"] = m.get("category", category)
                all_memories.extend(results)
            except Exception:
                continue

        # Filter by timestamp (last N hours)
        cutoff = datetime.now() - timedelta(hours=hours)
        filtered = []
        for m in all_memories:
            ts_str = m.get("timestamp", "") or m.get("created_at", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts >= cutoff:
                        filtered.append({
                            "timestamp": ts_str,
                            "category": m.get("category", "unknown"),
                            "text": m.get("text", ""),
                        })
                except (ValueError, TypeError):
                    continue
            else:
                # No timestamp — include anyway
                filtered.append({
                    "timestamp": "unknown",
                    "category": m.get("category", "unknown"),
                    "text": m.get("text", ""),
                })

        # Sort by timestamp
        filtered.sort(key=lambda e: e["timestamp"])
        return filtered

    # ── Recurring pattern detection ────────────────────────────────────────

    def detect_recurring_patterns(self, user_id: str) -> List[str]:
        """
        Fetch all cgm_pattern memories from LTM.
        Group by time-of-day.
        Send to LLM to identify patterns occurring 3+ times.
        Return list of pattern strings.
        """
        try:
            all_memories = self.controller.get_all_ltm_memories(user_id)
        except Exception:
            return []

        cgm_memories = [
            m for m in all_memories
            if m.get("category") == "cgm_pattern"
        ]

        if len(cgm_memories) < 3:
            return []

        # Format for LLM analysis
        patterns_text = "\n".join(
            f"- [{m.get('timestamp', 'unknown')}] {m.get('text', '')}"
            for m in cgm_memories
        )

        try:
            response = self.llm.invoke([
                SystemMessage(
                    content=(
                        "Analyze these CGM-related memory entries. "
                        "Identify any recurring patterns that appear 3 or more times. "
                        "Focus on time-of-day patterns, food-triggered responses, "
                        "and mood-glucose correlations. "
                        "Return ONLY a bulleted list of pattern descriptions."
                    )
                ),
                HumanMessage(content=patterns_text),
            ])
            patterns = [
                line.strip("- ").strip()
                for line in response.content.strip().split("\n")
                if line.strip().startswith("-") or line.strip()
            ]
            return patterns

        except Exception:
            return []

    # ── Food-glucose correlation ───────────────────────────────────────────

    def get_food_glucose_correlation(
        self,
        user_id: str,
        food_item: str,
    ) -> Optional[str]:
        """
        Search LTM for dietary memories containing food_item.
        Find corresponding cgm_pattern memories within 2 hours after each food log.
        Return correlation description or None if insufficient data.
        """
        # Search for dietary memories mentioning the food
        try:
            dietary_results = self.controller.get_ltm_memories_semantic(
                user_id, food_item, top_k=LTM_TOP_K
            )
        except Exception:
            return None

        food_events = [
            m for m in dietary_results
            if m.get("category") == "dietary" and food_item.lower() in m.get("text", "").lower()
        ]

        if not food_events:
            return None

        correlations = []
        for event in food_events:
            ts_str = event.get("timestamp", "")
            if not ts_str:
                continue

            try:
                food_time = datetime.fromisoformat(ts_str)
                window_end = food_time + timedelta(hours=2)
            except (ValueError, TypeError):
                continue

            # Find CGM memories within 2 hours after the food event
            try:
                all_ltm = self.controller.get_all_ltm_memories(user_id)
            except Exception:
                continue

            cgm_in_window = [
                m for m in all_ltm
                if m.get("category") == "cgm_pattern"
            ]

            for cgm in cgm_in_window:
                cgm_ts_str = cgm.get("timestamp", "")
                if not cgm_ts_str:
                    continue
                try:
                    cgm_time = datetime.fromisoformat(cgm_ts_str)
                    if food_time <= cgm_time <= window_end:
                        correlations.append({
                            "food_event": event.get("text", ""),
                            "food_timestamp": ts_str,
                            "cgm_response": cgm.get("text", ""),
                            "cgm_timestamp": cgm_ts_str,
                        })
                except (ValueError, TypeError):
                    continue

        if len(correlations) < 1:
            return None

        # Format correlation description
        result_parts = [
            f"Found {len(correlations)} food-glucose correlation(s) for '{food_item}':"
        ]
        for i, corr in enumerate(correlations, 1):
            result_parts.append(
                f"\n{i}. Ate: {corr['food_event']} ({corr['food_timestamp']})\n"
                f"   CGM response: {corr['cgm_response']} ({corr['cgm_timestamp']})"
            )

        return "\n".join(result_parts)
