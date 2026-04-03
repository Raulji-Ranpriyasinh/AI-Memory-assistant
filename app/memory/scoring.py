# app/memory/scoring.py
import re
from typing import Dict, Any


def score_importance(text: str, context: Dict[str, Any] = None) -> float:
    """
    importance scoring with context awareness
    """
    context = context or {}
    base_score = 0.0

    # Keyword-based scoring
    high_importance_keywords = [
        "urgent",
        "asap",
        "important",
        "critical",
        "deadline",
        "meeting",
        "call",
        "emergency",
        "fail",
        "error",
        "bug",
    ]

    medium_importance_keywords = [
        "reminder",
        "note",
        "remember",
        "follow up",
        "task",
        "todo",
        "schedule",
        "plan",
    ]

    # Score based on keywords
    text_lower = text.lower()
    high_matches = sum(1 for word in high_importance_keywords if word in text_lower)
    medium_matches = sum(1 for word in medium_importance_keywords if word in text_lower)

    base_score += high_matches * 0.3
    base_score += medium_matches * 0.15

    # Length-based scoring (longer texts often more important)
    if len(text) > 200:
        base_score += 0.1
    elif len(text) > 500:
        base_score += 0.2

    # Context-based adjustments
    if context.get("conversation_length", 0) > 10:  # Long conversation
        base_score += 0.1

    if context.get("user_explicitly_asked_to_remember"):
        base_score += 0.4

    # Question detection (questions often important to remember)
    if "?" in text or text_lower.startswith(("what", "how", "why", "when", "where")):
        base_score += 0.1

    # Normalize to 0-1 range
    return min(base_score, 1.0)
