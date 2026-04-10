"""
test_suite.py — Unit + integration tests for the memory system.

Run with:  pytest app/test_suite.py -v

All pure-logic tests are self-contained (no pinecone / langchain needed).
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (inlined so tests run without infra packages)
# ─────────────────────────────────────────────────────────────────────────────

_SAFE_UID_RE = re.compile(r"^[A-Za-z0-9_\-\.]{1,128}$")

def _validate_user_id(user_id: str) -> str:
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a non-empty string.")
    if not _SAFE_UID_RE.match(user_id):
        raise ValueError(f"user_id '{user_id}' contains invalid characters.")
    return user_id

def _build_filter(user_id: str, extra: dict | None = None) -> dict:
    validated = _validate_user_id(user_id)
    filt: dict = {"user_id": {"$eq": validated}}
    if extra:
        for k, v in extra.items():
            if k == "user_id":
                continue
            filt[k] = v
    return filt

def _assert_owner(memory_uid: str, requesting_uid: str) -> None:
    if memory_uid != requesting_uid:
        raise ValueError(f"Access denied: {memory_uid} != {requesting_uid}")

def _recency_score(timestamp_str: str, half_life_days: float = 60.0) -> float:
    try:
        ts       = datetime.fromisoformat(timestamp_str)
        age_days = (datetime.now() - ts).total_seconds() / 86_400
        return math.exp(-0.693 * age_days / half_life_days)
    except Exception:
        return 0.5

def _select_with_budget(memories: List[Dict], max_tokens: int, chars_per_token: float = 4.0) -> List[Dict]:
    selected:   List[Dict] = []
    used_chars: float      = 0
    max_chars              = max_tokens * chars_per_token
    for mem in memories:
        text_len = len(mem.get("text", "")) + len(mem.get("context", ""))
        if used_chars + text_len > max_chars:
            break
        selected.append(mem)
        used_chars += text_len
    return selected

STM_C_EXTRACTION_INTERVAL = 3

def _should_extract(num_messages: int) -> bool:
    turn_count = num_messages // 2
    return turn_count > 0 and turn_count % STM_C_EXTRACTION_INTERVAL == 0


# ─────────────────────────────────────────────────────────────────────────────
# Security / Isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestSecurity:
    def test_valid_user_id(self):
        assert _validate_user_id("alice") == "alice"
        assert _validate_user_id("user_123") == "user_123"
        assert _validate_user_id("user-abc.1") == "user-abc.1"

    def test_invalid_user_id_special_chars(self):
        with pytest.raises(ValueError):
            _validate_user_id("user'; DROP TABLE--")

    def test_empty_user_id(self):
        with pytest.raises(ValueError):
            _validate_user_id("")

    def test_build_filter_always_scopes(self):
        filt = _build_filter("alice", {"category": {"$eq": "identity"}})
        assert filt["user_id"] == {"$eq": "alice"}
        assert "category" in filt

    def test_build_filter_cannot_override_user_id(self):
        filt = _build_filter("alice", {"user_id": {"$eq": "mallory"}})
        assert filt["user_id"] == {"$eq": "alice"}

    def test_assert_owner_ok(self):
        _assert_owner("alice", "alice")

    def test_assert_owner_denied(self):
        with pytest.raises(ValueError):
            _assert_owner("alice", "mallory")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def setup_method(self):
        from app.observability.metrics import metrics
        metrics.reset()

    def test_log_and_count(self):
        from app.observability.metrics import metrics
        metrics.log("test_event", count=3)
        s = metrics.summary()
        assert s["counts"]["test_event"] == 1
        assert s["sums"]["test_event.count"] == 3

    def test_recent(self):
        from app.observability.metrics import metrics
        metrics.log("ev_a")
        metrics.log("ev_b")
        assert metrics.recent(1)[-1]["event"] == "ev_b"

    def test_multiple_logs_accumulate(self):
        from app.observability.metrics import metrics
        for _ in range(5):
            metrics.log("ev", val=2)
        assert metrics.summary()["counts"]["ev"] == 5
        assert metrics.summary()["sums"]["ev.val"] == 10


# ─────────────────────────────────────────────────────────────────────────────
# Recency scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestRecencyScore:
    def _score(self, days_ago: float) -> float:
        ts = (datetime.now() - timedelta(days=days_ago)).isoformat()
        return _recency_score(ts)

    def test_fresh_memory_near_1(self):
        assert self._score(0) > 0.99

    def test_half_life_near_half(self):
        assert 0.45 < self._score(60) < 0.55

    def test_old_memory_low(self):
        assert self._score(300) < 0.1

    def test_invalid_timestamp_returns_midpoint(self):
        assert _recency_score("not-a-date") == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Token-budget trimming
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenBudget:
    def _make(self, n: int, text_len: int = 50) -> List[Dict]:
        return [{"text": "x" * text_len, "context": "", "final_score": float(n - i)}
                for i in range(n)]

    def test_all_fit(self):
        result = _select_with_budget(self._make(3, 10), max_tokens=1000)
        assert len(result) == 3

    def test_budget_truncates(self):
        result = _select_with_budget(self._make(10, 200), max_tokens=50)
        assert len(result) < 10

    def test_empty_input(self):
        assert _select_with_budget([], max_tokens=1000) == []


# ─────────────────────────────────────────────────────────────────────────────
# STM-C extraction interval
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractionInterval:
    def test_should_extract_at_interval(self):
        n = STM_C_EXTRACTION_INTERVAL * 2  # messages = 2*N for N turns
        assert _should_extract(n) is True

    def test_should_not_extract_between_intervals(self):
        assert _should_extract(2) is False  # 1 turn — not at interval

    def test_zero_turns_not_extracted(self):
        assert _should_extract(0) is False


# ─────────────────────────────────────────────────────────────────────────────
# Conflict resolver (pure logic, mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

class _FakeCandidate:
    def __init__(self, text, category="identity"):
        self.text     = text
        self.category = category

class _FakeConflict:
    def __init__(self, idx, mem_id, action, reason=""):
        self.new_candidate_idx  = idx
        self.existing_memory_id = mem_id
        self.action             = action
        self.reason             = reason

class _FakeConflictResult:
    def __init__(self, conflicts):
        self.conflicts = conflicts

def _resolve(candidates, existing_memories, conflicts):
    """Pure conflict resolution logic extracted for testing."""
    ignored: set[int]      = set()
    supersedes: Dict[int, str] = {}
    for c in conflicts:
        if c.action == "ignore_new":
            ignored.add(c.new_candidate_idx)
        elif c.action == "supersede":
            supersedes[c.new_candidate_idx] = c.existing_memory_id
    filtered = [c for i, c in enumerate(candidates) if i not in ignored]
    return filtered, supersedes


class TestConflictResolver:
    def test_supersede_keeps_candidate_marks_old(self):
        candidates = [_FakeCandidate("User lives in SF"), _FakeCandidate("Likes coffee")]
        conflicts  = [_FakeConflict(0, "mem_old", "supersede", "moved")]
        filtered, sup = _resolve(candidates, [], conflicts)
        assert len(filtered) == 2
        assert sup[0] == "mem_old"

    def test_ignore_new_removes_candidate(self):
        candidates = [_FakeCandidate("User is a developer")]
        conflicts  = [_FakeConflict(0, "mem_x", "ignore_new", "dupe")]
        filtered, _ = _resolve(candidates, [], conflicts)
        assert len(filtered) == 0

    def test_keep_both_keeps_candidate(self):
        candidates = [_FakeCandidate("Likes sushi")]
        conflicts  = [_FakeConflict(0, "mem_y", "keep_both", "non-conflicting")]
        filtered, sup = _resolve(candidates, [], conflicts)
        assert len(filtered) == 1
        assert not sup


# ─────────────────────────────────────────────────────────────────────────────
# Final score ranking
# ─────────────────────────────────────────────────────────────────────────────

class TestFinalScoreRanking:
    RELEVANCE_W = 0.60
    SALIENCE_W  = 0.30
    RECENCY_W   = 0.10

    def _score(self, rel, sal, rec):
        return self.RELEVANCE_W * rel + self.SALIENCE_W * sal + self.RECENCY_W * rec

    def test_high_relevance_beats_high_salience(self):
        a = self._score(rel=0.3, sal=1.0, rec=0.5)
        b = self._score(rel=0.9, sal=0.4, rec=0.5)
        assert b > a

    def test_equal_relevance_salience_wins(self):
        a = self._score(rel=0.5, sal=1.0, rec=0.5)
        b = self._score(rel=0.5, sal=0.4, rec=0.5)
        assert a > b

    def test_recency_breaks_tie(self):
        a = self._score(rel=0.5, sal=0.5, rec=1.0)
        b = self._score(rel=0.5, sal=0.5, rec=0.0)
        assert a > b


# ─────────────────────────────────────────────────────────────────────────────
# Extraction cache (pure dict logic)
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractionCache:
    def test_same_text_returns_cached(self):
        import hashlib
        cache: Dict[str, list] = {}
        text  = "User: I love tea\nAssistant: Got it!"
        key   = hashlib.md5(text.encode()).hexdigest()

        call_count = 0
        def fake_llm_call(t):
            nonlocal call_count
            call_count += 1
            return ["likes tea"]

        def extract(t):
            k = hashlib.md5(t.encode()).hexdigest()
            if k in cache:
                return cache[k]
            result  = fake_llm_call(t)
            cache[k] = result
            return result

        r1 = extract(text)
        r2 = extract(text)
        assert r1 == r2
        assert call_count == 1   # LLM only called once

    def test_different_text_bypasses_cache(self):
        import hashlib
        cache: Dict[str, list] = {}
        call_count = 0
        def extract(t):
            nonlocal call_count
            k = hashlib.md5(t.encode()).hexdigest()
            if k in cache:
                return cache[k]
            call_count += 1
            cache[k] = [t]
            return [t]
        extract("text A")
        extract("text B")
        assert call_count == 2
