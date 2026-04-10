"""
PineconeLTMManager — Long-Term Memory backed by Pinecone.

Fixes applied vs. original:
  1. Weighted final_score (relevance + salience + recency) — no more salience-dominance.
  2. list_all_memories uses list+fetch (no zero-vector hack).
  3. decay_and_prune: time-based forgetting.
  4. boost_accessed_memories: reinforcement on retrieval.
  5. check_duplicates_cosine: fast cosine pre-filter before LLM scorer.
  6. build_pinecone_filter from security module — user isolation always enforced.
  7. select_memories_with_budget: token-budget-aware trimming.
  8. Full metrics instrumentation.
"""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from app.config.settings import (
    ACCESS_BOOST,
    CHARS_PER_TOKEN,
    COSINE_DEDUP_THRESHOLD,
    DECAY_HALF_LIFE_DAYS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    LTM_SALIENCE_THRESHOLD,
    LTM_TOKEN_BUDGET,
    LTM_TOP_K,
    MAX_MEMORIES_PER_USER,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX,
    PINECONE_REGION,
    PRUNE_THRESHOLD,
    RECENCY_WEIGHT,
    RELEVANCE_WEIGHT,
    SALIENCE_WEIGHT,
)
from app.models.schemas import MemoryCandidate, ScoredCandidate
from app.observability.metrics import metrics
from app.security.isolation import (
    assert_memory_owner,
    build_pinecone_filter,
    validate_user_id,
)


class PineconeLTMManager:
    """All Long-Term Memory I/O via Pinecone."""

    def __init__(self) -> None:
        if not PINECONE_API_KEY:
            raise ValueError(
                "PINECONE_API_KEY not set. Add it to your .env file."
            )

        print("🔧 Loading embedding model …")
        self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("✅ Embedding model ready.")

        self._pc    = Pinecone(api_key=PINECONE_API_KEY)
        self._index = self._get_or_create_index()

    # ── Index management ──────────────────────────────────────────────────

    def _get_or_create_index(self):
        existing = [idx.name for idx in self._pc.list_indexes()]
        if PINECONE_INDEX not in existing:
            print(f"📦 Creating Pinecone index '{PINECONE_INDEX}' …")
            self._pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
            print(f"✅ Index '{PINECONE_INDEX}' created.")
        else:
            print(f"✅ Connected to Pinecone index '{PINECONE_INDEX}'.")
        return self._pc.Index(PINECONE_INDEX)

    # ── Embeddings ────────────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        return self._embedder.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.encode(texts, convert_to_numpy=True).tolist()

    # ── Recency scoring ───────────────────────────────────────────────────

    @staticmethod
    def _recency_score(timestamp_str: str, half_life_days: float = DECAY_HALF_LIFE_DAYS) -> float:
        """Exponential decay: 1.0 when fresh, ~0.5 after half_life_days."""
        try:
            ts       = datetime.fromisoformat(timestamp_str)
            age_days = (datetime.now() - ts).total_seconds() / 86_400
            return math.exp(-0.693 * age_days / half_life_days)
        except Exception:
            return 0.5

    # ── Write ─────────────────────────────────────────────────────────────

    def write_memory(
        self,
        user_id: str,
        text: str,
        category: str,
        salience_score: float,
        context: Optional[str] = None,
        timestamp: Optional[str] = None,
        supersedes: str = "",
    ) -> str:
        validate_user_id(user_id)
        memory_id = f"{user_id}__{uuid.uuid4()}"
        vector    = self.embed(text)
        ts        = timestamp or datetime.now().isoformat()

        metadata: Dict[str, Any] = {
            "user_id":        user_id,
            "text":           text,
            "category":       category,
            "salience_score": salience_score,
            "context":        context or "",
            "timestamp":      ts,
            "version":        1,
            "supersedes":     supersedes,
            "last_accessed":  ts,
        }
        self._index.upsert(vectors=[(memory_id, vector, metadata)])
        metrics.log("ltm_write", user_id=user_id, category=category)
        return memory_id

    def write_memories_batch(
        self,
        user_id: str,
        scored_candidates: List[ScoredCandidate],
        supersedes_map: Dict[int, str] | None = None,
    ) -> List[str]:
        """
        Batch-upsert high-salience, non-duplicate candidates.
        supersedes_map: {candidate_index -> existing_memory_id_to_delete}
        """
        validate_user_id(user_id)
        supersedes_map = supersedes_map or {}

        to_write = [
            (i, sc) for i, sc in enumerate(scored_candidates)
            if sc.salience_score >= LTM_SALIENCE_THRESHOLD and not sc.is_duplicate
        ]
        if not to_write:
            metrics.log("ltm_batch_write_skipped", user_id=user_id, reason="no_candidates")
            return []

        texts   = [sc.candidate.text for _, sc in to_write]
        vectors = self.embed_batch(texts)

        upsert_payload: list[tuple] = []
        written_ids: list[str]      = []
        ids_to_delete: list[str]    = []

        for (orig_idx, sc), vec in zip(to_write, vectors):
            memory_id    = f"{user_id}__{uuid.uuid4()}"
            supersedes   = supersedes_map.get(orig_idx, "")
            if supersedes:
                ids_to_delete.append(supersedes)

            written_ids.append(memory_id)
            upsert_payload.append(
                (
                    memory_id,
                    vec,
                    {
                        "user_id":        user_id,
                        "text":           sc.candidate.text,
                        "category":       sc.candidate.category,
                        "salience_score": sc.salience_score,
                        "context":        sc.candidate.context or "",
                        "timestamp":      sc.candidate.timestamp,
                        "version":        1,
                        "supersedes":     supersedes,
                        "last_accessed":  datetime.now().isoformat(),
                    },
                )
            )

        # Delete superseded memories first
        if ids_to_delete:
            for i in range(0, len(ids_to_delete), 100):
                self._index.delete(ids=ids_to_delete[i : i + 100])
            metrics.log("ltm_supersede", user_id=user_id, count=len(ids_to_delete))

        # Upsert in batches of 100
        for i in range(0, len(upsert_payload), 100):
            self._index.upsert(vectors=upsert_payload[i : i + 100])

        metrics.log("ltm_batch_write", user_id=user_id, count=len(written_ids))
        return written_ids

    # ── Retrieval ─────────────────────────────────────────────────────────

    def search_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = LTM_TOP_K,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with weighted final_score = relevance + salience + recency.
        Applies token budget trimming before returning.
        """
        validate_user_id(user_id)
        t0        = time.monotonic()
        query_vec = self.embed(query)

        extra = {}
        if category_filter:
            extra["category"] = {"$eq": category_filter}
        filt = build_pinecone_filter(user_id, extra)

        response  = self._index.query(
            vector=query_vec,
            top_k=top_k,
            filter=filt,
            include_metadata=True,
        )

        memories: List[Dict[str, Any]] = []
        for match in response.matches:
            meta    = match.metadata or {}
            recency = self._recency_score(meta.get("timestamp", ""))
            sal     = meta.get("salience_score", 0.0)
            rel     = match.score

            final_score = (
                RELEVANCE_WEIGHT * rel
                + SALIENCE_WEIGHT  * sal
                + RECENCY_WEIGHT   * recency
            )
            memories.append(
                {
                    "id":              match.id,
                    "text":            meta.get("text", ""),
                    "category":        meta.get("category", "fact"),
                    "salience_score":  sal,
                    "context":         meta.get("context", ""),
                    "timestamp":       meta.get("timestamp", ""),
                    "relevance_score": rel,
                    "recency_score":   recency,
                    "final_score":     final_score,
                }
            )

        memories.sort(key=lambda m: m["final_score"], reverse=True)

        # Boost accessed memories (non-blocking best-effort)
        if memories:
            self._boost_accessed_memories([m["id"] for m in memories[:5]], user_id)

        latency_ms = (time.monotonic() - t0) * 1000
        metrics.log(
            "ltm_retrieval",
            user_id=user_id,
            hits=len(memories),
            latency_ms=round(latency_ms, 1),
        )

        return self.select_memories_with_budget(memories, LTM_TOKEN_BUDGET)

    # ── Token-budget trimming ─────────────────────────────────────────────

    @staticmethod
    def select_memories_with_budget(
        memories: List[Dict[str, Any]],
        max_tokens: int = LTM_TOKEN_BUDGET,
    ) -> List[Dict[str, Any]]:
        """
        Return the highest-ranked subset whose combined text fits within
        max_tokens (estimated at CHARS_PER_TOKEN chars/token).
        Memories are already sorted by final_score descending.
        """
        selected: list[Dict[str, Any]] = []
        used_tokens = 0
        max_chars   = max_tokens * CHARS_PER_TOKEN

        for mem in memories:
            text_len = len(mem.get("text", "")) + len(mem.get("context", ""))
            if used_tokens + text_len > max_chars:
                break
            selected.append(mem)
            used_tokens += text_len

        metrics.log(
            "ltm_budget_trim",
            original=len(memories),
            selected=len(selected),
            budget_tokens=max_tokens,
        )
        return selected

    # ── List all memories ─────────────────────────────────────────────────

    def list_all_memories(
        self, user_id: str, limit: int = MAX_MEMORIES_PER_USER
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all memories using Pinecone's list+fetch API.
        Replaces the zero-vector hack; uses the <user_id>__ ID prefix.
        """
        validate_user_id(user_id)
        memories: list[Dict[str, Any]] = []
        prefix = f"{user_id}__"
        fetched_count = 0

        for ids_batch in self._index.list(prefix=prefix):
            if not ids_batch or fetched_count >= limit:
                break
            batch = ids_batch[: limit - fetched_count]
            fetched = self._index.fetch(ids=batch)
            for vid, vec_data in fetched.vectors.items():
                meta = vec_data.metadata or {}
                # Security double-check
                if meta.get("user_id") != user_id:
                    continue
                memories.append(
                    {
                        "id":             vid,
                        "text":           meta.get("text", ""),
                        "category":       meta.get("category", "fact"),
                        "salience_score": meta.get("salience_score", 0.0),
                        "context":        meta.get("context", ""),
                        "timestamp":      meta.get("timestamp", ""),
                    }
                )
            fetched_count += len(batch)

        memories.sort(key=lambda m: m["salience_score"], reverse=True)
        return memories

    # ── Cosine dedup pre-filter ───────────────────────────────────────────

    def check_duplicates_cosine(
        self, user_id: str, candidates: List[MemoryCandidate]
    ) -> tuple[List[MemoryCandidate], List[MemoryCandidate]]:
        """
        Split candidates into (auto_duplicates, needs_llm_check).
        Candidates with cosine sim >= COSINE_DEDUP_THRESHOLD are auto-dropped.
        """
        validate_user_id(user_id)
        filt       = build_pinecone_filter(user_id)
        auto_dupes: list[MemoryCandidate] = []
        needs_check: list[MemoryCandidate] = []

        for candidate in candidates:
            vec     = self.embed(candidate.text)
            results = self._index.query(
                vector=vec,
                top_k=3,
                filter=filt,
                include_metadata=True,
            )
            max_sim = max((m.score for m in results.matches), default=0.0)
            if max_sim >= COSINE_DEDUP_THRESHOLD:
                auto_dupes.append(candidate)
            else:
                needs_check.append(candidate)

        metrics.log(
            "dedup_cosine",
            user_id=user_id,
            auto_dupes=len(auto_dupes),
            needs_llm=len(needs_check),
        )
        return auto_dupes, needs_check

    # ── Decay & pruning ───────────────────────────────────────────────────

    def decay_and_prune(self, user_id: str) -> int:
        """
        Apply exponential decay to salience scores.
        Delete memories whose effective score drops below PRUNE_THRESHOLD.
        Returns the number of pruned memories.
        """
        validate_user_id(user_id)
        all_mems   = self.list_all_memories(user_id)
        to_delete: list[str] = []

        for mem in all_mems:
            try:
                ts       = datetime.fromisoformat(mem["timestamp"])
                age_days = (datetime.now() - ts).total_seconds() / 86_400
                decay    = math.exp(-0.693 * age_days / DECAY_HALF_LIFE_DAYS)
                if mem["salience_score"] * decay < PRUNE_THRESHOLD:
                    to_delete.append(mem["id"])
            except Exception:
                continue

        if to_delete:
            for i in range(0, len(to_delete), 100):
                self._index.delete(ids=to_delete[i : i + 100])

        metrics.log("ltm_prune", user_id=user_id, pruned=len(to_delete))
        return len(to_delete)

    # ── Access-boost ──────────────────────────────────────────────────────

    def _boost_accessed_memories(self, memory_ids: List[str], user_id: str) -> None:
        """Increment salience for recently accessed memories (best-effort)."""
        for mid in memory_ids:
            try:
                result = self._index.fetch(ids=[mid])
                if mid not in result.vectors:
                    continue
                vec_data = result.vectors[mid]
                meta     = dict(vec_data.metadata or {})

                # Security: skip if the fetched vector doesn't belong to user
                if meta.get("user_id") != user_id:
                    continue

                meta["salience_score"] = min(
                    1.0, meta.get("salience_score", 0.5) + ACCESS_BOOST
                )
                meta["last_accessed"] = datetime.now().isoformat()
                self._index.upsert(vectors=[(mid, vec_data.values, meta)])
            except Exception:
                pass  # Never let boost errors crash retrieval

    # ── Delete helpers ────────────────────────────────────────────────────

    def delete_memory(self, memory_id: str, requesting_user_id: str) -> None:
        """Delete a single memory. Validates ownership before deleting."""
        result = self._index.fetch(ids=[memory_id])
        if memory_id in result.vectors:
            owner = result.vectors[memory_id].metadata.get("user_id", "")
            assert_memory_owner(owner, requesting_user_id)
        self._index.delete(ids=[memory_id])
        metrics.log("ltm_delete_single", user_id=requesting_user_id)

    def delete_user_memories(self, user_id: str) -> None:
        """Delete ALL memories for a user."""
        validate_user_id(user_id)
        filt = build_pinecone_filter(user_id)
        self._index.delete(filter=filt)
        metrics.log("ltm_delete_all", user_id=user_id)
