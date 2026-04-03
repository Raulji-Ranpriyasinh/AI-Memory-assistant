"""
MemoryController — orchestrates STM-A/B/C + Pinecone LTM.

Improvements over original:
  - Tiered LLM strategy (cheap model for extraction/scoring, better for chat)
  - Extraction cache (md5-keyed, avoids re-extracting same recent window)
  - Multi-query dedup (each candidate checks its own semantic neighborhood)
  - Cosine pre-filter before LLM scorer
  - Conflict resolution integrated into LTM gate
  - Full metrics instrumentation
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.store.base import BaseStore

from app.config.settings import (
    CHAT_MODEL,
    CHAT_TEMPERATURE,
    LTM_SALIENCE_THRESHOLD,
    LTM_TOP_K,
    MEMORY_MODEL,
    MEMORY_TEMPERATURE,
    STM_A_WINDOW_SIZE,
    STM_C_EXTRACTION_INTERVAL,
    SUMMARY_LAYER_THRESHOLDS,
)
from app.memory.conflict import ConflictResolver
from app.memory.pinecone_ltm import PineconeLTMManager
from app.models.schemas import (
    CandidateExtractionResult,
    ConversationSummary,
    MemoryCandidate,
    MergedSummary,
    ScoredCandidate,
    ScoringResult,
)
from app.observability.metrics import metrics
from app.prompts.templates import (
    CANDIDATE_EXTRACTION_PROMPT,
    SCORING_PROMPT,
    SUMMARY_GENERATION_PROMPT,
    SUMMARY_MERGE_PROMPT,
)


class MemoryController:
    """Orchestrates all memory operations."""

    def __init__(self, db_uri: str) -> None:
        self.db_uri = db_uri

        # ── Tiered LLMs ───────────────────────────────────────────────────
        # Tier 1: cheap + fast → high-volume extraction / scoring
        self.memory_llm = ChatGoogleGenerativeAI(
            model=MEMORY_MODEL, temperature=MEMORY_TEMPERATURE
        )
        # Tier 2: better model → user-facing chat responses
        self.chat_llm = ChatGoogleGenerativeAI(
            model=CHAT_MODEL, temperature=CHAT_TEMPERATURE
        )

        # Structured extractors (all use cheap model)
        self.candidate_extractor = self.memory_llm.with_structured_output(
            CandidateExtractionResult
        )
        self.scorer    = self.memory_llm.with_structured_output(ScoringResult)
        self.summarizer = self.memory_llm.with_structured_output(ConversationSummary)
        self.merger     = self.memory_llm.with_structured_output(MergedSummary)

        # Sub-modules
        self.ltm              = PineconeLTMManager()
        self.conflict_resolver = ConflictResolver(self.memory_llm)

        # Simple in-process extraction cache (keyed by md5 of recent_text)
        self._extraction_cache: Dict[str, List[MemoryCandidate]] = {}

    # ── STM-A ─────────────────────────────────────────────────────────────

    def get_stm_a_raw(self, all_messages: List[BaseMessage]) -> List[BaseMessage]:
        return (
            all_messages[-STM_A_WINDOW_SIZE:]
            if len(all_messages) > STM_A_WINDOW_SIZE
            else all_messages
        )

    # ── STM-B ─────────────────────────────────────────────────────────────

    def get_stm_b_summaries(
        self, store: BaseStore, user_id: str, thread_id: str
    ) -> List[ConversationSummary]:
        ns    = ("summaries", user_id, thread_id)
        items = store.search(ns)
        summaries: list[ConversationSummary] = []
        for item in items:
            try:
                summaries.append(ConversationSummary(**item.value.get("data", {})))
            except Exception:
                continue
        return sorted(summaries, key=lambda s: s.timestamp)

    # ── STM-C extraction (cached, interval-gated) ─────────────────────────

    def should_extract(self, all_messages: List[BaseMessage]) -> bool:
        turn_count = len(all_messages) // 2
        return turn_count > 0 and turn_count % STM_C_EXTRACTION_INTERVAL == 0

    def extract_candidates_stm_c(
        self, messages: List[BaseMessage]
    ) -> List[MemoryCandidate]:
        if len(messages) < 2:
            return []

        recent_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in messages[-6:]
        )

        cache_key = hashlib.md5(recent_text.encode()).hexdigest()
        if cache_key in self._extraction_cache:
            metrics.log("extraction_cache_hit")
            return self._extraction_cache[cache_key]

        try:
            result: CandidateExtractionResult = self.candidate_extractor.invoke(
                [
                    SystemMessage(
                        content=CANDIDATE_EXTRACTION_PROMPT.format(
                            recent_messages=recent_text
                        )
                    ),
                    HumanMessage(content="Extract memory candidates."),
                ]
            )
            self._extraction_cache[cache_key] = result.candidates
            metrics.log("extraction_llm_call", count=len(result.candidates))
            return result.candidates
        except Exception as e:
            print(f"⚠️  Candidate extraction failed: {e}")
            return []

    # ── Scoring ───────────────────────────────────────────────────────────

    def score_candidates(
        self,
        candidates: List[MemoryCandidate],
        existing_memories: List[str],
    ) -> List[ScoredCandidate]:
        if not candidates:
            return []

        candidates_text = "\n".join(
            f"{i+1}. [{c.category}] {c.text}" for i, c in enumerate(candidates)
        )
        existing_text = "\n".join(existing_memories) if existing_memories else "(none)"

        try:
            result: ScoringResult = self.scorer.invoke(
                [
                    SystemMessage(
                        content=SCORING_PROMPT.format(
                            candidates=candidates_text,
                            existing_memories=existing_text,
                        )
                    ),
                    HumanMessage(content="Score these candidates."),
                ]
            )
            metrics.log("scoring_llm_call", count=len(result.scored_candidates))
            return result.scored_candidates
        except Exception as e:
            print(f"⚠️  Candidate scoring failed: {e}")
            return []

    # ── LTM helpers ───────────────────────────────────────────────────────

    def write_to_ltm(
        self,
        user_id: str,
        scored_candidates: List[ScoredCandidate],
        supersedes_map: Dict[int, str] | None = None,
    ) -> None:
        written = self.ltm.write_memories_batch(
            user_id, scored_candidates, supersedes_map=supersedes_map or {}
        )
        if written:
            print(f"  💾 Wrote {len(written)} memories to Pinecone LTM.")

    def get_ltm_memories_semantic(
        self,
        user_id: str,
        query: str,
        top_k: int = LTM_TOP_K,
    ) -> List[Dict[str, Any]]:
        return self.ltm.search_memories(user_id, query, top_k=top_k)

    def get_all_ltm_memories(self, user_id: str) -> List[Dict[str, Any]]:
        return self.ltm.list_all_memories(user_id)

    # ── Multi-query dedup retrieval ───────────────────────────────────────

    def get_existing_texts_for_candidates(
        self, user_id: str, candidates: List[MemoryCandidate]
    ) -> List[str]:
        """
        Query Pinecone once per candidate so each gets its own semantic
        neighbourhood — not a single shared query for all candidates.
        """
        all_texts: set[str] = set()
        for candidate in candidates:
            existing = self.get_ltm_memories_semantic(
                user_id, candidate.text, top_k=10
            )
            for m in existing:
                all_texts.add(m.get("text", ""))
        return list(all_texts)

    # ── Summarization ─────────────────────────────────────────────────────

    def generate_summary(
        self,
        messages: List[BaseMessage],
        turn_range: str,
        previous_summary: Optional[ConversationSummary] = None,
    ) -> ConversationSummary:
        messages_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in messages
        )
        prev_text = (
            json.dumps(previous_summary.model_dump(), indent=2)
            if previous_summary
            else ""
        )
        try:
            summary: ConversationSummary = self.summarizer.invoke(
                [
                    SystemMessage(
                        content=SUMMARY_GENERATION_PROMPT.format(
                            messages=messages_text,
                            previous_summary=prev_text,
                        )
                    ),
                    HumanMessage(content="Generate summary."),
                ]
            )
            summary.turn_range = turn_range
            metrics.log("summary_generated", turn_range=turn_range)
            return summary
        except Exception as e:
            print(f"⚠️  Summary generation failed: {e}")
            return ConversationSummary(turn_range=turn_range)

    def merge_summaries(self, summaries: List[ConversationSummary]) -> MergedSummary:
        if not summaries:
            return MergedSummary()

        summaries_text = "\n\n".join(
            f"Summary ({s.turn_range}):\n{json.dumps(s.model_dump(), indent=2)}"
            for s in summaries
        )
        try:
            merged: MergedSummary = self.merger.invoke(
                [
                    SystemMessage(
                        content=SUMMARY_MERGE_PROMPT.format(summaries=summaries_text)
                    ),
                    HumanMessage(content="Merge summaries."),
                ]
            )
            return merged
        except Exception as e:
            print(f"⚠️  Summary merge failed: {e}")
            return MergedSummary()

    def update_summaries(
        self,
        store: BaseStore,
        user_id: str,
        thread_id: str,
        all_messages: List[BaseMessage],
    ) -> None:
        total_turns = len(all_messages) // 2
        for threshold in SUMMARY_LAYER_THRESHOLDS:
            if total_turns >= threshold:
                ns       = ("summaries", user_id, thread_id)
                existing = store.search(ns)
                has_summary = any(
                    str(threshold) in item.value.get("data", {}).get("turn_range", "")
                    for item in existing
                )
                if not has_summary:
                    start_idx = max(0, threshold - 20) * 2
                    end_idx   = threshold * 2
                    segment   = all_messages[start_idx:end_idx]
                    if segment:
                        summary = self.generate_summary(
                            segment, f"turns {threshold-20}-{threshold}"
                        )
                        store.put(
                            ns, f"summary_{threshold}", {"data": summary.model_dump()}
                        )
