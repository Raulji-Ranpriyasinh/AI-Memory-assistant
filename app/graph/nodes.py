"""
LangGraph node functions.

Each node is a standalone function that receives (state, config, *, store)
and returns a state patch dict.

Fixes applied:
  - STM-C: interval-gated + intra-buffer dedup
  - LTM Gate: cosine pre-filter → per-candidate multi-query dedup → conflict resolution → write
  - Chat: token-budget-aware LTM block in system prompt
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

from app.config.settings import LTM_TOP_K
from app.memory.controller import MemoryController
from app.models.schemas import MemoryCandidate
from app.observability.metrics import metrics
from app.prompts.templates import CHAT_SYSTEM_PROMPT


def make_nodes(controller: MemoryController):
    """
    Factory: returns a dict of node functions bound to the given controller.
    This avoids class-level state and makes testing easier.
    """

    # ── STM-A ─────────────────────────────────────────────────────────────

    def stm_a_node(
        state: MessagesState, config: RunnableConfig, *, store: BaseStore
    ) -> dict:
        """Pass-through; window is extracted on demand in chat_node."""
        return {}

    # ── STM-B ─────────────────────────────────────────────────────────────

    def stm_b_node(
        state: MessagesState, config: RunnableConfig, *, store: BaseStore
    ) -> dict:
        """Update multi-layer summaries in Postgres."""
        cfg       = config["configurable"]
        user_id   = cfg["user_id"]
        thread_id = cfg["thread_id"]
        controller.update_summaries(store, user_id, thread_id, state["messages"])
        return {}

    # ── STM-C ─────────────────────────────────────────────────────────────

    def stm_c_node(
        state: MessagesState, config: RunnableConfig, *, store: BaseStore
    ) -> dict:
        """
        Extract typed memory candidates every N turns.
        Deduplicates new candidates against the existing Postgres buffer.
        """
        # Interval gate
        if not controller.should_extract(state["messages"]):
            return {}

        candidates = controller.extract_candidates_stm_c(state["messages"])
        if not candidates:
            return {}

        cfg       = config["configurable"]
        user_id   = cfg["user_id"]
        thread_id = cfg["thread_id"]
        ns        = ("candidates", user_id, thread_id)

        # Intra-buffer dedup: load existing texts from Postgres buffer
        existing_items = store.search(ns)
        existing_texts: set[str] = set()
        for item in existing_items:
            try:
                existing_texts.add(item.value.get("data", {}).get("text", ""))
            except Exception:
                continue

        added = 0
        for c in candidates:
            if c.text not in existing_texts:
                store.put(ns, str(uuid.uuid4()), {"data": c.model_dump()})
                existing_texts.add(c.text)
                added += 1

        metrics.log("stm_c_buffer", added=added, skipped=len(candidates) - added)
        return {}

    # ── LTM Gate ──────────────────────────────────────────────────────────

    def ltm_gate_node(
        state: MessagesState, config: RunnableConfig, *, store: BaseStore
    ) -> dict:
        """
        Full LTM write pipeline:
          1. Load buffered candidates from Postgres
          2. Cosine pre-filter (fast, no LLM)
          3. Per-candidate multi-query dedup fetch
          4. LLM scorer
          5. Conflict resolution (supersede / ignore)
          6. Write to Pinecone
          7. Clear Postgres buffer
        """
        cfg       = config["configurable"]
        user_id   = cfg["user_id"]
        thread_id = cfg["thread_id"]
        ns        = ("candidates", user_id, thread_id)

        candidate_items = store.search(ns)
        if not candidate_items:
            return {}

        candidates: List[MemoryCandidate] = []
        for item in candidate_items:
            try:
                candidates.append(MemoryCandidate(**item.value.get("data", {})))
            except Exception:
                continue

        if not candidates:
            _clear_buffer(store, ns, candidate_items)
            return {}

        # ── Step 2: cosine pre-filter ──────────────────────────────────────
        auto_dupes, needs_check = controller.ltm.check_duplicates_cosine(
            user_id, candidates
        )
        if auto_dupes:
            print(f"  🔁 {len(auto_dupes)} auto-deduplicated via cosine similarity")

        if not needs_check:
            _clear_buffer(store, ns, candidate_items)
            return {}

        # ── Step 3: per-candidate multi-query dedup ────────────────────────
        existing_texts = controller.get_existing_texts_for_candidates(
            user_id, needs_check
        )

        # ── Step 4: LLM scoring ────────────────────────────────────────────
        scored = controller.score_candidates(needs_check, existing_texts)
        if not scored:
            _clear_buffer(store, ns, candidate_items)
            return {}

        # ── Step 5: conflict resolution ────────────────────────────────────
        high_salience_candidates = [
            sc.candidate
            for sc in scored
            if sc.salience_score >= 0.6  # only run conflict check on plausible writers
        ]
        existing_mems = controller.get_ltm_memories_semantic(
            user_id, needs_check[0].text, top_k=20
        )
        filtered_candidates, supersedes_map = controller.conflict_resolver.resolve(
            high_salience_candidates, existing_mems
        )

        # Rebuild scored list honoring ignore_new decisions
        keep_texts: set[str] = {c.text for c in filtered_candidates}
        scored_filtered = [sc for sc in scored if sc.candidate.text in keep_texts]

        # Remap supersedes_map to indices in scored_filtered
        text_to_supersede: Dict[str, str] = {
            fc.text: supersedes_map.get(i, "")
            for i, fc in enumerate(high_salience_candidates)
        }
        remapped_supersedes: Dict[int, str] = {}
        for i, sc in enumerate(scored_filtered):
            s = text_to_supersede.get(sc.candidate.text, "")
            if s:
                remapped_supersedes[i] = s

        # ── Step 6: write to Pinecone ──────────────────────────────────────
        controller.write_to_ltm(user_id, scored_filtered, supersedes_map=remapped_supersedes)

        # ── Step 7: clear Postgres buffer ─────────────────────────────────
        _clear_buffer(store, ns, candidate_items)
        return {}

    # ── Chat ──────────────────────────────────────────────────────────────

    def chat_node(
        state: MessagesState, config: RunnableConfig, *, store: BaseStore
    ) -> dict:
        """
        Assemble context from all memory layers and generate a response.
        LTM block is automatically trimmed to token budget.
        """
        cfg       = config["configurable"]
        user_id   = cfg["user_id"]
        thread_id = cfg["thread_id"]

        # STM-A
        stm_a = controller.get_stm_a_raw(state["messages"])

        # STM-B
        stm_b          = controller.get_stm_b_summaries(store, user_id, thread_id)
        merged_summary = controller.merge_summaries(stm_b)
        summary_text   = (
            json.dumps(merged_summary.model_dump(), indent=2) if stm_b else "(none)"
        )

        # LTM — semantic retrieval on latest user message
        latest_user_msg = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        ltm_memories = controller.get_ltm_memories_semantic(
            user_id, latest_user_msg or "general context"
        )
        # Build LTM text block (already token-budgeted by search_memories)
        ltm_text = (
            "\n".join(
                f"[{m.get('category','fact')}] {m.get('text','')} "
                f"(salience={m.get('salience_score',0):.2f}, "
                f"relevance={m.get('relevance_score',0):.2f}, "
                f"recency={m.get('recency_score',0):.2f})"
                for m in ltm_memories
            )
            if ltm_memories
            else "(none)"
        )

        system_msg = SystemMessage(
            content=CHAT_SYSTEM_PROMPT.format(
                user_context=f"User ID: {user_id}",
                conversation_summaries=summary_text,
                ltm_content=ltm_text,
            )
        )
        response = controller.chat_llm.invoke([system_msg] + stm_a)
        metrics.log("chat_turn", user_id=user_id, ltm_hits=len(ltm_memories))
        return {"messages": [response]}

    # ── Helpers ───────────────────────────────────────────────────────────

    def _clear_buffer(store, ns, items) -> None:
        for item in items:
            try:
                store.delete(ns, item.key)
            except Exception:
                pass

    return {
        "stm_a_update":  stm_a_node,
        "stm_b_update":  stm_b_node,
        "stm_c_extract": stm_c_node,
        "ltm_gate":      ltm_gate_node,
        "chat":          chat_node,
    }
