"""
MultiLayerChatbot — public API surface.

Wraps the LangGraph graph, MemoryController, and Postgres persistence.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from app.config.settings import DB_URI
from app.graph.builder import build_graph
from app.memory.controller import MemoryController
from app.models.schemas import ConversationSummary
from app.observability.metrics import metrics
from app.security.isolation import validate_user_id


class MultiLayerChatbot:
    """Main chatbot with multi-layer memory (STM-A/B/C + Pinecone LTM)."""

    def __init__(self, user_id: str = "default_user", db_uri: str = DB_URI) -> None:
        self.user_id    = validate_user_id(user_id)
        self.db_uri     = db_uri
        self.controller = MemoryController(db_uri)
        self._builder   = build_graph(self.controller)

        # One-time DB schema setup
        with PostgresSaver.from_conn_string(self.db_uri) as cp:
            cp.setup()
        with PostgresStore.from_conn_string(self.db_uri) as st:
            st.setup()

    # ── Configurable context ──────────────────────────────────────────────

    @property
    def _config(self) -> dict:
        return {
            "configurable": {
                "user_id":   self.user_id,
                "thread_id": f"thread-{self.user_id}",
            }
        }

    # ── Core chat ─────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """Send a message and return the assistant's reply."""
        with PostgresSaver.from_conn_string(self.db_uri) as checkpointer:
            with PostgresStore.from_conn_string(self.db_uri) as store:
                graph  = self._builder.compile(checkpointer=checkpointer, store=store)
                result = graph.invoke(
                    {"messages": [HumanMessage(content=user_message)]},
                    self._config,
                )
                return result["messages"][-1].content

    # ── Memory inspection ─────────────────────────────────────────────────

    def get_memories(self) -> Dict[str, Any]:
        """Return all LTM memories + conversation summaries."""
        ltm       = self.controller.get_all_ltm_memories(self.user_id)
        thread_id = f"thread-{self.user_id}"
        with PostgresStore.from_conn_string(self.db_uri) as store:
            summaries = self.controller.get_stm_b_summaries(
                store, self.user_id, thread_id
            )
        return {
            "ltm":       ltm,
            "summaries": [s.model_dump() for s in summaries],
        }

    def search_memories(
        self,
        query: str,
        top_k: int = 10,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over LTM memories."""
        return self.controller.ltm.search_memories(
            self.user_id, query, top_k=top_k, category_filter=category
        )

    def get_conversation_history(self) -> List[dict]:
        """Retrieve recent conversation history (STM-A window)."""
        with PostgresSaver.from_conn_string(self.db_uri) as checkpointer:
            with PostgresStore.from_conn_string(self.db_uri) as store:
                graph = self._builder.compile(checkpointer=checkpointer, store=store)
                try:
                    state    = graph.get_state(self._config)
                    messages = state.values.get("messages", [])
                    stm_a    = self.controller.get_stm_a_raw(messages)
                    history: list[dict] = []
                    for msg in stm_a:
                        if isinstance(msg, HumanMessage):
                            history.append({"role": "user", "content": msg.content})
                        elif isinstance(msg, AIMessage):
                            history.append({"role": "assistant", "content": msg.content})
                    return history
                except Exception:
                    return []

    # ── Maintenance ───────────────────────────────────────────────────────

    def prune_stale_memories(self) -> int:
        """Run decay + pruning. Returns number of deleted memories."""
        pruned = self.controller.ltm.decay_and_prune(self.user_id)
        print(f"  🗑️  Pruned {pruned} stale memories.")
        return pruned

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregated observability metrics."""
        return metrics.summary()

    def delete_memory(self, memory_id: str) -> None:
        """Delete a specific memory (ownership verified)."""
        self.controller.ltm.delete_memory(memory_id, self.user_id)

    def delete_all_memories(self) -> None:
        """Delete ALL memories for this user. Irreversible."""
        self.controller.ltm.delete_user_memories(self.user_id)
