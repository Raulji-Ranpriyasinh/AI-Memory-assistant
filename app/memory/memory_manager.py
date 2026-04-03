# app/memory/memory_manager.py
from typing import Optional, Dict, List, Any
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory
from app.memory.scoring import score_importance
from app.memory.backends import InMemorySTMBackend, RedisSTMBackend
from app.memory.schema import MemoryEntry
from app.config.settings import settings  # ✅ import config
from datetime import datetime
import threading
import time


class MemoryManager:
    def __init__(
        self,
        stm_backend: str = settings.STM_BACKEND,
        stm_ttl_minutes: int = settings.STM_TTL_MINUTES,
        redis_url: str = settings.REDIS_URL,
        collection_name: str = settings.LTM_COLLECTION_NAME,
        embedding_model: str = settings.LTM_EMBEDDING_MODEL,
        enable_cleanup: bool = settings.ENABLE_CLEANUP,
        cleanup_interval_minutes: int = settings.CLEANUP_INTERVAL_MINUTES,
        # 🔹 New optional hooks
        on_memory_add=None,
        on_memory_recall=None,
        on_stm_promote=None,
    ):
        if stm_backend == "redis":
            backend = RedisSTMBackend(redis_url, stm_ttl_minutes)
        else:
            backend = InMemorySTMBackend(stm_ttl_minutes)

        self.stm = ShortTermMemory(backend)
        self.ltm = LongTermMemory(
            collection_name=collection_name,
            embedding_model=embedding_model,
            host=settings.LTM_QDRANT_HOST,
            port=settings.LTM_QDRANT_PORT,
            vector_size=settings.LTM_VECTOR_SIZE,
        )

        # Store hooks
        self.on_memory_add = on_memory_add
        self.on_memory_recall = on_memory_recall
        self.on_stm_promote = on_stm_promote

        if enable_cleanup:
            self._start_cleanup_thread(cleanup_interval_minutes)

    def _start_cleanup_thread(self, interval_minutes: int):
        def cleanup_worker():
            while True:
                time.sleep(interval_minutes * 60)
                try:
                    self.stm.cleanup_expired()
                except Exception as e:
                    print(f"Cleanup error: {e}")

        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()

    # STM methods (same interface as before)
    def set_short_term(self, session_id: str, key: str, value: str):
        self.stm.set(session_id, key, value)
        if callable(self.on_memory_add):
            self.on_memory_add(
                "short_term", {"session_id": session_id, "key": key, "value": value}
            )

    def get_short_term(self, session_id: str, key: str) -> str:
        return self.stm.get(session_id, key)

    def get_all_short_term(self, session_id: str) -> Dict[str, str]:
        return self.stm.get_all(session_id)

    def clear_short_term(self, session_id: str):
        self.stm.clear(session_id)

    #  LTM methods

    def add_long_term(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
        conversation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        metadata = metadata or {}
        if importance is None:
            importance = score_importance(text, context)
        metadata["importance"] = importance

        result = self.ltm.add_entry(user_id, text, metadata, conversation_id)

        if callable(self.on_memory_add) and result:
            self.on_memory_add(
                "long_term", {"user_id": user_id, "text": text, "metadata": metadata}
            )

        return result

    def search_long_term(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.3,
    ) -> List[MemoryEntry]:
        search_filters = filters or {}
        if user_id:
            search_filters["user_id"] = user_id

        ltm_results = self.ltm.search(
            query_text=query, filters=search_filters, top_k=top_k, min_score=min_score
        )

        return [
            MemoryEntry(
                id=entry.id,
                user_id=entry.user_id,
                text=entry.text,
                metadata=entry.metadata,
                timestamp=entry.timestamp,
                source="long_term",
                importance=entry.metadata.get("importance", 0.5),
            )
            for entry in ltm_results
        ]

    #  recall with conversation awareness
    def recall(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MemoryEntry]:
        stm_entries: List[MemoryEntry] = []

        # Check STM first
        if session_id:
            stm_data = self.stm.get_all(session_id)
            for key, value in stm_data.items():
                if query.lower() in value.lower():
                    stm_entries.append(
                        MemoryEntry(
                            id=None,
                            user_id=user_id,
                            text=value,
                            metadata={"key": key},
                            timestamp=datetime.now(),
                            source="short_term",
                            importance=0.5,
                        )
                    )
            if len(stm_entries) >= top_k:
                results = stm_entries[:top_k]
                if callable(self.on_memory_recall):
                    self.on_memory_recall(
                        {"user_id": user_id, "query": query, "results": results}
                    )
                return results

        # Search LTM with conversation context
        search_filters = {}
        if conversation_id:
            search_filters["conversation_id"] = conversation_id

        ltm_entries = self.search_long_term(
            query=query, user_id=user_id, top_k=top_k, filters=search_filters
        )

        # Combine and deduplicate
        combined = stm_entries + [
            e for e in ltm_entries if e.text not in {x.text for x in stm_entries}
        ]
        results = combined[:top_k]

        # 🔹 Hook call here
        if callable(self.on_memory_recall):
            self.on_memory_recall(
                {"user_id": user_id, "query": query, "results": results}
            )

        return results

    def promote_stm_to_ltm(
        self,
        session_id: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        min_importance: float = 0.3,
    ) -> Optional[str]:
        stm_data = self.stm.get_all(session_id)
        if not stm_data:
            return None

        combined_text = "\n".join([f"{k}: {v}" for k, v in stm_data.items()])
        context = {"conversation_length": len(stm_data)}
        score = score_importance(combined_text, context)

        if score >= min_importance:
            ltm_id = self.add_long_term(
                user_id=user_id,
                text=combined_text,
                metadata={"source": "stm_promotion"},
                importance=score,
                conversation_id=conversation_id,
                context=context,
            )
            self.clear_short_term(session_id)

            # 🔹 Hook call here
            if callable(self.on_stm_promote) and ltm_id:
                self.on_stm_promote({"session_id": session_id, "ltm_id": ltm_id})

            return ltm_id

        return None
