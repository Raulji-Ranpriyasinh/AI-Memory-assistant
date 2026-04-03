# app/memory/long_term.py

from typing import Optional, Dict, Any, List
from utils.logger import logger
from app.memory.backends import QdrantLTMBackend, LTMBackend
from app.memory.schema import LongTermMemoryEntry
from app.config.settings import settings
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime, timedelta


class LongTermMemory:
    ...

    def export_json(self, filepath: str):
        """Export all LTM entries to JSON."""
        # This assumes backend can return all stored entries
        # For Qdrant, we need to scroll through points
        all_entries = self.backend.export_all()
        Path(filepath).write_text(
            json.dumps([e.model_dump() for e in all_entries], default=str, indent=2)
        )

    def import_json(self, filepath: str):
        """Import LTM entries from JSON."""
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(filepath)
        entries = json.loads(file_path.read_text())
        for entry_dict in entries:
            entry = LongTermMemoryEntry(**entry_dict)
            self.backend.add_entry(
                user_id=entry.user_id,
                text=entry.text,
                metadata=entry.metadata,
                conversation_id=entry.metadata.get("conversation_id"),
            )


class LongTermMemory:
    """
    A wrapper around an LTM backend (default: Qdrant) that adds:
    - Deduplication before insertion
    - Optional conversation-aware filtering
    """

    def __init__(
        self,
        backend: Optional[LTMBackend] = None,
        host: str = settings.LTM_QDRANT_HOST,
        port: int = settings.LTM_QDRANT_PORT,
        collection_name: str = settings.LTM_COLLECTION_NAME,
        embedding_model: str = settings.LTM_EMBEDDING_MODEL,
        vector_size: int = settings.LTM_VECTOR_SIZE,
    ):
        # Use provided backend or default to Qdrant
        self.backend = backend or QdrantLTMBackend(
            host=host,
            port=port,
            collection_name=collection_name,
            embedding_model=embedding_model,
            vector_size=vector_size,
        )

    def add_entry(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        # Deduplication
        search_filters = {"user_id": user_id}
        if conversation_id:
            search_filters["conversation_id"] = conversation_id

        existing_entries = self.search(query_text=text, top_k=3, filters=search_filters)

        # New embedding
        new_embedding = self.backend.model.encode(text).reshape(
            1, -1
        )  # 2D array for sklearn

        for e in existing_entries:
            if e.embedding is not None:
                existing_vec = np.array(e.embedding).reshape(1, -1)
                sim = cosine_similarity(new_embedding, existing_vec)[0][0]
                if sim >= settings.DEDUPLICATION_THRESHOLD:
                    logger.info(
                        f"Similar entry found (cosine sim={sim:.3f}). Skipping: {text[:50]}..."
                    )
                    return None

        # If no near-duplicate found, store
        return self.backend.add_entry(user_id, text, metadata, conversation_id)

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = settings.MIN_SEARCH_SCORE,
    ) -> List[LongTermMemoryEntry]:
        return self.backend.search(
            query_text, top_k=top_k, filters=filters, min_score=min_score
        )

    def _texts_similar(self, text1: str, text2: str, threshold: float = 0.95) -> bool:
        """Basic text similarity check using Jaccard similarity."""
        text1_clean = text1.strip().lower()
        text2_clean = text2.strip().lower()

        if text1_clean == text2_clean:
            return True

        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())

        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union >= threshold

    def summarize_old_memories(self, user_id: str, days_old: int = 30) -> Optional[str]:
        """Placeholder for summarizing old memories."""
        return None

    def export_json(self, filepath: str):
        """Export all LTM entries to JSON."""
        # This assumes backend can return all stored entries
        # For Qdrant, we need to scroll through points
        all_entries = self.backend.export_all()
        Path(filepath).write_text(
            json.dumps([e.model_dump() for e in all_entries], default=str, indent=2)
        )

    def import_json(self, filepath: str):
        """Import LTM entries from JSON."""
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(filepath)
        entries = json.loads(file_path.read_text())
        for entry_dict in entries:
            entry = LongTermMemoryEntry(**entry_dict)
            self.backend.add_entry(
                user_id=entry.user_id,
                text=entry.text,
                metadata=entry.metadata,
                conversation_id=entry.metadata.get("conversation_id"),
            )

    def summarize_old_memories(
        self, user_id: str, days_old: int = settings.LTM_SUMMARIZATION_DAYS
    ) -> Optional[str]:
        """
        Summarizes old memories older than `days_old`.
        Uses a placeholder for LLM summarization - can be replaced with OpenAI/Ollama.
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        all_entries = self.backend.export_all()

        old_entries = [
            e for e in all_entries if e.user_id == user_id and e.timestamp < cutoff_date
        ]
        if not old_entries:
            logger.info("No old memories found for summarization.")
            return None

        # Placeholder summarization logic
        combined_text = "\n".join([e.text for e in old_entries])
        summary_text = (
            f"Summary of {len(old_entries)} old memories:\n{combined_text[:500]}..."
        )

        # Store summary in LTM
        summary_id = self.add_entry(
            user_id=user_id,
            text=summary_text,
            metadata={"type": "summary", "source_entries": [e.id for e in old_entries]},
        )

        # Optionally delete old memories
        if settings.LTM_PRUNE_AFTER_SUMMARY:
            self.backend.delete_entries([e.id for e in old_entries])

        logger.info(f"Summarized {len(old_entries)} memories into {summary_id}")
        return summary_id


'''
    def summarize_old_memories(self, user_id: str, days_old: int = settings.LTM_SUMMARIZATION_DAYS) -> Optional[str]:
        """
        Summarizes old memories using Gemini API, falls back to Ollama.
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        all_entries = self.backend.export_all()

        old_entries = [e for e in all_entries if e.user_id == user_id and e.timestamp < cutoff_date]
        if not old_entries:
            logger.info("No old memories found for summarization.")
            return None

        combined_text = "\n".join([e.text for e in old_entries])

        # Try Gemini first
        summary_text = None
        if settings.GEMINI_API_KEY:
            try:
                summary_text = self._summarize_with_gemini(combined_text)
                logger.info("Summarization done with Gemini API.")
            except Exception as e:
                logger.warning(f"Gemini summarization failed: {e}")

        # Fallback to Ollama
        if not summary_text:
            try:
                summary_text = self._summarize_with_ollama(combined_text)
                logger.info("Summarization done with Ollama fallback.")
            except Exception as e:
                logger.error(f"Ollama summarization failed: {e}")
                return None

        # Store summary in LTM
        summary_id = self.add_entry(
            user_id=user_id,
            text=summary_text,
            metadata={
                "type": "summary",
                "source_entries": [e.id for e in old_entries],
                "generated_by": "gemini" if settings.GEMINI_API_KEY else "ollama"
            },
        )

        # Optionally delete old memories
        if settings.LTM_PRUNE_AFTER_SUMMARY:
            self.backend.delete_entries([e.id for e in old_entries])

        logger.info(f"Summarized {len(old_entries)} memories into {summary_id}")
        return summary_id

    def _summarize_with_gemini(self, text: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": f"Summarize the following memories:\n{text}"}]}]}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _summarize_with_ollama(self, text: str) -> str:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": f"Summarize the following memories:\n{text}",
            "stream": False
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
'''
