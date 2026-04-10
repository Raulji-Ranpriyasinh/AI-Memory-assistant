"""
Knowledge Directory — admin-managed healthcare content retrieval (Phase 8).
Uses a completely separate Pinecone namespace from user personal memories.
Namespace: 'knowledge_directory'. No user_id filter on searches.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.memory.pinecone_ltm import PineconeLTMManager
from app.observability.metrics import metrics

# Valid content categories
VALID_KD_CATEGORIES = {
    "program_content",
    "nutrition_guide",
    "health_info",
    "user_generated",
}


class KnowledgeDirectory:
    """
    Separate Pinecone namespace/index for admin-managed healthcare content.
    The chat node queries both personal LTM AND the knowledge directory,
    with personal memories weighted higher.
    """

    NAMESPACE = "knowledge_directory"

    def __init__(self, ltm_manager: Optional[PineconeLTMManager] = None):
        """
        Store PineconeLTMManager reference.
        If not provided, creates a new instance.
        """
        if ltm_manager is None:
            ltm_manager = PineconeLTMManager()
        self.ltm = ltm_manager

    # ── Search ─────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embed query. Pinecone query in knowledge_directory namespace
        (no user_id filter). Return [{text, category, source, relevance_score}].
        Called by chat_node for every conversation turn.
        """
        try:
            query_vec = self.ltm.embed(query)
            response = self.ltm._index.query(
                vector=query_vec,
                top_k=top_k,
                include_metadata=True,
                namespace=self.NAMESPACE,
            )

            formatted = []
            for match in response.get("matches", []):
                meta = match.get("metadata", {})
                formatted.append({
                    "text": meta.get("text", ""),
                    "category": meta.get("category", ""),
                    "source": meta.get("source", ""),
                    "relevance_score": match.get("score", 0.0),
                })

            metrics.log(
                "kd_search",
                query_len=len(query),
                results_count=len(formatted),
                top_k=top_k,
            )
            return formatted

        except Exception as exc:
            metrics.log("kd_search_failed", error=str(exc))
            return []

    # ── Single content ingestion ───────────────────────────────────────────

    def ingest_content(
        self,
        content: str,
        category: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Embed content text. Upsert into knowledge_directory namespace
        with metadata {category, source, ingested_at, ...metadata}.
        Return content_id (UUID).
        """
        if category not in VALID_KD_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of: {VALID_KD_CATEGORIES}"
            )

        content_id = str(uuid.uuid4())
        ingested_at = datetime.utcnow().isoformat() + "Z"

        vector_meta = {
            "text": content,
            "category": category,
            "source": source,
            "ingested_at": ingested_at,
            "content_id": content_id,
            **(metadata or {}),
        }

        # Embed and upsert via PineconeLTMManager
        embedding = self.ltm.embed(content)
        self.ltm._index.upsert(
            vectors=[(content_id, embedding, vector_meta)],
            namespace=self.NAMESPACE,
        )

        metrics.log("kd_ingest", category=category, content_len=len(content))
        return content_id

    # ── Batch ingestion ────────────────────────────────────────────────────

    def ingest_batch(
        self, items: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Each item = {content, category, source, metadata}.
        Embed all. Batch upsert. Return list of content_ids.
        """
        if not items:
            return []

        content_ids = [str(uuid.uuid4()) for _ in items]
        texts = [item["content"] for item in items]
        ingested_at = datetime.utcnow().isoformat() + "Z"

        # Batch embed
        embeddings = self.ltm.embed_batch(texts)

        vectors = []
        for i, item in enumerate(items):
            category = item.get("category", "health_info")
            if category not in VALID_KD_CATEGORIES:
                metrics.log("kd_ingest_invalid_category", category=category)
                continue

            vector_meta = {
                "text": item["content"],
                "category": category,
                "source": item.get("source", ""),
                "ingested_at": ingested_at,
                "content_id": content_ids[i],
                **(item.get("metadata") or {}),
            }
            vectors.append((content_ids[i], embeddings[i], vector_meta))

        if vectors:
            self.ltm._index.upsert(
                vectors=vectors,
                namespace=self.NAMESPACE,
            )

        metrics.log("kd_batch_ingest", count=len(vectors))
        return content_ids

    # ── Delete content ────────────────────────────────────────────────────

    def delete_content(self, content_id: str) -> bool:
        """Delete vector by ID from knowledge_directory namespace."""
        try:
            self.ltm._index.delete(
                ids=[content_id],
                namespace=self.NAMESPACE,
            )
            metrics.log("kd_delete", content_id=content_id)
            return True
        except Exception as exc:
            metrics.log("kd_delete_failed", content_id=content_id, error=str(exc))
            return False

    # ── List content ──────────────────────────────────────────────────────

    def list_content(
        self,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Pinecone list or query with optional filter by category.
        Return metadata for up to `limit` items.
        """
        results = []

        try:
            # Use Pinecone list to iterate vectors in namespace
            # Since Pinecone doesn't support direct listing without a query,
            # we do a broad-match query with a dummy embedding
            embedding = self.ltm.embed("health")  # broad seed query
            query_resp = self.ltm._index.query(
                vector=embedding,
                top_k=limit,
                include_metadata=True,
                namespace=self.NAMESPACE,
            )

            for match in query_resp.get("matches", []):
                meta = match.get("metadata", {})
                if category and meta.get("category") != category:
                    continue
                results.append({
                    "content_id": meta.get("content_id", match.get("id", "")),
                    "text": meta.get("text", "")[:200] + "...",
                    "category": meta.get("category", ""),
                    "source": meta.get("source", ""),
                    "ingested_at": meta.get("ingested_at", ""),
                    "score": match.get("score", 0.0),
                })

        except Exception as exc:
            metrics.log("kd_list_failed", error=str(exc))

        metrics.log("kd_list", count=len(results), category=category)
        return results[:limit]
