# app/memory/backends.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import redis
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.memory.schema import ShortTermMemoryEntry, LongTermMemoryEntry
from app.config.settings import settings
from redis import asyncio as aioredis
import asyncio
from qdrant_client import AsyncQdrantClient


# ======================
# SHORT-TERM MEMORY BACKENDS
# ======================
class STMBackend(ABC):
    @abstractmethod
    def set(self, session_id: str, key: str, entry: ShortTermMemoryEntry):
        pass

    @abstractmethod
    def get(self, session_id: str, key: str) -> Optional[ShortTermMemoryEntry]:
        pass

    @abstractmethod
    def get_all(self, session_id: str) -> Dict[str, ShortTermMemoryEntry]:
        pass

    @abstractmethod
    def clear(self, session_id: str):
        pass

    @abstractmethod
    def cleanup_expired(self):
        pass


class InMemorySTMBackend(STMBackend):
    def __init__(self, ttl_minutes: int = 30):
        self._store: Dict[str, Dict[str, ShortTermMemoryEntry]] = {}
        self.ttl = timedelta(minutes=ttl_minutes)

    def set(self, session_id: str, key: str, entry: ShortTermMemoryEntry):
        self._store.setdefault(session_id, {})[key] = entry

    def get(self, session_id: str, key: str) -> Optional[ShortTermMemoryEntry]:
        session_data = self._store.get(session_id, {})
        entry = session_data.get(key)
        if entry and not self._is_expired(entry.timestamp):
            return entry
        return None

    def get_all(self, session_id: str) -> Dict[str, ShortTermMemoryEntry]:
        session_data = self._store.get(session_id, {})
        return {
            k: v for k, v in session_data.items() if not self._is_expired(v.timestamp)
        }

    def clear(self, session_id: str):
        self._store.pop(session_id, None)

    def cleanup_expired(self):
        expired_keys = []
        for session_id, session_data in self._store.items():
            expired_session_keys = [
                key
                for key, entry in session_data.items()
                if self._is_expired(entry.timestamp)
            ]
            for key in expired_session_keys:
                del session_data[key]
            if not session_data:
                expired_keys.append(session_id)

        for key in expired_keys:
            del self._store[key]

    def _is_expired(self, timestamp: datetime) -> bool:
        return datetime.now(timestamp.tzinfo) - timestamp > self.ttl


class RedisSTMBackend(STMBackend):
    def __init__(
        self, redis_url: str = "redis://localhost:6379", ttl_minutes: int = 30
    ):
        self.redis_client = redis.from_url(redis_url)
        self.ttl_seconds = ttl_minutes * 60

    def _key(self, session_id: str, key: str = None) -> str:
        return f"stm:{session_id}:{key}" if key else f"stm:{session_id}:*"

    def set(self, session_id: str, key: str, entry: ShortTermMemoryEntry):
        redis_key = self._key(session_id, key)
        data = entry.model_dump_json()
        self.redis_client.setex(redis_key, self.ttl_seconds, data)

    def get(self, session_id: str, key: str) -> Optional[ShortTermMemoryEntry]:
        redis_key = self._key(session_id, key)
        data = self.redis_client.get(redis_key)
        return ShortTermMemoryEntry.model_validate_json(data) if data else None

    def get_all(self, session_id: str) -> Dict[str, ShortTermMemoryEntry]:
        keys = self.redis_client.keys(self._key(session_id))
        result = {}
        for redis_key in keys:
            data = self.redis_client.get(redis_key)
            if data:
                entry = ShortTermMemoryEntry.model_validate_json(data)
                key = redis_key.decode().split(":")[-1]
                result[key] = entry
        return result

    def clear(self, session_id: str):
        keys = self.redis_client.keys(self._key(session_id))
        if keys:
            self.redis_client.delete(*keys)

    def cleanup_expired(self):
        pass  # Redis TTL handles this automatically


# ---------------- Async version of the Redisstmbackend
class AsyncRedisSTMBackend(STMBackend):
    def __init__(
        self, redis_url: str = "redis://localhost:6379", ttl_minutes: int = 30
    ):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_minutes * 60
        self.redis_client = None  # Will be set in async init

    async def init(self):
        self.redis_client = await aioredis.from_url(self.redis_url)

    async def set(self, session_id: str, key: str, entry: ShortTermMemoryEntry):
        redis_key = f"stm:{session_id}:{key}"
        await self.redis_client.setex(
            redis_key, self.ttl_seconds, entry.model_dump_json()
        )

    async def get(self, session_id: str, key: str) -> Optional[ShortTermMemoryEntry]:
        redis_key = f"stm:{session_id}:{key}"
        data = await self.redis_client.get(redis_key)
        return ShortTermMemoryEntry.model_validate_json(data) if data else None

    async def get_all(self, session_id: str) -> Dict[str, ShortTermMemoryEntry]:
        pattern = f"stm:{session_id}:*"
        keys = await self.redis_client.keys(pattern)
        result = {}
        for redis_key in keys:
            data = await self.redis_client.get(redis_key)
            if data:
                entry = ShortTermMemoryEntry.model_validate_json(data)
                key = redis_key.decode().split(":")[-1]
                result[key] = entry
        return result

    async def clear(self, session_id: str):
        keys = await self.redis_client.keys(f"stm:{session_id}:*")
        if keys:
            await self.redis_client.delete(*keys)

    async def cleanup_expired(self):
        pass  # Redis handles TTL automatically


# ======================
# LONG-TERM MEMORY BACKENDS
# ======================
class LTMBackend(ABC):
    @abstractmethod
    def add_entry(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        pass

    @abstractmethod
    def search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.3,
    ) -> List[LongTermMemoryEntry]:
        pass


class QdrantLTMBackend(LTMBackend):
    def __init__(
        self,
        host: str = settings.LTM_QDRANT_HOST,
        port: int = settings.LTM_QDRANT_PORT,
        collection_name: str = settings.LTM_COLLECTION_NAME,
        embedding_model: str = settings.LTM_EMBEDDING_MODEL,
        vector_size: int = settings.LTM_VECTOR_SIZE,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.embedding_model_name = embedding_model
        self.model = self._load_embedding_model()
        self._ensure_collection(vector_size)

    def _load_embedding_model(self):
        """
        Loads the embedding model from settings.
        Later, we can expand this to support OpenAI, Cohere, etc.
        """
        try:
            return SentenceTransformer(self.embedding_model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{self.embedding_model_name}': {e}"
            )

    def _ensure_collection(self, vector_size: int):
        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add_entry(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        embedding = self.model.encode(text).tolist()
        entry_id = str(uuid4())
        metadata = metadata or {}
        if conversation_id:
            metadata["conversation_id"] = conversation_id
        point = PointStruct(
            id=entry_id,
            vector=embedding,
            payload={"user_id": user_id, "text": text, "metadata": metadata},
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])
        return entry_id

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.3,
    ) -> List[LongTermMemoryEntry]:
        query_vector = self.model.encode(query_text).tolist()
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                field_key = key if key == "user_id" else f"metadata.{key}"
                conditions.append(
                    FieldCondition(key=field_key, match=MatchValue(value=value))
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=min_score,
        )
        return [
            LongTermMemoryEntry(
                id=str(res.id),
                user_id=res.payload["user_id"],
                text=res.payload["text"],
                metadata=res.payload.get("metadata", {}),
                embedding=res.vector,
            )
            for res in results
        ]

    def export_all(self) -> List[LongTermMemoryEntry]:
        """Fetches all entries from Qdrant."""
        results = self.client.scroll(
            collection_name=self.collection_name, with_payload=True, with_vectors=True
        )[0]
        return [
            LongTermMemoryEntry(
                id=str(res.id),
                user_id=res.payload["user_id"],
                text=res.payload["text"],
                metadata=res.payload.get("metadata", {}),
                embedding=res.vector,
            )
            for res in results
        ]

    def delete_entries(self, ids: List[str]):
        self.client.delete(
            collection_name=self.collection_name, points_selector={"points": ids}
        )


# ---------------- Async version of the qdrant backend


class AsyncQdrantLTMBackend(LTMBackend):
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        embedding_model: str,
        vector_size: int,
    ):
        self.collection_name = collection_name
        self.client = AsyncQdrantClient(host=host, port=port)
        self.embedding_model_name = embedding_model
        self.model = SentenceTransformer(self.embedding_model_name)
        asyncio.create_task(self._ensure_collection(vector_size))

    async def _ensure_collection(self, vector_size: int):
        collections = await self.client.get_collections()
        if self.collection_name not in [c.name for c in collections.collections]:
            await self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    async def add_entry(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
    ) -> Optional[str]:
        embedding = self.model.encode(text).tolist()
        entry_id = str(uuid4())
        metadata = metadata or {}
        if conversation_id:
            metadata["conversation_id"] = conversation_id
        point = PointStruct(
            id=entry_id,
            vector=embedding,
            payload={"user_id": user_id, "text": text, "metadata": metadata},
        )
        await self.client.upsert(collection_name=self.collection_name, points=[point])
        return entry_id

    async def search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.3,
    ) -> List[LongTermMemoryEntry]:
        query_vector = self.model.encode(query_text).tolist()
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                field_key = key if key == "user_id" else f"metadata.{key}"
                conditions.append(
                    FieldCondition(key=field_key, match=MatchValue(value=value))
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=min_score,
        )
        return [
            LongTermMemoryEntry(
                id=str(res.id),
                user_id=res.payload["user_id"],
                text=res.payload["text"],
                metadata=res.payload.get("metadata", {}),
                embedding=res.vector,
            )
            for res in results
        ]
