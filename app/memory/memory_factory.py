# app/memory/memory_factory.py
from app.memory.memory_manager import MemoryManager
from app.config.settings import settings


def get_memory_manager() -> MemoryManager:
    return MemoryManager(
        stm_backend=settings.STM_BACKEND,
        stm_ttl_minutes=settings.STM_TTL_MINUTES,
        redis_url=settings.REDIS_URL,
        collection_name=settings.LTM_COLLECTION_NAME,
        embedding_model=settings.LTM_EMBEDDING_MODEL,
        enable_cleanup=settings.ENABLE_CLEANUP,
        cleanup_interval_minutes=settings.CLEANUP_INTERVAL_MINUTES,
    )
