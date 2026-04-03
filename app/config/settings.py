# app/config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field

from typing import Literal


class Settings(BaseSettings):
    # Memory Backend
    STM_BACKEND: Literal["memory", "redis"] = "redis"
    REDIS_URL: str = "redis://localhost:6379"

    # Long-Term Memory
    LTM_COLLECTION_NAME: str = "memory_management_long_term_memory"
    LTM_VECTOR_SIZE: int = 384
    LTM_QDRANT_HOST: str = "localhost"
    LTM_QDRANT_PORT: int = 6333
    LTM_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # LTM_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    # LTM_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
    DEDUPLICATION_THRESHOLD: float = 0.92
    LTM_SUMMARIZATION_DAYS: int = 30  # Age threshold
    LTM_PRUNE_AFTER_SUMMARY: bool = True
    # Summarization & pruning
    LTM_SUMMARIZATION_DAYS: int = 30
    LTM_PRUNE_AFTER_SUMMARY: bool = True

    # AI model configs
    GEMINI_API_KEY: str = ""  # Add in .env
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OLLAMA_MODEL: str = "mistral"  # Local model name

    # Short-Term Memory
    STM_TTL_MINUTES: int = 30
    # STM Persistence Settings
    STM_PERSISTENCE_MODE: str = "aof"  # options: "aof", "rdb", "none"
    STM_RDB_SAVE_INTERVAL: int = 60  # seconds, only applies if "rdb"
    STM_RDB_SAVE_CHANGES: int = 1  # number of changes before saving

    # Cleanup
    ENABLE_CLEANUP: bool = True
    CLEANUP_INTERVAL_MINUTES: int = 60

    # Search
    MIN_SEARCH_SCORE: float = 0.3

    model_config = {"env_file": ".env"}


settings = Settings()
