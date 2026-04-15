"""
Central configuration — all constants and environment loading.
Import this module everywhere instead of repeating os.getenv() calls.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ───────────────────────────────────────────────────────────────────
DB_URI: str = os.getenv(
    "DB_URI", "postgresql://postgres:postgres@localhost:5442/postgres"
)

# ── Pinecone ───────────────────────────────────────────────────────────────────
PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX: str          = os.getenv("PINECONE_INDEX", "ltm-memories")
PINECONE_CLOUD: str          = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str         = os.getenv("PINECONE_REGION", "us-east-1")

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int        = 384

# ── STM / Memory layer ────────────────────────────────────────────────────────
STM_A_WINDOW_SIZE: int        = 10
SUMMARY_LAYER_THRESHOLDS: list[int] = [20, 50, 100]

# ── STM-C extraction ──────────────────────────────────────────────────────────
STM_C_EXTRACTION_INTERVAL: int = 3   # Only extract every N turns

# ── LTM gating ────────────────────────────────────────────────────────────────
LTM_SALIENCE_THRESHOLD: float  = 0.7
LTM_TOP_K: int                 = 20
COSINE_DEDUP_THRESHOLD: float  = 0.90

# ── Retrieval token budget ─────────────────────────────────────────────────────
# Approximate chars-per-token ratio used for budget estimation
CHARS_PER_TOKEN: float   = 4.0
LTM_TOKEN_BUDGET: int    = 2_000   # Max tokens the LTM block may consume in prompt

# ── Memory decay / pruning ────────────────────────────────────────────────────
DECAY_HALF_LIFE_DAYS: float = 60.0
PRUNE_THRESHOLD: float      = 0.15
ACCESS_BOOST: float         = 0.05

# ── Retrieval score weights ────────────────────────────────────────────────────
RELEVANCE_WEIGHT: float = 0.60
SALIENCE_WEIGHT: float  = 0.30
RECENCY_WEIGHT: float   = 0.10

# ── LLM models ────────────────────────────────────────────────────────────────
CHAT_MODEL: str   = "gemini-2.5-flash"
MEMORY_MODEL: str = "gemini-2.5-flash"   # Cheaper for high-volume ops
CHAT_TEMPERATURE: float   = 0.7
MEMORY_TEMPERATURE: float = 0.0

# ── Security ───────────────────────────────────────────────────────────────────
# Hard limit: no single fetch may return more than this many memories
MAX_MEMORIES_PER_USER: int = 10_000

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
API_PORT: int = int(os.getenv('API_PORT', '8000'))
CORS_ORIGINS: list[str] = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
JWT_SECRET: str = os.getenv('JWT_SECRET', 'change-me-in-production')
JWT_ALGORITHM: str = 'HS256'
RATE_LIMIT_PER_MINUTE: int = 60

# ── Health categories ──────────────────────────────────────────────────────────
VALID_CATEGORIES: list[str] = [
    'identity', 'health_condition', 'medication', 'dietary',
    'cgm_pattern', 'mood_pattern', 'activity', 'program', 'preferences'
]

# ── Feature flags ──────────────────────────────────────────────────────────────
ENABLE_CGM: bool = os.getenv('ENABLE_CGM', 'true').lower() == 'true'
ENABLE_NUDGES: bool = os.getenv('ENABLE_NUDGES', 'true').lower() == 'true'
ENABLE_FOOD_RECOGNITION: bool = os.getenv('ENABLE_FOOD_RECOGNITION', 'true').lower() == 'true'
ENABLE_VOICE: bool = os.getenv('ENABLE_VOICE', 'false').lower() == 'true'

# ── CGM thresholds ─────────────────────────────────────────────────────────────
CGM_NORMAL_LOW: float = 70.0
CGM_NORMAL_HIGH: float = 140.0
CGM_SPIKE_THRESHOLD: float = 180.0
CGM_HYPO_THRESHOLD: float = 70.0

# ── Nudge settings ─────────────────────────────────────────────────────────────
NUDGE_CHECK_INTERVAL_MINUTES: int = 30
MEAL_REMINDER_HOURS: int = 5

# ── External API keys ──────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY', '')
FIREBASE_CREDENTIALS_PATH: str = os.getenv('FIREBASE_CREDENTIALS_PATH', '')
NUTRITIONIX_APP_ID: str = os.getenv('NUTRITIONIX_APP_ID', '')
NUTRITIONIX_API_KEY: str = os.getenv('NUTRITIONIX_API_KEY', '')
GOOGLE_SPEECH_API_KEY: str = os.getenv('GOOGLE_SPEECH_API_KEY', '')
HIPAA_ENCRYPTION_KEY: str = os.getenv('HIPAA_ENCRYPTION_KEY', '')
