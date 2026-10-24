from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SRC_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "multimodal-video-rag"
    APP_VERSION: str = "2.0.0"
    LOG_LEVEL: str = "INFO"

    # Empty disables auth; otherwise every /api/v1 call needs an `X-API-Key` header.
    API_KEY: str = ""

    # ========================= Storage =========================
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "multimodal_rag"
    ASSETS_DIR: str = str(SRC_DIR / "assets")
    KEEP_MEDIA_FILES: bool = False

    # ========================= Ingestion =========================
    MAX_VIDEO_DURATION_MINUTES: int = 60
    CHUNK_MAX_CHARS: int = 800
    CHUNK_OVERLAP_SEGMENTS: int = 1

    # ========================= Jobs =========================
    # "arq" runs jobs in a separate worker through Redis, "inline" runs them inside the API process.
    JOB_QUEUE_BACKEND: str = "inline"
    REDIS_URL: str = "redis://localhost:6379"

    # ========================= Transcription =========================
    TRANSCRIPTION_BACKEND: str = "GROQ"  # GROQ | LOCAL
    WHISPER_LOCAL_MODEL: str = "small"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    GROQ_WHISPER_MODEL: str = "whisper-large-v3-turbo"

    # ========================= Generation =========================
    GENERATION_BACKEND: str = "GROQ"  # GROQ | GEMINI | OLLAMA
    GENERATION_FALLBACK_BACKEND: str = "GEMINI"
    GENERATION_TEMPERATURE: float = 0.1
    GENERATION_MAX_TOKENS: int = 1024
    LLM_MAX_RETRIES: int = 3

    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:1b-instruct-q8_0"
    OLLAMA_VISION_MODEL: str = "llava"

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_VISION_MODEL: str = ""

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.6-flash"
    GEMINI_VISION_MODEL: str = "gemini-3.6-flash"

    # ========================= Vision =========================
    VISION_BACKEND: str = "GEMINI"  # GEMINI | OLLAMA | GROQ; empty disables frame analysis
    VISION_MAX_FRAMES: int = 24
    VISION_SCENE_THRESHOLD: float = 27.0
    VISION_MIN_SCENE_SECONDS: float = 4.0

    # ========================= Embeddings & retrieval =========================
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_QUERY_PREFIX: str = ""
    EMBEDDING_DOCUMENT_PREFIX: str = ""
    PRELOAD_MODELS: bool = False

    RETRIEVAL_MODE: str = "hybrid"  # dense | sparse | hybrid
    RETRIEVAL_CANDIDATES: int = 20
    RERANK_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # ========================= Vector DB =========================
    VECTOR_DB_BACKEND: str = "QDRANT"
    # Set QDRANT_URL to use a Qdrant server (required with the arq worker);
    # otherwise an embedded on-disk database is created under assets/database.
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    VECTOR_DB_PATH: str = "qdrant_db"
    VECTOR_DB_DISTANCE_METHOD: str = "cosine"

    # ========================= Templates =========================
    PRIMARY_LANG: str = "en"
    DEFAULT_LANG: str = "en"

    @property
    def assets_path(self) -> Path:
        return Path(self.ASSETS_DIR)


@lru_cache
def get_settings() -> Settings:
    return Settings()
