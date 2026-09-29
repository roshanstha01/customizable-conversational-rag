from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, overridable via environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Customizable Conversational RAG"
    log_level: str = "INFO"
    # IANA time zone used for "now"/"today" in bookings (dates and times are local to it).
    timezone: str = "Asia/Kathmandu"

    # Storage
    database_url: str = "sqlite:///./app.db"
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 10

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"
    qdrant_timeout: int = 10

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_timeout: int = 5

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3"

    # Chunking (sizes are in embedding-model tokens; capped at the model's limit)
    chunk_size: int = 200
    chunk_overlap: int = 40

    # Retrieval
    top_k: int = 5
    max_top_k: int = 20
    min_similarity_score: float = 0.25
    snippet_length: int = 200

    # Conversation
    max_history_messages: int = 6
    chat_history_ttl_seconds: int = 60 * 60 * 24
    chat_history_max_messages: int = 50
    # How long a half-finished booking is remembered between messages.
    booking_state_ttl_seconds: int = 60 * 30

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"Unknown time zone {value!r}; use an IANA name like 'Asia/Kathmandu'.") from error
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


def local_now(timezone: str | None = None) -> datetime:
    """Current wall-clock time in the configured time zone, as a naive datetime
    (booking dates/times are stored as naive local values)."""
    zone = ZoneInfo(timezone or get_settings().timezone)
    return datetime.now(zone).replace(tzinfo=None)
