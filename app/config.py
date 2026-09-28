from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, overridable via environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI/ML Intern Task API"
    log_level: str = "INFO"

    # Storage
    database_url: str = "sqlite:///./app.db"
    upload_dir: str = "uploads"

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

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    paragraph_max_length: int = 500

    # Retrieval
    top_k: int = 5
    max_history_messages: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
