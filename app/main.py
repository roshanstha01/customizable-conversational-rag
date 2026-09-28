import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.booking import router as booking_router
from app.api.chat import router as chat_router
from app.api.document import router as document_router
from app.api.health import router as health_router
from app.api.ingestion import router as ingestion_router
from app.config import get_settings
from app.db.database import Base, engine
from app.errors import ServiceUnavailableError, register_error_handlers
from app.logging_config import setup_logging
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import OllamaService
from app.services.memory_service import RedisMemoryService
from app.services.vector_store import QdrantVectorStore

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s", settings.app_name)

    Base.metadata.create_all(bind=engine)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    # Loaded once and shared by every request.
    embedding_service = EmbeddingService(settings.embedding_model)

    vector_store = QdrantVectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection,
        vector_size=embedding_service.dimension,
        timeout=settings.qdrant_timeout,
    )
    try:
        vector_store.ensure_collection()
    except ServiceUnavailableError:
        logger.warning(
            "Qdrant is not reachable at %s:%s; starting anyway. "
            "Ingestion and chat will return 503 until it is available.",
            settings.qdrant_host,
            settings.qdrant_port,
        )

    app.state.embedding_service = embedding_service
    app.state.vector_store = vector_store
    app.state.memory_service = RedisMemoryService(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        timeout=settings.redis_timeout,
    )
    app.state.llm_service = OllamaService(host=settings.ollama_host, model_name=settings.llm_model)

    logger.info("Startup complete")
    yield
    logger.info("Shutting down")


app = FastAPI(title=settings.app_name, lifespan=lifespan)

register_error_handlers(app)

app.include_router(health_router)
app.include_router(ingestion_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(document_router, prefix="/api")
app.include_router(booking_router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": "API is running"}
