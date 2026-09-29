"""FastAPI dependencies that hand out the shared services created in the app lifespan."""

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.services.booking_service import BookingService
from app.services.chunking import Chunker
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import OllamaService
from app.services.memory_service import RedisMemoryService
from app.services.rag_service import RAGService
from app.services.vector_store import QdrantVectorStore


def get_embedding_service(request: Request) -> EmbeddingService:
    return request.app.state.embedding_service


def get_chunker(request: Request) -> Chunker:
    return request.app.state.chunker


def get_vector_store(request: Request) -> QdrantVectorStore:
    vector_store: QdrantVectorStore = request.app.state.vector_store
    # No-op once the collection exists; retries if Qdrant was down at startup.
    # Raises ServiceUnavailableError (-> 503) if Qdrant is still unreachable.
    vector_store.ensure_collection()
    return vector_store


def get_llm_service(request: Request) -> OllamaService:
    return request.app.state.llm_service


def get_memory_service(request: Request) -> RedisMemoryService:
    return request.app.state.memory_service


def get_booking_service(
    llm_service: OllamaService = Depends(get_llm_service),
    memory_service: RedisMemoryService = Depends(get_memory_service),
    settings: Settings = Depends(get_settings),
) -> BookingService:
    return BookingService(
        llm_service=llm_service,
        memory_service=memory_service,
        state_ttl_seconds=settings.booking_state_ttl_seconds,
    )


def get_rag_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    llm_service: OllamaService = Depends(get_llm_service),
    settings: Settings = Depends(get_settings),
) -> RAGService:
    return RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
        top_k=settings.top_k,
        max_history_messages=settings.max_history_messages,
        min_score=settings.min_similarity_score,
    )
