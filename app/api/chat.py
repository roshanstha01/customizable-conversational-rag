import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.dependencies import get_booking_service, get_memory_service, get_rag_service
from app.schemas import ChatRequest, ChatResponse
from app.services.booking_service import BookingService
from app.services.memory_service import RedisMemoryService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Conversational RAG"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    memory_service: RedisMemoryService = Depends(get_memory_service),
    booking_service: BookingService = Depends(get_booking_service),
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    # Unexpected errors are logged and turned into a generic 500 by the error
    # middleware; unreachable Redis/Qdrant/Ollama become a 503.
    history = memory_service.get_history(request.session_id)

    if booking_service.is_booking_intent(request.message):
        logger.info("Booking intent detected for session %s", request.session_id)
        response_text = booking_service.save_booking(db=db, message=request.message)
    else:
        response_text = rag_service.answer_query(
            query=request.message,
            history=history,
        )

    memory_service.add_message(request.session_id, "user", request.message)
    memory_service.add_message(request.session_id, "assistant", response_text)

    return ChatResponse(
        session_id=request.session_id,
        response=response_text,
    )
