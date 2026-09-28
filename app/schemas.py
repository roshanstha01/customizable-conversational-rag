from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings


class DocumentIngestionResponse(BaseModel):
    document_id: int
    filename: str
    file_type: str
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    total_chunks: int
    message: str


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_type: str
    chunking_strategy: str
    total_chunks: int
    uploaded_at: Optional[datetime] = None


class DocumentDeleteResponse(BaseModel):
    document_id: int
    message: str


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=get_settings().max_top_k,
        description="Number of chunks to retrieve (defaults to the TOP_K setting).",
    )
    document_ids: Optional[List[int]] = Field(
        default=None, description="Only search chunks from these documents."
    )
    chunking_strategy: Optional[Literal["fixed", "paragraph"]] = Field(
        default=None, description="Only search chunks created with this strategy."
    )


class Source(BaseModel):
    document_id: int
    filename: str
    chunk_index: int
    score: float
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: List[Source] = []
