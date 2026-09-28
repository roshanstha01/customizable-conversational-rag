from pydantic import BaseModel


class DocumentIngestionResponse(BaseModel):
    document_id: int
    filename: str
    file_type: str
    chunking_strategy: str
    total_chunks: int
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str

