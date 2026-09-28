from pathlib import Path
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.database import get_db
from app.db.models import Document
from app.dependencies import get_chunker, get_embedding_service, get_vector_store
from app.schemas import DocumentIngestionResponse
from app.services.chunking import Chunker, resolve_chunk_params
from app.services.document_parser import extract_text, validate_file_extension
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Document Ingestion"])

COPY_BUFFER_SIZE = 1024 * 1024


def save_upload(file: UploadFile, destination: Path, max_bytes: int) -> None:
    """Stream the upload to disk, aborting with 413 once it exceeds max_bytes."""
    written = 0
    try:
        with destination.open("wb") as buffer:
            while chunk := file.file.read(COPY_BUFFER_SIZE):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is too large. Maximum size is {max_bytes // (1024 * 1024)} MB.",
                    )
                buffer.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


@router.post("/ingest", response_model=DocumentIngestionResponse)
def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(..., description="'fixed' or 'paragraph'"),
    chunk_size: Optional[int] = Form(
        None, description="Max tokens per chunk (defaults to CHUNK_SIZE; capped at the model limit)."
    ),
    overlap: Optional[int] = Form(
        None, description="Token overlap between chunks (fixed strategy only)."
    ),
    db: Session = Depends(get_db),
    chunker: Chunker = Depends(get_chunker),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
) -> DocumentIngestionResponse:
    # Unexpected errors are logged and turned into a generic 500 by the error
    # middleware; an unreachable Qdrant becomes a 503.
    strategy = chunking_strategy.lower()

    try:
        file_extension = validate_file_extension(file.filename)
        chunk_size, overlap = resolve_chunk_params(
            strategy,
            chunk_size,
            overlap,
            default_chunk_size=min(settings.chunk_size, chunker.max_tokens),
            default_overlap=settings.chunk_overlap,
        )
        chunker.validate(strategy, chunk_size, overlap)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    stored_filename = f"{uuid.uuid4()}{file_extension}"
    saved_file_path = Path(settings.upload_dir) / stored_filename
    save_upload(file, saved_file_path, settings.max_upload_size_mb * 1024 * 1024)

    document = None
    try:
        try:
            extracted_text = extract_text(str(saved_file_path), file_extension)
            chunks = chunker.chunk(extracted_text, strategy, chunk_size, overlap)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks were created from the document.")

        document = Document(
            filename=file.filename,
            stored_filename=stored_filename,
            file_type=file_extension,
            chunking_strategy=strategy,
            total_chunks=len(chunks),
            raw_text=extracted_text,
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        embeddings = embedding_service.embed_texts(chunks)

        vector_store.upsert_chunks(
            document_id=document.id,
            filename=document.filename,
            strategy=document.chunking_strategy,
            chunks=chunks,
            embeddings=embeddings,
        )
    except Exception:
        # Don't leave an uploaded file or a document row without vectors behind.
        saved_file_path.unlink(missing_ok=True)
        if document is not None and document.id is not None:
            logger.warning("Indexing failed for document %s; removing its database row", document.id)
            db.delete(document)
            db.commit()
        raise

    logger.info(
        "Ingested document %s (%s) into %d chunks using '%s' strategy (size=%d, overlap=%d)",
        document.id,
        document.filename,
        len(chunks),
        strategy,
        chunk_size,
        overlap,
    )

    return DocumentIngestionResponse(
        document_id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        chunking_strategy=document.chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        total_chunks=document.total_chunks,
        message="Document ingested successfully."
    )
