import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.database import get_db
from app.db.models import Document
from app.dependencies import get_vector_store
from app.schemas import DocumentDeleteResponse, DocumentOut
from app.services.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Documents"])


@router.get("/documents", response_model=List[DocumentOut])
def get_documents(db: Session = Depends(get_db)):
    documents = db.query(Document).all()
    return documents


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
) -> DocumentDeleteResponse:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Vectors first: if Qdrant is down this raises a 503 and nothing is deleted.
    vector_store.delete_document(document_id)

    if document.stored_filename:
        # Only the basename is used, so a bad DB value can't point outside upload_dir.
        (Path(settings.upload_dir) / Path(document.stored_filename).name).unlink(missing_ok=True)
    else:
        logger.info("Document %s has no stored file (ingested before files were tracked)", document_id)

    db.delete(document)
    db.commit()

    logger.info("Deleted document %s (%s)", document_id, document.filename)
    return DocumentDeleteResponse(document_id=document_id, message="Document deleted successfully.")
