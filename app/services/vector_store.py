import logging
from typing import List

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.errors import unavailable_on

logger = logging.getLogger(__name__)

QDRANT_ERRORS = (ResponseHandlingException, httpx.TransportError, ConnectionError)


class QdrantVectorStore:
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        vector_size: int,
        timeout: int = 10,
    ) -> None:
        # Creating the client does not open a connection, so this is safe while Qdrant is down.
        self.client = QdrantClient(host=host, port=port, timeout=timeout)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._collection_ready = False

    def ensure_collection(self) -> None:
        if self._collection_ready:
            return

        with unavailable_on(QDRANT_ERRORS, "Qdrant"):
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                logger.info("Creating Qdrant collection '%s'", self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )

        self._collection_ready = True

    def ping(self) -> None:
        with unavailable_on(QDRANT_ERRORS, "Qdrant"):
            self.client.get_collections()

    def upsert_chunks(
        self,
        document_id: int,
        filename: str,
        strategy: str,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> None:
        points = []

        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=document_id * 100000 + index,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "chunk_text": chunk,
                    "strategy": strategy,
                },
            )
            points.append(point)

        with unavailable_on(QDRANT_ERRORS, "Qdrant"):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(self, query_embedding: List[float], limit: int = 5):
        with unavailable_on(QDRANT_ERRORS, "Qdrant"):
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
            )
        return results.points
