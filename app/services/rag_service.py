import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import OllamaService
from app.services.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

REWRITE_PROMPT = (
    "You rewrite a user's latest chat message into a standalone search query for a "
    "document search engine.\n"
    "Rules:\n"
    "- Do NOT answer the message. Only rewrite it.\n"
    "- Replace pronouns and references (it, they, that, the second one) with what they "
    "refer to in the conversation.\n"
    "- Keep the user's key terms. If the message is already standalone, return it unchanged.\n"
    "- Output only the rewritten query on one line, with no explanation, quotes or prefix.\n"
    "\n"
    "Example:\n"
    "Conversation:\nUser: What is FastAPI?\nAssistant: FastAPI is a Python web framework.\n"
    "Latest message: Who created it?\n"
    "Standalone query: Who created FastAPI?"
)


@dataclass
class RetrievedChunk:
    document_id: int
    filename: str
    chunk_index: int
    score: float
    text: str


@dataclass
class RAGAnswer:
    answer: str
    search_query: str
    chunks: List[RetrievedChunk] = field(default_factory=list)


class RAGService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        llm_service: OllamaService,
        top_k: int = 5,
        max_history_messages: int = 6,
        min_score: Optional[float] = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.top_k = top_k
        self.max_history_messages = max_history_messages
        self.min_score = min_score

    def _recent_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        max_history = self.max_history_messages
        return history[-max_history:] if len(history) > max_history else history

    def rewrite_query(self, message: str, history: List[Dict[str, str]]) -> str:
        """Turn the latest message into a standalone query using recent history."""
        recent_history = self._recent_history(history)
        if not recent_history:
            return message

        transcript = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}" for turn in recent_history
        )
        messages = [
            {"role": "system", "content": REWRITE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation:\n{transcript}\n\nLatest message: {message}\n\n"
                    "Rewrite the latest message as a standalone search query. Do not answer it."
                ),
            },
        ]

        rewritten = self.llm_service.generate_response(messages, temperature=0)
        # Models sometimes add a label, quotes or extra lines; keep just the query.
        lines = [line.strip() for line in rewritten.strip().splitlines() if line.strip()]
        query = lines[0] if lines else ""
        for prefix in ("standalone query:", "query:"):
            if query.lower().startswith(prefix):
                query = query[len(prefix):].strip()
        query = query.strip("\"'` ")

        if not query:
            return message

        logger.info("Rewrote query %r -> %r", message, query)
        return query

    def retrieve(
        self,
        query: str,
        limit: int,
        document_ids: Optional[List[int]] = None,
        strategy: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        query_embedding = self.embedding_service.embed_query(query)
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit,
            document_ids=document_ids,
            strategy=strategy,
            score_threshold=self.min_score,
        )

        chunks = []
        for result in search_results:
            payload = result.payload
            if payload and "chunk_text" in payload:
                chunks.append(
                    RetrievedChunk(
                        document_id=payload.get("document_id"),
                        filename=payload.get("filename", ""),
                        chunk_index=payload.get("chunk_index", 0),
                        score=result.score,
                        text=payload["chunk_text"],
                    )
                )

        return chunks

    def build_messages(
        self,
        query: str,
        context_chunks: List[str],
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        context_text = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."

        system_prompt = (
            "You are a helpful AI assistant.\n"
            "Use ONLY the provided context to answer.\n"
            "If the answer is not in the context, say: 'I could not find this in the document.'\n"
            "Keep answers short and clear."
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": f"Context:\n{context_text}"
            },
        ]

        messages.extend(self._recent_history(history))
        messages.append({"role": "user", "content": query})

        return messages

    def answer_query(
        self,
        query: str,
        history: List[Dict[str, str]],
        top_k: Optional[int] = None,
        document_ids: Optional[List[int]] = None,
        strategy: Optional[str] = None,
    ) -> RAGAnswer:
        search_query = self.rewrite_query(query, history)
        chunks = self.retrieve(
            query=search_query,
            limit=top_k or self.top_k,
            document_ids=document_ids,
            strategy=strategy,
        )
        messages = self.build_messages(
            query=query,
            context_chunks=[chunk.text for chunk in chunks],
            history=history,
        )
        answer = self.llm_service.generate_response(messages)
        return RAGAnswer(answer=answer, search_query=search_query, chunks=chunks)
