"""Shared fakes and fixtures. Nothing here talks to Qdrant, Redis, Ollama or
downloads a model, so the suite runs offline."""

import hashlib
import json
import math
import re
from typing import Callable, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings, get_settings
from app.db.database import Base, get_db
from app.errors import ServiceUnavailableError
from app.services.chunking import Chunker

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


class FakeTokenizer:
    """Word/punctuation tokenizer with the same call interface as a HF fast tokenizer."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(TOKEN_PATTERN.finditer(text))
        result = {"input_ids": [hash(m.group()) % 30000 for m in matches]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(m.start(), m.end()) for m in matches]
        return result

    def num_special_tokens_to_add(self) -> int:
        return 2


STOPWORDS = {
    "a", "an", "and", "are", "for", "in", "is", "it", "of", "on", "s", "the", "to", "what", "which", "with",
}


class FakeEmbeddingService:
    """Bag-of-words hashing embeddings: texts sharing content words get similar
    vectors, texts sharing none score 0."""

    model_name = "fake-embedding"
    dimension = 4096
    max_tokens = 254

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.queries: List[str] = []

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for word in re.findall(r"\w+", text.lower()):
            if word in STOPWORDS:
                continue
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> List[float]:
        self.queries.append(query)
        return self._embed(query)


class FakePoint:
    def __init__(self, score: float, payload: Dict) -> None:
        self.score = score
        self.payload = payload


class FakeVectorStore:
    def __init__(self) -> None:
        self.points: Dict[int, Dict] = {}
        self.available = True

    def _check(self) -> None:
        if not self.available:
            raise ServiceUnavailableError("Qdrant")

    def ensure_collection(self) -> None:
        self._check()

    def ping(self) -> None:
        self._check()

    def upsert_chunks(self, document_id, filename, strategy, chunks, embeddings) -> None:
        self._check()
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.points[document_id * 100000 + index] = {
                "vector": embedding,
                "payload": {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "chunk_text": chunk,
                    "strategy": strategy,
                },
            }

    def search(self, query_embedding, limit=5, document_ids=None, strategy=None, score_threshold=None):
        self._check()
        results = []
        for point in self.points.values():
            payload = point["payload"]
            if document_ids and payload["document_id"] not in document_ids:
                continue
            if strategy and payload["strategy"] != strategy:
                continue
            score = sum(a * b for a, b in zip(query_embedding, point["vector"]))
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(FakePoint(score, payload))
        return sorted(results, key=lambda p: p.score, reverse=True)[:limit]

    def delete_document(self, document_id: int) -> None:
        self._check()
        self.points = {
            key: point for key, point in self.points.items()
            if point["payload"]["document_id"] != document_id
        }


class FakeMemoryService:
    def __init__(self) -> None:
        self.history: Dict[str, List[Dict[str, str]]] = {}
        self.state: Dict[str, Dict] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self.history.get(session_id, []))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.history.setdefault(session_id, []).append({"role": role, "content": content})

    def get_state(self, session_id: str, name: str) -> Optional[Dict]:
        value = self.state.get(f"{name}:{session_id}")
        return json.loads(value) if value is not None else None

    def set_state(self, session_id: str, name: str, value: Dict, ttl_seconds: int) -> None:
        self.state[f"{name}:{session_id}"] = json.dumps(value)

    def clear_state(self, session_id: str, name: str) -> None:
        self.state.pop(f"{name}:{session_id}", None)

    def ping(self) -> bool:
        return True


class FakeLLM:
    """Routes each call by its system prompt to a configurable handler."""

    def __init__(
        self,
        classify: Optional[Callable[[str, str], str]] = None,
        extract: Optional[Callable[[str], Dict]] = None,
        rewrite: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.classify = classify or (lambda message, system: "question")
        self.extract = extract or (lambda message: {})
        self.rewrite = rewrite
        self.calls: List[Dict] = []

    def generate_response(self, messages, temperature=None, json_format=False) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        self.calls.append({"system": system, "user": user, "messages": messages, "json": json_format})

        if "You classify" in system:
            return self.classify(user, system)
        if "Extract interview booking" in system:
            result = self.extract(user)
            return result if isinstance(result, str) else json.dumps(result)
        if "standalone search query" in system:
            latest = user.split("Latest message: ", 1)[1].split("\n", 1)[0]
            return self.rewrite(latest) if self.rewrite else latest
        context = next(m["content"] for m in messages if m["content"].startswith("Context:"))
        return f"ANSWER using {context[:60]!r}"

    def calls_of(self, marker: str) -> List[Dict]:
        return [call for call in self.calls if marker in call["system"]]

    def list_models(self) -> List[str]:
        return ["llama3:latest"]


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def chunker(fake_tokenizer) -> Chunker:
    return Chunker(fake_tokenizer, max_tokens=254)


@pytest.fixture
def db_session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    yield factory
    engine.dispose()


@pytest.fixture
def db(db_session_factory):
    session = db_session_factory()
    yield session
    session.close()


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return Settings(
        _env_file=None,
        upload_dir=str(upload_dir),
        min_similarity_score=0.15,
        chunk_size=50,
        chunk_overlap=10,
    )


@pytest.fixture
def app_fakes(db_session_factory, test_settings):
    """The FastAPI app wired to fakes. The lifespan is not run, so no model is
    loaded and no service is contacted."""
    from app.main import app

    fakes = {
        "embedding": FakeEmbeddingService(),
        "vector_store": FakeVectorStore(),
        "memory": FakeMemoryService(),
        "llm": FakeLLM(),
    }
    app.state.embedding_service = fakes["embedding"]
    app.state.chunker = Chunker(fakes["embedding"].tokenizer, max_tokens=254)
    app.state.vector_store = fakes["vector_store"]
    app.state.memory_service = fakes["memory"]
    app.state.llm_service = fakes["llm"]

    def override_db():
        session = db_session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_settings] = lambda: test_settings

    fakes["app"] = app
    fakes["client"] = TestClient(app)
    yield fakes

    app.dependency_overrides.clear()
