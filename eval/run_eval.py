"""Retrieval evaluation: compare chunking strategies on a fixed question set.

For each chunking configuration, the sample documents in eval/docs are parsed,
chunked, embedded and indexed with the app's own components, then every
question in eval/questions.jsonl is embedded and searched. A result counts as a
hit when a retrieved chunk from the expected document contains the question's
`expected` phrase (case- and whitespace-insensitive).

Metrics per configuration: hit rate@1/3/5, MRR@k and query latency
(embedding + vector search). Results are written as a markdown table.

    python -m eval.run_eval                           # in-memory Qdrant
    python -m eval.run_eval --qdrant-host localhost   # a running Qdrant server
"""

import argparse
import json
import logging
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from qdrant_client import QdrantClient

from app.config import get_settings
from app.services.chunking import Chunker, normalize_text
from app.services.document_parser import extract_text, validate_file_extension
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import QdrantVectorStore

EVAL_DIR = Path(__file__).parent
DOCS_DIR = EVAL_DIR / "docs"
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
RESULTS_FILE = EVAL_DIR / "results.md"


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str
    chunk_size: int
    overlap: int = 0

    @property
    def name(self) -> str:
        if self.strategy == "fixed":
            return f"fixed-{self.chunk_size}-o{self.overlap}"
        return f"paragraph-{self.chunk_size}"


DEFAULT_CONFIGS = [
    ChunkConfig("fixed", 64, 12),
    ChunkConfig("fixed", 128, 25),
    ChunkConfig("fixed", 254, 50),
    ChunkConfig("paragraph", 128),
    ChunkConfig("paragraph", 254),
]


@dataclass
class ConfigResult:
    config: ChunkConfig
    total_chunks: int
    avg_chunk_tokens: float
    index_seconds: float
    ranks: Dict[str, Optional[int]] = field(default_factory=dict)  # question id -> rank of first hit
    latencies_ms: List[float] = field(default_factory=list)

    def hit_rate(self, k: int) -> float:
        return sum(1 for r in self.ranks.values() if r is not None and r <= k) / len(self.ranks)

    def mrr(self) -> float:
        return sum(1 / r for r in self.ranks.values() if r is not None) / len(self.ranks)

    def latency_percentile(self, pct: float) -> float:
        values = sorted(self.latencies_ms)
        return values[min(len(values) - 1, round(pct / 100 * (len(values) - 1)))]


def normalize(text: str) -> str:
    return " ".join(normalize_text(text).lower().split())


def load_documents() -> Dict[str, str]:
    documents = {}
    for path in sorted(DOCS_DIR.iterdir()):
        try:
            extension = validate_file_extension(path.name)
        except ValueError:
            continue
        documents[path.name] = extract_text(str(path), extension)
    if not documents:
        sys.exit(f"No .txt or .pdf documents found in {DOCS_DIR}")
    return documents


def load_questions(documents: Dict[str, str]) -> List[dict]:
    lines = QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]

    # Catch typos early: each expected phrase must occur exactly once in its document.
    problems = []
    for q in questions:
        if q["doc"] not in documents:
            problems.append(f"{q['id']}: unknown document {q['doc']!r}")
            continue
        count = normalize(documents[q["doc"]]).count(normalize(q["expected"]))
        if count != 1:
            problems.append(f"{q['id']}: expected phrase found {count} times in {q['doc']}")
    if problems:
        sys.exit("Invalid question set:\n  " + "\n  ".join(problems))
    return questions


def evaluate_config(
    config: ChunkConfig,
    documents: Dict[str, str],
    questions: List[dict],
    embedding_service: EmbeddingService,
    chunker: Chunker,
    client: QdrantClient,
    top_k: int,
) -> ConfigResult:
    store = QdrantVectorStore(
        host="",
        port=0,
        collection_name=f"eval_{config.name.replace('-', '_')}",
        vector_size=embedding_service.dimension,
        client=client,
    )
    if client.collection_exists(store.collection_name):
        client.delete_collection(store.collection_name)
    store.ensure_collection()

    token_counts = []
    started = time.perf_counter()
    for doc_id, (name, text) in enumerate(documents.items(), start=1):
        chunks = chunker.chunk(text, config.strategy, config.chunk_size, config.overlap)
        token_counts.extend(chunker.count_tokens(chunk) for chunk in chunks)
        store.upsert_chunks(
            document_id=doc_id,
            filename=name,
            strategy=config.strategy,
            chunks=chunks,
            embeddings=embedding_service.embed_texts(chunks),
        )

    result = ConfigResult(
        config=config,
        total_chunks=len(token_counts),
        avg_chunk_tokens=statistics.mean(token_counts),
        index_seconds=time.perf_counter() - started,
    )

    for q in questions:
        started = time.perf_counter()
        query_embedding = embedding_service.embed_query(q["question"])
        points = store.search(query_embedding=query_embedding, limit=top_k)
        result.latencies_ms.append((time.perf_counter() - started) * 1000)

        expected = normalize(q["expected"])
        result.ranks[q["id"]] = next(
            (
                rank
                for rank, point in enumerate(points, start=1)
                if point.payload["filename"] == q["doc"] and expected in normalize(point.payload["chunk_text"])
            ),
            None,
        )

    client.delete_collection(store.collection_name)
    return result


def render_markdown(
    results: List[ConfigResult],
    questions: List[dict],
    documents: Dict[str, str],
    model: str,
    top_k: int,
    backend: str,
) -> str:
    direct = sum(q["type"] == "direct" for q in questions)
    lines = [
        "# Retrieval Evaluation",
        "",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Embedding model: `{model}`",
        f"- Documents: {', '.join(f'`{name}`' for name in documents)}",
        f"- Questions: {len(questions)} ({direct} direct, {len(questions) - direct} paraphrased)",
        f"- Vector store: {backend}; no similarity threshold",
        "- Latency: query embedding + vector search, per question",
        "",
        f"| Configuration | Chunks | Avg tokens | Hit@1 | Hit@3 | Hit@5 | MRR@{top_k} | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r.config.name}` | {r.total_chunks} | {r.avg_chunk_tokens:.0f} "
            f"| {r.hit_rate(1):.2f} | {r.hit_rate(3):.2f} | {r.hit_rate(5):.2f} "
            f"| {r.mrr():.3f} | {r.latency_percentile(50):.1f} | {r.latency_percentile(95):.1f} |"
        )

    lines += [
        "",
        "Configuration names: `fixed-<chunk tokens>-o<overlap tokens>` and `paragraph-<max tokens>`.",
        "",
        f"## Questions missed in the top {top_k}",
        "",
    ]
    by_id = {q["id"]: q for q in questions}
    misses_found = False
    for r in results:
        misses = [qid for qid, rank in r.ranks.items() if rank is None]
        if misses:
            misses_found = True
            details = "; ".join(f"{qid} \"{by_id[qid]['question']}\"" for qid in misses)
            lines.append(f"- `{r.config.name}`: {details}")
    if not misses_found:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qdrant-host", help="Use a Qdrant server instead of in-memory Qdrant")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=RESULTS_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")  # local-mode Qdrant warns that payload indexes are no-ops

    settings = get_settings()
    embedding_service = EmbeddingService(settings.embedding_model)
    chunker = Chunker(embedding_service.tokenizer, max_tokens=embedding_service.max_tokens)

    if args.qdrant_host:
        client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
        backend = f"Qdrant server at {args.qdrant_host}:{args.qdrant_port}"
    else:
        client = QdrantClient(":memory:")
        backend = "in-memory Qdrant (exact search)"

    documents = load_documents()
    questions = load_questions(documents)
    for q in questions[:5]:  # keep model start-up out of the latency numbers
        embedding_service.embed_query(q["question"])

    results = []
    for config in DEFAULT_CONFIGS:
        result = evaluate_config(config, documents, questions, embedding_service, chunker, client, args.top_k)
        results.append(result)
        print(
            f"{config.name:<18} chunks={result.total_chunks:<4} hit@1={result.hit_rate(1):.2f} "
            f"hit@3={result.hit_rate(3):.2f} hit@5={result.hit_rate(5):.2f} mrr={result.mrr():.3f} "
            f"p50={result.latency_percentile(50):.1f}ms"
        )

    args.output.write_text(
        render_markdown(results, questions, documents, settings.embedding_model, args.top_k, backend),
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
