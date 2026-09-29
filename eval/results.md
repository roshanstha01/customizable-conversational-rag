# Retrieval Evaluation

- Generated: 2026-09-29 12:23
- Embedding model: `all-MiniLM-L6-v2`
- Documents: `helio_h5_manual.pdf`, `northwind_handbook.txt`, `vector_search_primer.txt`
- Questions: 31 (14 direct, 17 paraphrased)
- Vector store: in-memory Qdrant (exact search); no similarity threshold
- Latency: query embedding + vector search, per question

| Configuration | Chunks | Avg tokens | Hit@1 | Hit@3 | Hit@5 | MRR@5 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fixed-64-o12` | 44 | 52 | 0.65 | 0.90 | 0.97 | 0.783 | 10.6 | 12.7 |
| `fixed-128-o25` | 23 | 108 | 0.68 | 1.00 | 1.00 | 0.833 | 10.9 | 12.6 |
| `fixed-254-o50` | 11 | 227 | 0.84 | 0.97 | 1.00 | 0.910 | 11.1 | 13.5 |
| `paragraph-128` | 25 | 88 | 0.94 | 1.00 | 1.00 | 0.968 | 10.8 | 11.4 |
| `paragraph-254` | 11 | 200 | 0.77 | 0.97 | 1.00 | 0.872 | 10.4 | 13.1 |

Configuration names: `fixed-<chunk tokens>-o<overlap tokens>` and `paragraph-<max tokens>`.

## Questions missed in the top 5

- `fixed-64-o12`: vs-03 "Which setting lets me get better recall from HNSW searches at the cost of speed?"
