# Hybrid RAG: keyword + vector search, measured

Code for the video **Hybrid RAG explained: keyword + vector search, measured** on [Show Me the Query](https://www.youtube.com/@ShowMeTheQuery).

Keyword search and vector search fail on *different* questions. This project puts both into one SQLite file over the PostgreSQL 16 docs, merges them with Reciprocal Rank Fusion, and measures all three on 30 questions.

```
             ┌─ chunks_fts  (FTS5, BM25)  ──┐
question ────┤                              ├── RRF ── top 5 ── cited prompt ── any LLM
             └─ chunks_vec  (sqlite-vec)  ──┘
```

No GPU, no PyTorch, no server: the embedding model (all-MiniLM-L6-v2) runs in plain NumPy (`minilm.py`).

## Run it

```bash
pip install -r requirements.txt
python3 minilm.py                  # downloads the model once (~90 MB) and checks it against the reference output
python3 prepare_docs.py            # PostgreSQL 16 docs -> 615 chunks
python3 hybrid_rag.py ingest       # builds rag.db: chunks + FTS5 index + vector index
python3 hybrid_rag.py search "my table keeps growing even though I delete rows"
python3 hybrid_rag.py eval         # hit@5 and MRR for keyword vs vector vs hybrid
python3 hybrid_rag.py ask "my table keeps growing even though I delete rows"
```

`ask` prints a grounded prompt with numbered sources. To get an answer, point it at any OpenAI-compatible endpoint, for example a local model in Ollama:

```bash
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.2
python3 hybrid_rag.py ask "how do I stop two users from editing the same row at once"
```

## What we got

From [`data/eval-output.txt`](data/eval-output.txt), on the 30 questions in [`data/eval.jsonl`](data/eval.jsonl):

| method | hit@5 | MRR@5 | exact-term questions | natural questions |
|---|---|---|---|---|
| keyword (BM25) | 23/30 | 0.71 | 15/15 | 8/15 |
| vector (MiniLM) | 25/30 | 0.71 | 15/15 | 10/15 |
| **hybrid (RRF)** | **27/30** | **0.78** | 15/15 | 12/15 |

Latency per question on a 2-vCPU VM ([`data/latency.txt`](data/latency.txt)): keyword 0.6 ms, vector 33.2 ms, hybrid 35.0 ms. Almost all of that is embedding the question (32.5 ms); the vector lookup itself is 0.8 ms.

**Read these numbers carefully.** The questions were written and labelled before any retrieval ran, but 30 hand-written questions on one corpus is a smoke test, not a benchmark. Run `eval` on your own questions and documents.

## How the fusion works

```python
score[doc] = sum(1 / (60 + rank) for each retriever that returned doc)
```

Reciprocal Rank Fusion ignores raw scores (BM25 and cosine distance aren't comparable) and uses only ranks. `k = 60` comes from the original paper. A real example is in [`data/rrf-example.txt`](data/rrf-example.txt): *Index-Only Scans* is rank 1 for keyword and rank 9 for vector, so it beats *Searching a Table* (rank 12 and rank 1).

## Files

| File | What it does |
|---|---|
| `prepare_docs.py` | Downloads 8 chapters of the PostgreSQL 16 docs and splits them into ~1,200-character chunks with section URLs |
| `minilm.py` | all-MiniLM-L6-v2 in NumPy: BERT encoder, mean pooling, L2 normalisation |
| `hybrid_rag.py` | `ingest`, `search`, `ask`, `eval` |
| `data/eval.jsonl` | 30 questions with the section(s) that answer them |
| `data/*.txt` | The exact outputs shown in the video |

## Try this next

- Add a reranker over the fused top 20 and see if it recovers the three misses (e.g. "walk a tree of employees and managers" → *Recursive Queries*).
- Weight the retrievers: multiply the keyword term by 0.5 and see how hit@5 moves.
- Swap SQLite for Postgres: `pgvector` for vectors and `tsvector` for keywords. Note Postgres's built-in `ts_rank` is not BM25.

## Sources

- Cormack, Clarke & Büttcher (2009), [Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
- [SQLite FTS5](https://www.sqlite.org/fts5.html) (BM25 ranking, porter stemmer)
- [sqlite-vec](https://github.com/asg017/sqlite-vec)
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0)
- [PostgreSQL 16 documentation](https://www.postgresql.org/docs/16/) (PostgreSQL License, © PostgreSQL Global Development Group). The docs are downloaded at run time; only the excerpts in `data/ask-example.txt` are stored here.
