"""Hybrid RAG in one SQLite file: BM25 (FTS5) + vector search (sqlite-vec) fused with Reciprocal Rank Fusion.

    python3 prepare_docs.py                      # PostgreSQL 16 docs -> data/chunks.jsonl
    python3 hybrid_rag.py ingest                 # build rag.db (chunks, FTS5 index, vector index)
    python3 hybrid_rag.py search "why is my table bloated after deletes"
    python3 hybrid_rag.py ask "..."              # grounded prompt with citations (+ LLM answer if configured)
    python3 hybrid_rag.py eval                   # hit@5 and MRR for keyword vs vector vs hybrid

Embeddings: all-MiniLM-L6-v2 (384 dims), run in plain NumPy by minilm.py, so no PyTorch or GPU is needed.
"""
import json, os, re, sqlite3, sys, time
from pathlib import Path

import sqlite_vec

HERE = Path(__file__).parent
DB = HERE / "rag.db"
CHUNKS = HERE / "data" / "chunks.jsonl"
MODEL = "all-MiniLM-L6-v2"
K_RRF = 60          # standard RRF constant (Cormack, Clarke & Buettcher, SIGIR 2009)
CANDIDATES = 50     # how deep each retriever looks before fusion

_model = None
def embed(texts):
    global _model
    if _model is None:
        from minilm import MiniLM
        _model = MiniLM()
    return _model.encode(texts)


def connect():
    db = sqlite3.connect(DB)
    db.enable_load_extension(True); sqlite_vec.load(db); db.enable_load_extension(False)
    return db


# ---------------- ingest ----------------
def ingest():
    rows = [json.loads(l) for l in CHUNKS.open()]
    if DB.exists(): DB.unlink()
    db = connect()
    db.executescript("""
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, chapter TEXT, section TEXT, url TEXT, text TEXT);
        -- BM25 keyword index. 'porter' stems words, so 'vacuuming' matches 'vacuum'.
        CREATE VIRTUAL TABLE chunks_fts USING fts5(section, text, content='chunks', content_rowid='id', tokenize='porter');
        -- Vector index: one 384-dim embedding per chunk, cosine distance.
        CREATE VIRTUAL TABLE chunks_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[384] distance_metric=cosine);
    """)
    db.executemany("INSERT INTO chunks VALUES (:id, :chapter, :section, :url, :text)", rows)
    db.execute("INSERT INTO chunks_fts (rowid, section, text) SELECT id, section, text FROM chunks")
    t = time.perf_counter()
    vecs = embed([f"{r['section']}. {r['text']}" for r in rows])
    for r, v in zip(rows, vecs):
        db.execute("INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)", (r["id"], v.astype("float32").tobytes()))
    db.commit()
    print(f"ingested {len(rows)} chunks; embedded in {time.perf_counter() - t:.1f}s with {MODEL}")


# ---------------- retrievers ----------------
def fts_query(q):
    """Turn free text into an FTS5 OR-query of quoted words, so punctuation can't break the syntax."""
    words = re.findall(r"[A-Za-z0-9_]+", q.lower())
    return " OR ".join(f'"{w}"' for w in words) or '""'

def keyword(db, q, k=CANDIDATES):
    # bm25() is lower = better in SQLite, so ORDER BY ascending.
    return db.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                      (fts_query(q), k)).fetchall()

def vector(db, q, k=CANDIDATES):
    v = embed([q])[0].astype("float32").tobytes()
    return db.execute("SELECT id FROM chunks_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance", (v, k)).fetchall()

def hybrid(db, q, k=CANDIDATES):
    """Reciprocal Rank Fusion: score(d) = sum over retrievers of 1 / (K_RRF + rank_d). Ranks only, no raw scores."""
    kw, ve = [r[0] for r in keyword(db, q, k)], [r[0] for r in vector(db, q, k)]
    score = {}
    for ranked in (kw, ve):
        for rank, cid in enumerate(ranked, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (K_RRF + rank)
    return [(cid,) for cid in sorted(score, key=score.get, reverse=True)]

METHODS = {"keyword": keyword, "vector": vector, "hybrid": hybrid}

def top(db, method, q, n=5):
    ids = [r[0] for r in METHODS[method](db, q)][:n]
    rows = {r[0]: r for r in db.execute(f"SELECT id, section, url, text FROM chunks WHERE id IN ({','.join('?' * len(ids))})", ids)}
    return [rows[i] for i in ids]


# ---------------- commands ----------------
def search(q):
    db = connect()
    for m in METHODS:
        print(f"\n== {m} ==")
        for i, (cid, section, url, text) in enumerate(top(db, m, q), 1):
            print(f" {i}. [{cid}] {section}  ({url.split('/')[-1]})")


def ask(q, n=5):
    db = connect(); hits = top(db, "hybrid", q, n)
    context = "\n\n".join(f"[{i}] {section} ({url})\n{text}" for i, (_, section, url, text) in enumerate(hits, 1))
    prompt = ("Answer the question using only the sources below. Cite them like [1]. "
              "If the sources don't contain the answer, say so.\n\n"
              f"Sources:\n{context}\n\nQuestion: {q}\nAnswer:")
    base, model = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_MODEL")
    if not (base and model):
        print(prompt); print("\n(Set LLM_BASE_URL and LLM_MODEL for any OpenAI-compatible endpoint, e.g. Ollama, to get an answer.)")
        return
    import urllib.request
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'none')}"},
                                 data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode())
    ans = json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"]
    print(ans); print("\nSources:"); [print(f" [{i}] {s} {u}") for i, (_, s, u, _) in enumerate(hits, 1)]


def evaluate(path=HERE / "data" / "eval.jsonl", n=5):
    """Each line: {q, expect: [section titles that answer it]}. A hit = any expected section in the top n."""
    db = connect(); qs = [json.loads(l) for l in open(path)]
    print(f"{len(qs)} questions, hit@{n} and MRR@{n}\n")
    res = {}
    kinds = sorted({q.get("kind", "all") for q in qs})
    for m in METHODS:
        hits, rr, misses, by_kind = 0, 0.0, [], {k: 0 for k in kinds}
        for item in qs:
            secs = [r[1] for r in top(db, m, item["q"], n)]
            pos = next((i for i, s in enumerate(secs, 1) if s in item["expect"]), None)
            hits += pos is not None; rr += 1 / pos if pos else 0
            if pos: by_kind[item.get("kind", "all")] += 1
            else: misses.append(item["q"])
        res[m] = (hits, rr / len(qs), misses)
        split = "   ".join(f"{k} {by_kind[k]}/{sum(q.get('kind', 'all') == k for q in qs)}" for k in kinds)
        print(f"{m:8} hit@{n} {hits}/{len(qs)} ({hits / len(qs):.0%})   MRR {rr / len(qs):.2f}   [{split}]")
    for m, (_, _, misses) in res.items():
        print(f"\n{m} missed:"); [print("  -", q) for q in misses]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "ingest": ingest()
    elif cmd == "search": search(" ".join(sys.argv[2:]))
    elif cmd == "ask": ask(" ".join(sys.argv[2:]))
    elif cmd == "eval": evaluate()
    else: print(__doc__)
