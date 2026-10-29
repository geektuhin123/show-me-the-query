"""Search by meaning with word embeddings (Show Me the Query: "Embeddings: how computers search by meaning").

Needs Python 3 + numpy. First run downloads GloVe 6B 50d vectors (~66 MB, PDDL licence) from the
gensim-data release on GitHub and caches them in ~/.cache/glove.

    pip install numpy
    python3 search_by_meaning.py
    python3 search_by_meaning.py "your own query here"
"""
import gzip, sys, urllib.request
from pathlib import Path
import numpy as np

URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
CACHE = Path.home() / ".cache" / "glove" / "glove-wiki-gigaword-50.gz"
STOP = set("a an the to of in on at for and or is are was be my your our it its this that with how why what do does i you".split())

DOCS = [
    "how to train a dog to stop barking",
    "best espresso beans for a home coffee machine",
    "fixing a flat tire on a mountain bike",
    "why does my cat scratch the sofa",
    "cheap flights to tokyo in spring",
    "laptop battery drains too fast",
    "how to bake sourdough bread at home",
    "stop a kitten from biting your hands",
]


def load_vectors():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("downloading GloVe vectors (66 MB, once)...", file=sys.stderr)
        urllib.request.urlretrieve(URL, CACHE)
    vecs = {}
    with gzip.open(CACHE, "rt", encoding="utf8") as f:
        next(f)                                   # header line: "400000 50"
        for line in f:
            word, *nums = line.rstrip().split(" ")
            vecs[word] = np.asarray(nums, dtype=np.float32)
    return vecs


def words(text):
    return [w for w in text.lower().replace("?", "").split() if w not in STOP]


def embed(text, vecs):
    """Sentence vector = average of its word vectors (the simplest sentence embedding)."""
    found = [vecs[w] for w in words(text) if w in vecs]
    return np.mean(found, axis=0)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def keyword_search(query, docs):
    q = set(words(query))
    return [(len(q & set(words(d))), d) for d in docs if q & set(words(d))]


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "cheap dark roast for my morning brew"
    vecs = load_vectors()
    print(f"{len(vecs):,} words x {len(vecs['coffee'])} numbers each")
    print("coffee =", np.round(vecs["coffee"][:6], 2), "... (50 numbers)")

    print("\nword similarity (cosine):")
    for a, b in [("dog", "puppy"), ("dog", "cat"), ("dog", "laptop"), ("coffee", "espresso"), ("coffee", "tire")]:
        print(f"  {a:>7} ~ {b:<9} {cosine(vecs[a], vecs[b]):.2f}")

    print(f"\nquery: {query!r}")
    hits = keyword_search(query, DOCS)
    print("keyword search:", hits if hits else "no results (no shared words)")

    q = embed(query, vecs)
    ranked = sorted(((cosine(q, embed(d, vecs)), d) for d in DOCS), reverse=True)
    print("embedding search:")
    for score, d in ranked:
        print(f"  {score:.2f}  {d}")

    # 2D map: project the sentence vectors onto their two main directions (PCA) for plotting
    m = np.array([embed(d, vecs) for d in DOCS] + [q]); m = m - m.mean(axis=0)
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    xy = m @ vt[:2].T
    print("\n2D map (PCA of the sentence vectors):")
    for (x, y), label in zip(xy, DOCS + ["QUERY: " + query]):
        print(f"  {x:+.2f} {y:+.2f}  {label}")


if __name__ == "__main__":
    main()
