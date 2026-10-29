"""Download 8 chapters of the PostgreSQL 16 documentation and split them into chunks.

Output: data/chunks.jsonl, one chunk per line: {id, chapter, section, url, text}
The docs are (c) PostgreSQL Global Development Group, under the PostgreSQL License.
"""
import html, json, re, sys, urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/postgres/postgres/REL_16_STABLE/doc/src/sgml/"
CHAPTERS = ["indices", "perform", "mvcc", "maintenance", "textsearch", "ddl", "queries", "backup"]
MAX_CHARS = 1200          # target chunk size; long sections are split on paragraph boundaries
OUT = Path(__file__).parent / "data" / "chunks.jsonl"


def fetch(name, cache=Path(__file__).parent / "data" / "sgml"):
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{name}.sgml"
    if not p.exists():
        urllib.request.urlretrieve(BASE + p.name, p)
    return p.read_text(encoding="utf8")


def to_text(sgml):
    s = re.sub(r"<!--.*?-->", " ", sgml, flags=re.S)
    s = re.sub(r"<indexterm.*?</indexterm>", " ", s, flags=re.S)
    s = re.sub(r"</?(para|simpara)[^>]*>", "\n\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"&([a-z]+);", lambda m: {"mdash": "-", "ndash": "-", "nbsp": " "}.get(m.group(1), html.unescape(m.group(0))), s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n\s*", "\n\n", s)
    return s.strip()


def sections(sgml):
    """Yield (url, title, body) for every sect1/sect2/sect3, each without its nested subsections.
    sect1 is its own web page; sect2/sect3 are anchors on their sect1 page."""
    for s1 in re.finditer(r"<sect1 id=\"([^\"]+)\".*?</sect1>", sgml, re.S):
        page = s1.group(1)
        pat = re.compile(r"<(sect[123]) id=\"([^\"]+)\"[^>]*>\s*<title>(.*?)</title>(.*?)(?=<sect[123] id=|</sect[123]>)", re.S)
        for m in pat.finditer(s1.group(0)):
            title = re.sub(r"<[^>]+>", "", m.group(3)).strip()
            url = f"https://www.postgresql.org/docs/16/{page}.html" + ("" if m.group(1) == "sect1" else f"#{m.group(2).upper()}")
            yield url, title, to_text(m.group(4))


def split(text):
    out, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) > MAX_CHARS:
            out.append(cur.strip()); cur = ""
        cur += para + "\n\n"
    if cur.strip(): out.append(cur.strip())
    return out


def main():
    rows = []
    for ch in CHAPTERS:
        sgml = fetch(ch)
        chapter = re.search(r"<title>(.*?)</title>", sgml, re.S).group(1).strip()
        for url, title, body in sections(sgml):
            if len(body) < 200: continue                       # skip empty wrappers
            for part in split(body):
                rows.append({"id": len(rows) + 1, "chapter": chapter, "section": title, "url": url, "text": part})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} chunks from {len(CHAPTERS)} chapters -> {OUT}")


if __name__ == "__main__":
    main()
