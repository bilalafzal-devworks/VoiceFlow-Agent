"""rag.py - the Retrieval-Augmented Generation knowledge base (Groq only, no Hugging Face).

Pipeline:  upload (PDF/TXT/MD) -> extract text -> chunk -> store (JSON file, with page metadata)
           query -> BM25 keyword search (pure Python) -> Groq LLM reranks the candidates
                 -> top-k chunks (with source + page metadata)

Groq has no embeddings endpoint, so retrieval is done in two steps: fast BM25 keyword
scoring picks candidate chunks, then a Groq model (GROQ_MODEL) reads them and orders them by
how well they answer the question. No extra API key, no model download. If the Groq rerank
call fails, the BM25 order is used, so search never breaks.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import threading
from collections import Counter
from pathlib import Path

import config

CHARS_PER_TOKEN = 4  # rough rule of thumb for English text

active_backend = "BM25 + Groq rerank"  # shown in the UI

_lock = threading.Lock()
_store = None  # {chunk_id: {"text", "source", "page", "chunk"}}
_STOPWORDS = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were be been being "
    "it its this that these those i you he she we they what which who whom how why when where "
    "do does did can could should would will about into than then so not no".split()
)


# ---------------------------------------------------------------- chunking
def chunk_text(text, chunk_tokens=None, overlap_tokens=None):
    """Split text into overlapping chunks of roughly `chunk_tokens` tokens.

    We try to cut at a paragraph or sentence end near the end of each window so
    chunks do not stop mid-sentence.
    """
    chunk_tokens = chunk_tokens or config.CHUNK_TOKENS
    overlap_tokens = config.CHUNK_OVERLAP_TOKENS if overlap_tokens is None else overlap_tokens
    size = chunk_tokens * CHARS_PER_TOKEN
    overlap = min(overlap_tokens * CHARS_PER_TOKEN, size // 2)

    text = " ".join(text.split()) if "\n\n" not in text else text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window_start = start + int(size * 0.8)
            cut = max(text.rfind("\n\n", window_start, end), text.rfind(". ", window_start, end))
            if cut != -1:
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # always move forward
    return chunks


# ------------------------------------------------------------ chunk store
def _store_path():
    return Path(config.CHROMA_DIR) / "chunks.json"


def _load():
    global _store
    if _store is None:
        path = _store_path()
        try:
            _store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError):
            _store = {}
    return _store


def _save():
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_store), encoding="utf-8")
    tmp.replace(path)


def reset_handles():
    """Forget the cached store (used by tests after changing CHROMA_DIR)."""
    global _store
    _store = None


# ---------------------------------------------------------------- BM25
def _tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _bm25(query, chunks, k1=1.5, b=0.75):
    """Score every chunk against the query. Returns [(score, chunk_id)] best first."""
    q_terms = set(_tokens(query))
    if not q_terms or not chunks:
        return []
    docs = {cid: _tokens(c["text"]) for cid, c in chunks.items()}
    n = len(docs)
    avg_len = (sum(len(d) for d in docs.values()) / n) or 1.0
    df = Counter(t for d in docs.values() for t in set(d) if t in q_terms)
    scored = []
    for cid, toks in docs.items():
        tf = Counter(toks)
        score = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(toks) / avg_len))
        if score > 0:
            scored.append((score, cid))
    scored.sort(reverse=True)
    return scored


# ------------------------------------------------------------ Groq rerank
def _rerank_with_groq(query, candidates, k):
    """Ask Groq to pick the best chunk numbers. Returns an ordered list of indexes or None."""
    api_key = config.groq_api_key()
    if not api_key or len(candidates) <= 1:
        return None
    try:
        from groq import Groq

        listing = "\n\n".join(f"[{i}] {c['text'][:700]}" for i, c in enumerate(candidates))
        resp = Groq(api_key=api_key).chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You rank passages for a retrieval system. Reply with JSON only: "
                    '{"ranked": [passage numbers, most relevant first]}. Include only passages '
                    "that help answer the question.",
                },
                {"role": "user", "content": f"Question: {query}\n\nPassages:\n{listing}\n\nReturn the top {k}."},
            ],
        )
        text = resp.choices[0].message.content or ""
        match = re.search(r"\{.*\}", text, re.S)
        ranked = json.loads(match.group(0)).get("ranked", []) if match else []
        order = []
        for i in ranked:
            if isinstance(i, int) and 0 <= i < len(candidates) and i not in order:
                order.append(i)
        return order or None
    except Exception:
        return None  # rate limit, network, bad JSON: fall back to BM25 order


def extract_pages(filename, data):
    """Return a list of (page_number, text) for a PDF / TXT / MD file."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if ext in (".txt", ".md"):
        return [(1, data.decode("utf-8", errors="replace"))]
    raise ValueError(f"Unsupported file type '{ext}'. Upload PDF, TXT or MD files.")


def ingest_file(filename, data):
    """Chunk, embed and store a file. Returns the number of chunks stored.

    Chunks are made per page so every chunk keeps a correct page number.
    Re-uploading the same file is safe (chunks are upserted by a content hash).
    """
    ids, docs, metas = [], [], []
    for page_no, page_text in extract_pages(filename, data):
        for idx, chunk in enumerate(chunk_text(page_text)):
            digest = hashlib.sha1(f"{filename}|{page_no}|{idx}|{chunk}".encode()).hexdigest()[:20]
            ids.append(digest)
            docs.append(chunk)
            metas.append({"source": filename, "page": page_no, "chunk": idx})
    if not docs:
        raise ValueError(f"No readable text found in '{filename}' (scanned PDF? try OCR first).")
    with _lock:
        store = _load()
        for cid, doc, meta in zip(ids, docs, metas):
            store[cid] = {"text": doc, **meta}
        _save()
    return len(docs)


def search(query, k=None):
    """Return the top-k chunks for a query: [{id, text, source, page, distance}].

    `distance` is kept for the UI/prompts: lower = better (1 / (1 + BM25 score)).
    """
    k = k or config.TOP_K
    store = _load()
    scored = _bm25(query, store)[: max(k * 3, 12)]
    if not scored:
        return []
    candidates = [dict(store[cid], id=cid, score=score) for score, cid in scored]
    order = _rerank_with_groq(query, candidates, k)
    if order:
        candidates = [candidates[i] for i in order]
    return [
        {
            "id": c["id"],
            "text": c["text"],
            "source": c.get("source", "unknown"),
            "page": c.get("page", 1),
            "distance": round(1 / (1 + c["score"]), 4),
        }
        for c in candidates[:k]
    ]


def count():
    return len(_load())


def list_sources():
    """Return {filename: number_of_chunks}."""
    sources = {}
    for c in _load().values():
        name = c.get("source", "unknown")
        sources[name] = sources.get(name, 0) + 1
    return dict(sorted(sources.items()))


def delete_source(filename):
    with _lock:
        store = _load()
        for cid in [cid for cid, c in store.items() if c.get("source") == filename]:
            del store[cid]
        _save()


def clear():
    """Delete everything in the knowledge base."""
    with _lock:
        _load().clear()
        _save()
