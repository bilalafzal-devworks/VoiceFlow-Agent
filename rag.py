"""rag.py - the Retrieval-Augmented Generation knowledge base.

Pipeline:  upload (PDF/TXT/MD) -> extract text -> chunk -> embed -> store in ChromaDB
           query -> embed -> top-k similar chunks (with source + page metadata)

Embeddings come from Hugging Face (sentence-transformers/all-MiniLM-L6-v2) through the
free Inference API (needs HF_TOKEN). Set EMBEDDING_BACKEND=local to run the same model
on your own machine instead (pip install sentence-transformers).
"""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import requests

import config

CHARS_PER_TOKEN = 4  # rough rule of thumb for English text

_collection = None
_local_model = None


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


# -------------------------------------------------------------- embeddings
def _pool(data):
    """Hugging Face returns n x d vectors (or n x tokens x d). Make it n x d."""
    vectors = []
    for item in data:
        if item and isinstance(item[0], list):  # token-level -> mean pool
            dim = len(item[0])
            vectors.append([sum(tok[j] for tok in item) / len(item) for j in range(dim)])
        else:
            vectors.append(item)
    return vectors


def _embed_hf_api(texts):
    token = config.hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Add it to .env (or Streamlit secrets).")
    url = config.HF_EMBED_URL.format(model=config.EMBED_MODEL)
    headers = {"Authorization": f"Bearer {token}"}
    vectors = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        batch = texts[i : i + config.EMBED_BATCH]
        for attempt in range(4):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json={"inputs": batch, "options": {"wait_for_model": True}},
                    timeout=60,
                )
            except requests.RequestException as exc:
                raise RuntimeError(f"Could not reach Hugging Face: {exc}") from exc
            if resp.status_code == 200:
                break
            if resp.status_code in (429, 503):  # busy / model loading: wait and retry
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Hugging Face embedding failed ({resp.status_code}): {resp.text[:200]}")
        else:
            raise RuntimeError("Hugging Face embedding is still busy after several retries. Try again.")
        vectors.extend(_pool(resp.json()))
    return vectors


def _embed_local(texts):
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(config.EMBED_MODEL)
    return _local_model.encode(texts, normalize_embeddings=True).tolist()


def embed_texts(texts):
    """Return one embedding vector (list of floats) per input text."""
    if not texts:
        return []
    if config.EMBEDDING_BACKEND == "local":
        return _embed_local(texts)
    return _embed_hf_api(texts)


# ------------------------------------------------------------ vector store
def _get_collection():
    global _collection
    if _collection is None:
        import chromadb

        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def reset_handles():
    """Forget the cached collection (used by tests after changing CHROMA_DIR)."""
    global _collection
    _collection = None


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
    embeddings = embed_texts(docs)
    _get_collection().upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    return len(docs)


def search(query, k=None):
    """Return the top-k chunks for a query: [{id, text, source, page, distance}]."""
    col = _get_collection()
    total = col.count()
    if total == 0:
        return []
    k = min(k or config.TOP_K, total)
    query_vec = embed_texts([query])[0]
    res = col.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc_id, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "id": doc_id,
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 1),
                "distance": round(float(dist), 4),
            }
        )
    return hits


def count():
    return _get_collection().count()


def list_sources():
    """Return {filename: number_of_chunks}."""
    metas = _get_collection().get(include=["metadatas"])["metadatas"] or []
    sources = {}
    for meta in metas:
        name = meta.get("source", "unknown")
        sources[name] = sources.get(name, 0) + 1
    return dict(sorted(sources.items()))


def delete_source(filename):
    _get_collection().delete(where={"source": filename})


def clear():
    """Delete everything in the knowledge base."""
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    reset_handles()
