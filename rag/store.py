"""
rag/store.py
============
The one place that reads/writes the FAISS index + chunk metadata.

  models/faiss_index/index.faiss    vectors
  models/faiss_index/metadata.json  one dict per vector (same order)

Why a store module: before, ingest.py (rebuild from .txt) and pdf_ingest.py (append)
both wrote the index with different assumptions, so re-running ingest.py silently threw
away every uploaded PDF, and uploading the same PDF twice doubled its chunks.

Chunk dict fields: chunk_id, concept_id, text, source ("txt" | "pdf"), and for PDFs
also doc_id, section_id, section_title, page_start, page_end.
"""
import json
import os
import threading
from pathlib import Path

import numpy as np

from backend.config import INDEX_DIR, INDEX_PATH, INDEX_META_PATH, EMBEDDING_MODEL

DIM = 384                                   # all-MiniLM-L6-v2
_lock = threading.RLock()
_embedder = None
_cache = {"stamp": None, "index": None, "meta": []}


# ------------------------------------------------------------------- embeddings
def embed_texts(texts, batch_size: int = 32) -> np.ndarray:
    """L2-normalised float32 vectors (so inner product == cosine similarity)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    vecs = _embedder.encode(list(texts), batch_size=batch_size,
                            normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


# ----------------------------------------------------------------- disk helpers
def _atomic_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)                   # never leaves a half-written file behind


def _stamp():
    try:
        return (INDEX_PATH.stat().st_mtime_ns, INDEX_META_PATH.stat().st_mtime_ns)
    except FileNotFoundError:
        return None


def load():
    """-> (index, meta); (None, []) if nothing has been indexed yet."""
    import faiss
    with _lock:
        stamp = _stamp()
        if stamp is None:
            return None, []
        if _cache["stamp"] == stamp:
            return _cache["index"], _cache["meta"]
        index = faiss.deserialize_index(np.frombuffer(INDEX_PATH.read_bytes(), dtype="uint8"))
        meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        if index.ntotal != len(meta):
            raise RuntimeError(
                f"index has {index.ntotal} vectors but metadata has {len(meta)} chunks - "
                "delete models/faiss_index and re-add your notes")
        _cache.update(stamp=stamp, index=index, meta=meta)
        return index, meta


def _save(index, meta):
    import faiss
    _atomic_write(INDEX_PATH, faiss.serialize_index(index).tobytes())
    _atomic_write(INDEX_META_PATH, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    _cache["stamp"] = None                  # force a reload on next read


# -------------------------------------------------------------------- mutations
def add_chunks(chunks: list) -> int:
    """Embed and append chunks.  Creates the index on first use."""
    if not chunks:
        return 0
    import faiss
    vecs = embed_texts([c.get("embed_text") or c["text"] for c in chunks])
    chunks = [{k: v for k, v in c.items() if k != "embed_text"} for c in chunks]   # not stored
    with _lock:
        index, meta = load()
        if index is None:
            index, meta = faiss.IndexFlatIP(vecs.shape[1]), []
        index.add(vecs)
        meta = list(meta) + list(chunks)
        _save(index, meta)
    return len(chunks)


def remove_where(predicate) -> int:
    """Delete every chunk for which predicate(chunk_dict) is True.  No re-embedding:
    the kept vectors are read back out of the existing index."""
    import faiss
    with _lock:
        index, meta = load()
        if index is None or not meta:
            return 0
        keep = [i for i, m in enumerate(meta) if not predicate(m)]
        removed = len(meta) - len(keep)
        if removed == 0:
            return 0
        old = index.reconstruct_n(0, index.ntotal)
        new = faiss.IndexFlatIP(old.shape[1])
        if keep:
            new.add(np.ascontiguousarray(old[keep]))
        _save(new, [meta[i] for i in keep])
        return removed


# ------------------------------------------------------------------------ reads
def search(query: str, k: int = 4, concept_ids=None, doc_id=None) -> list:
    """Top-k chunks (best first) optionally limited to some concepts or one document."""
    index, meta = load()
    if index is None or index.ntotal == 0 or not query.strip():
        return []
    q = embed_texts([query])
    wide = min(index.ntotal, max(k * 12, 60))      # look wide, then filter
    scores, ids = index.search(q, wide)
    out = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        m = meta[i]
        if concept_ids and m.get("concept_id") not in concept_ids:
            continue
        if doc_id and m.get("doc_id") != doc_id:
            continue
        out.append({**m, "similarity": float(score)})
        if len(out) >= k:
            break
    return out


def is_ready() -> bool:
    try:
        index, meta = load()
        return index is not None and len(meta) > 0
    except Exception:
        return False


def stats() -> dict:
    """{'total': n, 'pdf': n, 'txt': n, 'by_concept': {...}, 'by_doc': {...}}"""
    try:
        _, meta = load()
    except Exception:
        meta = []
    by_concept, by_doc = {}, {}
    for m in meta:
        by_concept[m.get("concept_id", "?")] = by_concept.get(m.get("concept_id", "?"), 0) + 1
        if m.get("doc_id"):
            by_doc[m["doc_id"]] = by_doc.get(m["doc_id"], 0) + 1
    pdf = sum(1 for m in meta if m.get("source") == "pdf")
    return {"total": len(meta), "pdf": pdf, "txt": len(meta) - pdf,
            "by_concept": by_concept, "by_doc": by_doc}


def status_text() -> tuple:
    """(ok, message) for a status pill - use this instead of checking file paths."""
    s = stats()
    if s["total"] == 0:
        return False, "Empty - upload a PDF in My Notes"
    return True, f"Ready - {s['total']} chunks"
