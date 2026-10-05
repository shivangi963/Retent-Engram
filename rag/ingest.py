"""
rag/ingest.py
=============
Indexes the hand-written notes in data/notes/*.txt  (os.txt, dbms.txt, cn.txt ...).

  python rag/ingest.py

Safe to re-run any time: it replaces only the .txt-sourced chunks.  Your uploaded PDFs
(managed from the My Notes page) are left alone - the first version of this script
rebuilt the whole index and silently dropped them.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # allow `python rag/ingest.py`

from backend.config import NOTES_DIR, CONCEPTS_PATH
from rag import store

CHUNK_WORDS, OVERLAP = 300, 50


def _concept_ids() -> set:
    try:
        return {c["concept_id"] for c in json.loads(Path(CONCEPTS_PATH).read_text(encoding="utf-8"))}
    except Exception:
        return set()


def _chunks_for(text: str, concept_id: str) -> list:
    words, out, step = text.split(), [], CHUNK_WORDS - OVERLAP
    for k, start in enumerate(range(0, max(len(words), 1), step)):
        piece = words[start:start + CHUNK_WORDS]
        if not piece:
            break
        out.append({"chunk_id": f"{concept_id}_chunk_{k}", "concept_id": concept_id, "source": "txt",
                    "text": " ".join(piece), "word_count": len(piece), "chunk_index": k})
        if start + CHUNK_WORDS >= len(words):
            break
    return out


def sync_text_notes() -> int:
    """Re-index data/notes/*.txt.  Returns the number of chunks written."""
    known = _concept_ids()
    chunks = []
    for f in sorted(Path(NOTES_DIR).glob("*.txt")):
        if re.search(r"_pdf_extracted$", f.stem):          # leftovers of the first PDF importer
            continue
        if known and f.stem not in known:
            print(f"  skipping {f.name}: '{f.stem}' is not a concept_id in data/concepts.json")
            continue
        text = f.read_text(encoding="utf-8").strip()
        if text:
            chunks.extend(_chunks_for(text, f.stem))
    store.remove_where(lambda m: m.get("source", "txt") == "txt")
    store.add_chunks(chunks)
    return len(chunks)


run_ingestion = sync_text_notes                              # old name


def is_index_available() -> bool:                            # old name, still imported by Study page
    return store.is_ready()


def load_index_and_model():                                  # old name; (index, metadata, embedder)
    index, meta = store.load()
    return index, meta, None


if __name__ == "__main__":
    n = sync_text_notes()
    s = store.stats()
    print(f"Indexed {n} text-note chunks. Knowledge base now has {s['total']} chunks "
          f"({s['pdf']} from PDFs, {s['txt']} from .txt notes).")
