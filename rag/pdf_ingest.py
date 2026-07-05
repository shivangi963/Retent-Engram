"""
rag/pdf_ingest.py
==================
IMPROVEMENT 1 — PDF Ingestion

PURPOSE
-------
Lets you upload your actual VTU notes PDFs instead of manually
writing .txt files. The pipeline:

  1. Extracts text from every page of the PDF
  2. Cleans the text (removes headers, page numbers, artifacts)
  3. Auto-detects which concepts are covered by keyword matching
  4. Chunks the text into ~300-word pieces
  5. Adds those chunks into the existing FAISS index
  6. Updates metadata.json with the new chunks

WHY pdfplumber?
  - Handles multi-column layouts better than PyPDF2
  - Preserves table structure as text
  - Works reliably on scanned-text PDFs (not image-only)
  - Lightweight: no system dependencies

INSTALL:
  pip install pdfplumber

HOW CONCEPT AUTO-DETECTION WORKS:
  Each concept in concepts.json has a name and aliases.
  We scan the extracted text for those keywords.
  Pages mentioning "deadlock" or "process scheduling" → tagged as "os".
  This isn't perfect, but it's good enough to chunk by topic.

CALLED BY:
  frontend/pages/8_pdf_upload.py (the upload UI)
  OR directly: python rag/pdf_ingest.py --file notes.pdf --concept os
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import re
import json
import argparse
import numpy as np

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES_DIR     = os.path.join(PROJECT_ROOT, "data", "notes")
INDEX_DIR     = os.path.join(PROJECT_ROOT, "models", "faiss_index")
INDEX_PATH    = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
CONCEPTS_PATH = os.path.join(PROJECT_ROOT, "data", "concepts.json")

# Keywords that help identify which concept a page belongs to
CONCEPT_KEYWORDS = {
    "os":            ["operating system", "process", "deadlock", "scheduling",
                      "memory management", "paging", "semaphore", "mutex",
                      "virtual memory", "cpu scheduling", "thread"],
    "dbms":          ["database", "sql", "normalization", "transaction",
                      "acid", "b-tree", "indexing", "relational", "er diagram",
                      "join", "functional dependency"],
    "cn":            ["network", "tcp", "ip address", "routing", "protocol",
                      "osi", "ethernet", "dns", "http", "subnet", "udp"],
    "dsa":           ["data structure", "algorithm", "linked list", "tree",
                      "graph", "sorting", "dynamic programming", "recursion",
                      "binary search", "hash", "stack", "queue"],
    "python_oop":    ["class", "object", "inheritance", "polymorphism",
                      "encapsulation", "abstraction", "method", "__init__",
                      "decorator", "property"],
    "sql":           ["select", "insert", "update", "delete", "join",
                      "group by", "having", "aggregate", "subquery", "index"],
    "process_mgmt":  ["process creation", "fork", "exec", "ipc", "pipe",
                      "process state", "pcb", "context switch"],
    "memory_mgmt":   ["page replacement", "lru", "fifo", "thrashing",
                      "frame", "segmentation", "fragmentation", "tlb"],
}


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> list:
    """
    Extracts text from every page of a PDF.

    Returns list of (page_number, text) tuples.
    Pages with no usable text (images, blank pages) are skipped.

    Args:
        pdf_path: absolute path to the PDF file

    Returns:
        list of (page_num, text) tuples
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber not installed. Run: pip install pdfplumber"
        )

    pages = []
    print(f"\n📄  Extracting text from: {os.path.basename(pdf_path)}")

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"    Total pages: {total_pages}")

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if not text or len(text.strip()) < 50:
                # Skip pages with very little text (likely images or blank)
                continue

            text = clean_extracted_text(text)
            pages.append((i + 1, text))

        print(f"    Usable pages extracted: {len(pages)}")

    return pages


def clean_extracted_text(text: str) -> str:
    """
    Cleans raw PDF text by removing common artifacts.

    WHAT IT REMOVES:
      - Page numbers (standalone digits or "Page X of Y")
      - Headers/footers (short lines repeated across pages)
      - Excessive whitespace
      - Unicode artifacts from PDF encoding

    Args:
        text: raw text from pdfplumber

    Returns:
        str: cleaned text
    """
    # Remove "Page X" or "X of Y" patterns
    text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)

    # Remove standalone numbers (page numbers at top/bottom)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove excessive blank lines (more than 2 in a row)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are just dashes or underscores (dividers)
    text = re.sub(r'^[-_=]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Fix common PDF spacing issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # "camelCase" → "camel Case"

    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)

    return text.strip()


# =============================================================================
# CONCEPT AUTO-DETECTION
# =============================================================================

def detect_concept_from_text(text: str, min_keyword_matches: int = 2) -> str | None:
    """
    Detects which concept a piece of text most likely belongs to
    by counting keyword matches.

    HOW IT WORKS:
      For each concept in CONCEPT_KEYWORDS, count how many of its
      keywords appear in the text (case-insensitive).
      The concept with the highest count wins.
      If no concept gets at least min_keyword_matches hits → return None.

    EXAMPLE:
      text contains: "deadlock", "process", "semaphore"
      os → 3 hits
      dbms → 0 hits
      → returns "os"

    Args:
        text:                 text to analyse
        min_keyword_matches:  minimum matches required to tag a concept

    Returns:
        str: concept_id if detected, None if unclear
    """
    text_lower = text.lower()
    scores = {}

    for concept_id, keywords in CONCEPT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[concept_id] = score

    best_concept = max(scores, key=scores.get)
    best_score   = scores[best_concept]

    if best_score >= min_keyword_matches:
        return best_concept

    return None   # could not confidently tag this text


# =============================================================================
# CHUNK AND EMBED
# =============================================================================

def chunk_pages_into_pieces(pages: list, concept_id: str,
                             chunk_size: int = 300,
                             overlap: int    = 50) -> list:
    """
    Combines all page texts into one string, then chunks it.

    SAME CHUNKING LOGIC AS ingest.py:
      We reuse the same approach so chunks from PDFs are compatible
      with chunks from .txt files in the same FAISS index.

    Args:
        pages:      list of (page_num, text) tuples
        concept_id: which concept these pages belong to
        chunk_size: words per chunk
        overlap:    word overlap between adjacent chunks

    Returns:
        list of chunk dicts (same format as ingest.py)
    """
    # Combine all page texts
    full_text = "\n\n".join(text for _, text in pages)
    words = full_text.split()

    chunks = []
    step = chunk_size - overlap
    chunk_index = 0

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break

        chunks.append({
            "chunk_id":    f"{concept_id}_pdf_chunk_{chunk_index}",
            "concept_id":  concept_id,
            "text":        " ".join(chunk_words),
            "word_count":  len(chunk_words),
            "chunk_index": chunk_index,
            "source":      "pdf"   # marks this as from PDF, not .txt
        })
        chunk_index += 1

        if end >= len(words):
            break

    return chunks


def embed_and_add_to_index(new_chunks: list) -> bool:
    """
    Embeds new chunks and ADDS them to the existing FAISS index.

    IMPORTANT — APPEND, NOT REBUILD:
      We don't rebuild the entire index. We just add new vectors.
      This is fast (only embeds the new chunks) and non-destructive
      (existing .txt file chunks remain in the index).

    HOW:
      1. Load existing index + metadata
      2. Embed new chunks
      3. index.add(new_embeddings)
      4. Append new chunks to metadata list
      5. Save both

    Args:
        new_chunks: list of chunk dicts from chunk_pages_into_pieces()

    Returns:
        bool: True if successful
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    if not os.path.exists(INDEX_PATH):
        print("⚠️  No existing index found. Run python rag/ingest.py first.")
        return False

    # Load existing index and metadata
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    print(f"    Existing index: {index.ntotal} vectors")

    # Embed new chunks
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in new_chunks]

    print(f"    Embedding {len(texts)} new chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32
    ).astype("float32")

    # Add to index
    index.add(embeddings)
    metadata.extend(new_chunks)

    # Save updated index and metadata
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"    New index total: {index.ntotal} vectors")
    print(f"    ✅  PDF chunks added to knowledge base!")
    return True


# =============================================================================
# SAVE TEXT VERSION (optional)
# =============================================================================

def save_as_txt(pages: list, concept_id: str):
    """
    Optionally saves the extracted PDF text as a .txt file in data/notes/.
    This means future runs of ingest.py will include this content too.

    Args:
        pages:      list of (page_num, text) tuples
        concept_id: which concept
    """
    os.makedirs(NOTES_DIR, exist_ok=True)
    output_path = os.path.join(NOTES_DIR, f"{concept_id}_pdf_extracted.txt")

    full_text = "\n\n".join(
        f"[Page {pnum}]\n{text}" for pnum, text in pages
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"CONCEPT: {concept_id.upper()}\nSOURCE: PDF\n\n{full_text}")

    print(f"    💾  Also saved to: {output_path}")


# =============================================================================
# MASTER FUNCTION — called by the upload UI
# =============================================================================

def ingest_pdf(pdf_path: str,
               concept_id: str = None,
               save_txt: bool = True) -> dict:
    """
    Complete pipeline: PDF file → FAISS index.

    Called by frontend/pages/8_pdf_upload.py

    STEPS:
      1. Extract text from each PDF page
      2. Auto-detect concept if not provided
      3. Chunk into 300-word pieces
      4. Embed and add to existing FAISS index
      5. Optionally save as .txt in data/notes/

    RETURN:
      {
        "success":      True,
        "concept_id":   "os",
        "pages":        24,
        "chunks":       18,
        "error":        None
      }

    Args:
        pdf_path:   path to the PDF file
        concept_id: concept to tag (if None, auto-detected)
        save_txt:   whether to also save extracted text as .txt

    Returns:
        dict with result info
    """
    result = {
        "success":    False,
        "concept_id": concept_id,
        "pages":      0,
        "chunks":     0,
        "error":      None
    }

    try:
        # Step 1: Extract text
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            result["error"] = "No usable text extracted. PDF may be image-only."
            return result

        result["pages"] = len(pages)

        # Step 2: Auto-detect concept if not specified
        if concept_id is None:
            combined = " ".join(text for _, text in pages[:5])  # check first 5 pages
            concept_id = detect_concept_from_text(combined)
            if concept_id is None:
                result["error"] = (
                    "Could not auto-detect concept. "
                    "Please specify the concept manually."
                )
                return result
            print(f"    🏷️   Auto-detected concept: '{concept_id}'")
            result["concept_id"] = concept_id

        # Step 3: Chunk
        chunks = chunk_pages_into_pieces(pages, concept_id)
        result["chunks"] = len(chunks)
        print(f"    ✂️   Created {len(chunks)} chunks")

        # Step 4: Embed and add to index
        success = embed_and_add_to_index(chunks)
        if not success:
            result["error"] = "Failed to add chunks to index."
            return result

        # Step 5: Save .txt
        if save_txt:
            save_as_txt(pages, concept_id)

        result["success"] = True
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


# =============================================================================
# CLI usage
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into Retent Engram")
    parser.add_argument("--file",    required=True,  help="Path to PDF file")
    parser.add_argument("--concept", required=False, help="Concept ID (auto-detected if omitted)")
    parser.add_argument("--no-txt",  action="store_true", help="Skip saving .txt file")
    args = parser.parse_args()

    r = ingest_pdf(
        pdf_path   = args.file,
        concept_id = args.concept,
        save_txt   = not args.no_txt
    )

    if r["success"]:
        print(f"\n✅  Done! Ingested {r['pages']} pages → {r['chunks']} chunks → concept '{r['concept_id']}'")
    else:
        print(f"\n❌  Failed: {r['error']}")