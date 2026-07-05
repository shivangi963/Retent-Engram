import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES_DIR     = os.path.join(PROJECT_ROOT, "data", "notes")
INDEX_DIR     = os.path.join(PROJECT_ROOT, "models", "faiss_index")
INDEX_PATH    = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

# Sentence-transformers model name
# all-MiniLM-L6-v2: small, fast, good quality
# Downloads ~80MB on first run, cached to ~/.cache/torch/
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunk size in words
CHUNK_SIZE_WORDS = 300

# Overlap between chunks in words
# Overlap ensures context from the end of one chunk appears in the start of next
# This prevents important sentences being split across chunks
CHUNK_OVERLAP_WORDS = 50

# Mapping: filename stem → concept_id (must match concepts.json)
FILENAME_TO_CONCEPT = {
    "os":          "os",
    "dbms":        "dbms",
    "cn":          "cn",
    "dsa":         "dsa",
    "python_oop":  "python_oop",
    "sql":         "sql",
    "recursion":   "recursion",
    "binary_search": "binary_search",
    "process_mgmt": "process_mgmt",
    "memory_mgmt":  "memory_mgmt",
}


# =============================================================================
# STEP 1 — READ NOTE FILES
# =============================================================================

def read_note_files() -> list:
    """
    Reads all .txt files from data/notes/ and returns a list of
    (concept_id, full_text) tuples.

    HOW IT WORKS:
      1. List all .txt files in data/notes/
      2. For each file, read its content
      3. Look up concept_id from FILENAME_TO_CONCEPT
      4. Return list of (concept_id, text) pairs

    SKIPS:
      Files not in FILENAME_TO_CONCEPT mapping are skipped with a warning.
      This prevents accidentally indexing unrelated files.

    Returns:
        list of tuples: [(concept_id, text), ...]
    """
    if not os.path.exists(NOTES_DIR):
        raise FileNotFoundError(
            f"Notes directory not found: {NOTES_DIR}\n"
            f"Create data/notes/ and add your concept .txt files."
        )

    txt_files = [f for f in os.listdir(NOTES_DIR) if f.endswith(".txt")]

    if not txt_files:
        raise ValueError(
            f"No .txt files found in {NOTES_DIR}\n"
            f"Add concept notes as .txt files."
        )

    documents = []
    print(f"\n📂  Reading notes from {NOTES_DIR}")

    for filename in sorted(txt_files):
        # Get concept_id from filename (without .txt extension)
        stem = filename.replace(".txt", "")
        concept_id = FILENAME_TO_CONCEPT.get(stem)

        if concept_id is None:
            print(f"  ⚠️  Skipping '{filename}' (not in FILENAME_TO_CONCEPT mapping)")
            continue

        # Read file content
        filepath = os.path.join(NOTES_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"  ⚠️  Skipping '{filename}' (empty file)")
            continue

        documents.append((concept_id, text))
        word_count = len(text.split())
        print(f"  ✅  {filename:<25} → concept_id='{concept_id}' ({word_count} words)")

    print(f"\n  Total documents loaded: {len(documents)}")
    return documents


# =============================================================================
# STEP 2 — CHUNK TEXT
# =============================================================================

def chunk_text(text: str, concept_id: str,
               chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int    = CHUNK_OVERLAP_WORDS) -> list:
    """
    Splits a long text into overlapping word-based chunks.

    WHY OVERLAPPING CHUNKS?
      If a sentence spans the boundary between chunk i and chunk i+1,
      without overlap, the complete sentence never appears in any chunk.
      With overlap, the last 50 words of chunk i reappear at the start
      of chunk i+1, ensuring sentences are never cut mid-thought.

    EXAMPLE (with chunk_size=10, overlap=3):
      text = "A B C D E F G H I J K L M N O"
      Chunk 0: "A B C D E F G H I J"          (words 0-9)
      Chunk 1: "H I J K L M N O P Q"          (words 7-16, starts at 10-3=7)
      Chunk 2: "O P Q R S T..."               (words 14-23)

    WHAT GETS RETURNED:
      A list of dicts, each representing one chunk:
      {
        "chunk_id":   "os_chunk_0",   ← unique ID for this chunk
        "concept_id": "os",           ← which concept it belongs to
        "text":       "...",          ← the actual text content
        "word_count": 298,            ← how many words in this chunk
        "chunk_index": 0              ← position in original document
      }

    Args:
        text:       full text of the note file
        concept_id: which concept this text belongs to
        chunk_size: target words per chunk (default 300)
        overlap:    words to repeat at start of each new chunk (default 50)

    Returns:
        list of chunk dicts
    """
    words = text.split()    # split by whitespace into word list
    chunks = []
    chunk_index = 0

    # Slide through words with step = chunk_size - overlap
    # This creates the overlapping window effect
    step = chunk_size - overlap   # 300 - 50 = 250 words per step

    for start in range(0, len(words), step):
        end = start + chunk_size              # end of this chunk
        chunk_words = words[start:end]        # slice the word list

        if not chunk_words:
            break

        chunk_text = " ".join(chunk_words)    # rejoin words into text

        chunks.append({
            "chunk_id":    f"{concept_id}_chunk_{chunk_index}",
            "concept_id":  concept_id,
            "text":        chunk_text,
            "word_count":  len(chunk_words),
            "chunk_index": chunk_index
        })
        chunk_index += 1

        # Stop if we've consumed all words
        if end >= len(words):
            break

    return chunks


def chunk_all_documents(documents: list) -> list:
    """
    Chunks all documents and returns a flat list of all chunks.

    Args:
        documents: list of (concept_id, text) tuples

    Returns:
        list of all chunk dicts across all documents
    """
    all_chunks = []
    print(f"\n✂️   Chunking documents...")

    for concept_id, text in documents:
        chunks = chunk_text(text, concept_id)
        all_chunks.extend(chunks)
        print(f"  {concept_id:<20} → {len(chunks)} chunks")

    print(f"\n  Total chunks created: {len(all_chunks)}")
    return all_chunks


# =============================================================================
# STEP 3 — EMBED CHUNKS
# =============================================================================

def embed_chunks(chunks: list) -> np.ndarray:
    """
    Converts all chunk texts into dense vector embeddings.

    HOW SENTENCE-TRANSFORMERS WORKS:
      1. Tokenizes each text into wordpieces
      2. Runs through BERT-based transformer network
      3. Mean-pools the token embeddings
      4. Returns a 384-dimensional float32 vector per text
      5. Similar texts → close vectors (high cosine similarity)

    BATCH PROCESSING:
      We embed all chunks at once in a single batch call.
      This is much faster than embedding one at a time because:
      - GPU/CPU computation is parallelized across the batch
      - Model overhead (loading) happens only once

    NORMALIZATION:
      We normalize embeddings to unit length (L2 norm = 1).
      This allows using dot product instead of cosine similarity in FAISS,
      which is faster. With normalized vectors: dot product = cosine similarity.

    EXAMPLE:
      "Operating system manages processes"
      → [0.12, -0.34, 0.78, ..., 0.05]  (384 numbers)

    Args:
        chunks: list of chunk dicts with "text" field

    Returns:
        np.ndarray of shape (n_chunks, 384), dtype=float32
    """
    print(f"\n🤖  Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"    (Downloads ~80MB on first run, cached after that)")

    # Load the sentence-transformer model
    # On first run: downloads from HuggingFace Hub to ~/.cache/
    # Subsequent runs: loads from cache instantly
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Extract just the text from each chunk dict
    texts = [chunk["text"] for chunk in chunks]

    print(f"  Embedding {len(texts)} chunks...")

    # encode() returns numpy array of shape (n_texts, 384)
    # show_progress_bar=True prints a progress bar during encoding
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,          # process 32 chunks at a time
        normalize_embeddings=True   # L2 normalize for cosine similarity via dot product
    )

    print(f"\n  Embedding shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Sample vector norm: {np.linalg.norm(embeddings[0]):.4f} (should be ~1.0)")

    return embeddings.astype("float32")   # FAISS requires float32


# =============================================================================
# STEP 4 — BUILD FAISS INDEX
# =============================================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Creates a FAISS index and adds all embeddings to it.

    WHICH INDEX TYPE — IndexFlatIP:
      Flat = brute-force search (checks every vector)
      IP  = Inner Product (= cosine similarity for normalized vectors)
      For our scale (< 10,000 chunks): brute-force is fast enough
      At larger scales, use IndexIVFFlat or IndexHNSW instead.

    HOW FAISS SEARCH WORKS:
      1. Query text gets embedded into a query vector
      2. FAISS computes dot product between query vector and ALL stored vectors
      3. Returns top-k most similar vectors (highest dot product = most similar)
      4. We use the chunk index to look up the original text in metadata.json

    DIMENSION = 384:
      Must match the embedding model's output dimension.
      all-MiniLM-L6-v2 outputs 384-dimensional vectors.

    Args:
        embeddings: float32 array of shape (n_chunks, 384)

    Returns:
        faiss.IndexFlatIP with all embeddings added
    """
    dimension = embeddings.shape[1]   # should be 384
    print(f"\n📊  Building FAISS index (dimension={dimension})...")

    # Create flat inner product index
    index = faiss.IndexFlatIP(dimension)

    # Add all embeddings to the index
    # faiss.IndexFlatIP expects float32 numpy array
    index.add(embeddings)

    print(f"  Vectors in index: {index.ntotal}")
    return index


# =============================================================================
# STEP 5 — SAVE INDEX AND METADATA
# =============================================================================

def save_index_and_metadata(index: faiss.IndexFlatIP, chunks: list):
    """
    Saves the FAISS index and chunk metadata to disk.

    TWO FILES SAVED:

    1. models/faiss_index/index.faiss
       Binary file containing all the embedding vectors.
       FAISS can load this in milliseconds.

    2. models/faiss_index/metadata.json
       JSON array where metadata[i] corresponds to index vector i.
       When FAISS returns "vector at index 3 is most similar",
       we look up metadata[3] to get the actual text.

       Each metadata entry:
       {
         "chunk_id":    "os_chunk_2",
         "concept_id":  "os",
         "text":        "Deadlock occurs when...",
         "word_count":  287,
         "chunk_index": 2
       }

    IMPORTANT: The order of metadata MUST match the order vectors were
    added to FAISS. We maintain this by building from the same chunks list.

    Args:
        index:  built FAISS index
        chunks: list of chunk dicts (same order as embeddings were added)
    """
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)
    print(f"\n💾  FAISS index saved to: {INDEX_PATH}")
    print(f"    File size: {os.path.getsize(INDEX_PATH) / 1024:.1f} KB")

    # Save metadata (strip large fields we don't need at query time)
    metadata = []
    for chunk in chunks:
        metadata.append({
            "chunk_id":    chunk["chunk_id"],
            "concept_id":  chunk["concept_id"],
            "text":        chunk["text"],
            "word_count":  chunk["word_count"],
            "chunk_index": chunk["chunk_index"]
        })

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"💾  Metadata saved to: {METADATA_PATH}")
    print(f"    Total chunks indexed: {len(metadata)}")


# =============================================================================
# VERIFY INDEX (sanity check)
# =============================================================================

def verify_index():
    """
    Quick sanity check: loads the saved index and does a test query.

    This confirms:
      - Index was saved correctly
      - Loading works
      - Search returns sensible results

    Called automatically at end of run_ingestion()
    """
    print(f"\n🔍  Verifying index with test query...")

    # Load index and metadata
    index    = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Test query: "what is deadlock"
    test_query = "what is deadlock in operating systems"
    query_vec  = model.encode([test_query], normalize_embeddings=True).astype("float32")

    # Search for top 3 most similar chunks
    distances, indices = index.search(query_vec, k=3)

    print(f"  Query: '{test_query}'")
    print(f"  Top 3 results:")
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        chunk = metadata[idx]
        preview = chunk["text"][:100].replace("\n", " ")
        print(f"    {rank+1}. [{chunk['concept_id']}] similarity={dist:.4f}")
        print(f"       {preview}...")

    print(f"\n  ✅ Index verified successfully!")


# =============================================================================
# MAIN INGESTION PIPELINE
# =============================================================================

def run_ingestion():
    """
    Full Phase 6 ingestion pipeline. Run this script to build the index.

    STEPS:
      1. Read all .txt files from data/notes/
      2. Chunk each document into ~300-word pieces
      3. Embed all chunks using sentence-transformers
      4. Build FAISS index
      5. Save index + metadata to models/faiss_index/
      6. Verify the index with a test query

    EXPECTED TIME:
      First run:  2-5 minutes (downloads embedding model ~80MB)
      Subsequent: 10-30 seconds (model cached, just re-indexes)
    """
    print("\n" + "=" * 60)
    print("  RETENT ENGRAM — Phase 6 Knowledge Base Ingestion")
    print("=" * 60)

    # Step 1: Read files
    documents = read_note_files()

    # Step 2: Chunk documents
    chunks = chunk_all_documents(documents)

    # Step 3: Embed chunks
    embeddings = embed_chunks(chunks)

    # Step 4: Build FAISS index
    index = build_faiss_index(embeddings)

    # Step 5: Save
    save_index_and_metadata(index, chunks)

    # Step 6: Verify
    verify_index()

    print("\n" + "=" * 60)
    print("  ✅  Ingestion Complete!")
    print(f"  Index: {INDEX_PATH}")
    print(f"  Chunks: {len(chunks)}")
    print("=" * 60)
    print("\n  Next step: Run Part 2 → rag/generate.py")
    print()


# =============================================================================
# LOAD FUNCTIONS (called by generate.py, not by this script)
# =============================================================================

_cached_index    = None
_cached_metadata = None
_cached_model    = None


def load_index_and_model():
    """
    Loads FAISS index, metadata, and embedding model.
    Caches all three in module-level variables after first load.

    CALLED BY: generate.py's retrieve_chunks() function.

    Returns:
        tuple: (faiss_index, metadata_list, sentence_transformer_model)
        Returns (None, None, None) if index files don't exist yet.
    """
    global _cached_index, _cached_metadata, _cached_model

    # Return cached versions if already loaded
    if _cached_index is not None:
        return _cached_index, _cached_metadata, _cached_model

    # Check if index files exist
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return None, None, None

    # Load FAISS index
    _cached_index = faiss.read_index(INDEX_PATH)

    # Load metadata
    with open(METADATA_PATH) as f:
        _cached_metadata = json.load(f)

    # Load embedding model (cached by sentence-transformers after first load)
    _cached_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"✅  FAISS index loaded: {_cached_index.ntotal} vectors")
    return _cached_index, _cached_metadata, _cached_model


def is_index_available() -> bool:
    """Returns True if the FAISS index has been built and saved."""
    return os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH)


# Allow running directly
if __name__ == "__main__":
    run_ingestion()