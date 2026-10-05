"""
backend/config.py
=================
One place for every path and setting.

Why this exists: several pages built paths like  os.path.dirname(__file__)/../models/...
A page that lives one folder deeper than the file the path was copied from silently
looks in the wrong place (this is the most likely reason Settings said "Knowledge base:
not built" while My Notes showed 65 chunks).  Everything now derives from ROOT below.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../Retent-Engram

DATA_DIR      = ROOT / "data"
NOTES_DIR     = DATA_DIR / "notes"                  # hand-written .txt notes (seed notes)
LIBRARY_DIR   = DATA_DIR / "library"                # uploaded PDFs -> sections (JSON)
CONCEPTS_PATH = DATA_DIR / "concepts.json"

MODELS_DIR      = ROOT / "models"
INDEX_DIR       = MODELS_DIR / "faiss_index"
INDEX_PATH      = INDEX_DIR / "index.faiss"
INDEX_META_PATH = INDEX_DIR / "metadata.json"
MODEL_PATH      = MODELS_DIR / "recall_model.pkl"
MODEL_META_PATH = MODELS_DIR / "model_metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---- local LLM ---------------------------------------------------------------
OLLAMA_URL = os.getenv("RETENT_OLLAMA_URL", "http://localhost:11434")

# Fastest-on-a-laptop first.  The first model in this list that is installed wins,
# so nothing breaks if you only have mistral.  Override with RETENT_LLM in .env,
# e.g.  RETENT_LLM=qwen2.5:3b,mistral
LLM_PREFERENCE = [
    m.strip()
    for m in os.getenv("RETENT_LLM", "llama3.2:3b,qwen2.5:3b,phi3.5,gemma2:2b,llama3.2:1b,mistral").split(",")
    if m.strip()
]
