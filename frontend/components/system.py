"""
frontend/components/system.py
==============================
Environment / service health checks (MongoDB, Ollama, FAISS index, ML model).
Ported from the old main.py so both the Dashboard (compact pill) and the
Settings page (full detail) can use the same checks.
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def check_mongodb() -> tuple:
    try:
        from backend.db import db
        db.command("ping")
        return True, "Connected"
    except Exception as e:
        return False, f"Cannot connect: {str(e)[:60]}"


def check_ollama() -> tuple:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            names = [m.get("name", "") for m in r.json().get("models", [])]
            if any("mistral" in n for n in names):
                return True, "Running · Mistral loaded"
            return True, "Running · run `ollama pull mistral`"
        return False, "Running but returned an error"
    except Exception:
        return False, "Not running · run `ollama serve`"


def check_faiss_index() -> tuple:
    path = os.path.join(ROOT, "models", "faiss_index", "index.faiss")
    if os.path.exists(path):
        return True, f"Built ({os.path.getsize(path) / 1024:.0f} KB)"
    return False, "Not built · run `python rag/ingest.py`"


def check_ml_model() -> tuple:
    model_path = os.path.join(ROOT, "models", "recall_model.pkl")
    meta_path = os.path.join(ROOT, "models", "model_metadata.json")
    if os.path.exists(model_path):
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            name = meta.get("active_model", "unknown")
            auc = meta.get(name, {}).get("auc")
            if auc:
                return True, f"{name} · AUC {auc:.3f}"
        return True, "Trained"
    return False, "Not trained · using formula fallback"


def all_checks() -> list:
    return [
        ("MongoDB", "database", *check_mongodb()),
        ("Ollama (Mistral)", "cpu", *check_ollama()),
        ("Knowledge base", "layers", *check_faiss_index()),
        ("ML model", "target", *check_ml_model()),
    ]
