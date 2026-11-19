"""
backend/service.py
==================
The application service layer: every operation the API exposes, as plain Python functions that
return JSON-safe data.  backend/api.py (FastAPI) is only a thin routing layer over these, so the
logic can be tested without a web server.

    health()            status of MongoDB, Ollama, the knowledge base and the recall model
    log_event()         save a study event, refresh recall, update streak          (logging)
    recall()            recall % + priority + urgency for every concept            (retention model)
    queue()             today's review queue, goal, streak                         (review scheduler)
    search()            nearest note passages                                      (retrieval)
    stream_llm()        stream text from the local LLM                             (generation)
    generate()          full RAG: retrieve -> prompt -> LLM -> save                (RAG pipeline)
    autogen_*()         automatic practice preparation                             (scheduler -> RAG)
    evaluation()        measured model comparison (Table 5.3)
"""
import json
from datetime import datetime, date

from backend import autogen
from backend.config import CONCEPTS_PATH, EVALUATION_PATH
from backend.ml import pipeline
from backend.ml.predict import active_model_name, get_model_metadata
from backend.remote import ERROR_MARK
from backend.study_log import log_study_local
from rag import generate as gen
from rag import store
from bson import ObjectId


def to_jsonable(obj):
    """Convert MongoDB and Python values into JSON-safe data."""

    if isinstance(obj, ObjectId):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            return str(obj)

    return obj



def _concepts() -> dict:
    try:
        return {c["concept_id"]: c for c in json.loads(CONCEPTS_PATH.read_text(encoding="utf-8"))}
    except Exception:
        return {}



def health() -> dict:
    mongo = False

    try:
        from backend.db import get_collection
        get_collection("users").find_one({}, {"_id": 1})
        mongo = True
    except Exception:
        mongo = False

    ok_kb, kb_msg = store.status_text()

    return {
        "status": "ok",
        "mongodb": mongo,
        "knowledge_base": ok_kb,
        "knowledge_base_detail": kb_msg,
        "recall_model": active_model_name(),
        "autogen": autogen.enabled(),
        "ollama": None,
        "llm": None
    }




def log_event(p: dict) -> dict:
    res = log_study_local(
        p["user_id"], p["concept_id"], p["score"], p.get("event_type") or "reading", minutes=p.get("minutes") or 5,
        source=p.get("source") or "api", doc_id=p.get("doc_id"), section_id=p.get("section_id"),
        timestamp=p.get("timestamp"), activity=p.get("activity"), note=p.get("note"))
    return to_jsonable(res) if res else {"ok": False}


def recall(user_id: str) -> dict:
    return {"user_id": user_id, "results": to_jsonable(pipeline.compute_scores_local(user_id))}


def queue(user_id: str) -> dict:
    from backend.scheduler import build_and_get_queue
    pipeline.compute_scores_local(user_id)
    return to_jsonable(build_and_get_queue(user_id, _concepts()))


def search(q: str, k: int = 4, concept_ids=None, doc_id=None) -> dict:
    return {"hits": to_jsonable(store.search_local(q, k=k, concept_ids=concept_ids or None, doc_id=doc_id or None))}


def stream_llm(prompt: str, kind: str = "answer"):
    """Text chunks; an error becomes ERROR_MARK + message at the end of the stream."""
    try:
        for tok in gen.stream_ollama_local(prompt, kind):
            yield tok
    except RuntimeError as exc:
        yield "\n" + ERROR_MARK + str(exc)


def generate(user_id: str, concept_id: str, content_type: str = "flashcard", force: bool = False) -> dict:
    name = _concepts().get(concept_id, {}).get("name", concept_id)
    scores = {r["concept_id"]: r["recall_score"] for r in pipeline.compute_scores_local(user_id, trigger=False)}
    return to_jsonable(gen.generate_content(user_id, concept_id, name, content_type, scores.get(concept_id, 50.0), force))


def autogen_run(user_id: str) -> dict:
    return to_jsonable(autogen.run_autogen(user_id))


def autogen_ready(user_id: str) -> dict:
    return {"user_id": user_id, "ready": to_jsonable(autogen.ready_items(user_id))}


def evaluation() -> dict:
    try:
        return json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "No evaluation yet. Run: python scripts/run_training.py"}


def models() -> dict:
    return to_jsonable(get_model_metadata())
