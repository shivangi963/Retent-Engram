"""
backend/autogen.py
==================
AUTOMATIC TRIGGERING: scheduler -> RAG, without waiting for the student to press Generate.

Whenever recall scores are computed (a page loads, a study event is saved, or the API's background
worker wakes up) this decides what the student will need next and prepares it in the background:

    recall < 65%  ->  High priority   : flashcards + quiz
                      Medium priority : summary
    skipped when: the concept has no notes, it is snoozed, or fresh content for the same difficulty
                  already exists (so nothing is generated twice)

The material is saved in `generated_content` (auto_generated = true) exactly like content the student
asked for, so the Study page shows it instantly from the cache.

Safe by design: one job per student at a time, at most MAX_ITEMS pieces per run, at most one run per
MIN_INTERVAL_S per student, nothing happens if Ollama is not running.  Turn it off with RETENT_AUTOGEN=0.
"""
import os
import threading
import time
from datetime import datetime, timedelta, timezone

from backend.db import get_collection

REVIEW_THRESHOLD = 65.0
PLAN = {"High": ["flashcard", "quiz"], "Medium": ["summary"], "Low": []}
MAX_ITEMS = 2
MIN_INTERVAL_S = 1800

_lock = threading.Lock()
_running = set()
_last_run = {}


def enabled() -> bool:
    return os.getenv("RETENT_AUTOGEN", "1") != "0"


def _snoozed(user_id: str, concept_id: str) -> bool:
    try:
        doc = get_collection("recall_scores").find_one({"user_id": user_id, "concept_id": concept_id}, {"snoozed_until": 1, "_id": 0})
        until = (doc or {}).get("snoozed_until")
        if until is None:
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        return until > datetime.now(timezone.utc)
    except Exception:
        return False


def plan(user_id: str, results: list, max_items: int = MAX_ITEMS) -> list:
    """[(concept_id, concept_name, content_type, recall)] that still need preparing, most urgent first."""
    from rag import generate as gen
    names = {cid: c.get("name", cid) for cid, c in gen._concepts().items()}
    todo = []
    for r in sorted(results, key=lambda r: r.get("urgency_score", 100 - r["recall_score"]), reverse=True):
        if r["recall_score"] >= REVIEW_THRESHOLD or _snoozed(user_id, r["concept_id"]):
            continue
        cid, name = r["concept_id"], names.get(r["concept_id"], r["concept_id"])
        level = gen.get_difficulty_level(r["recall_score"])
        for ctype in PLAN.get(r["priority"], []):
            cached = gen.get_cached_content(user_id, cid, ctype)
            if cached and cached.get("difficulty_level") == level:
                continue
            if not gen.retrieve_chunks(cid, name):             # no notes -> nothing to base practice on
                break
            todo.append((cid, name, ctype, r["recall_score"]))
            if len(todo) >= max_items:
                return todo
    return todo


def run_autogen(user_id: str, results=None, max_items: int = MAX_ITEMS) -> dict:
    """Prepare practice for the weakest concepts now (blocking).  Returns what happened."""
    from rag import generate as gen
    out = {"user_id": user_id, "created": [], "skipped": None}
    if not (gen.is_ollama_running() and gen.pick_model()):
        out["skipped"] = "Ollama is not running"
        return out
    if results is None:
        from backend.ml.pipeline import compute_scores_for_user
        results = compute_scores_for_user(user_id, trigger=False)
    for cid, name, ctype, recall in plan(user_id, results, max_items):
        res = gen.generate_content(user_id, cid, name, ctype, recall, force_regenerate=True, auto=True)
        out["created"].append({"concept_id": cid, "content_type": ctype, "ok": bool(res.get("content")) and not res.get("error"),
                               "error": res.get("error")})
    return out


def maybe_trigger(user_id: str, results=None, background: bool = True, force: bool = False) -> bool:
    """Start auto-generation if it is enabled, not already running and not run recently."""
    if not enabled():
        return False
    now = time.time()
    with _lock:
        if user_id in _running or (not force and now - _last_run.get(user_id, 0) < MIN_INTERVAL_S):
            return False
        _running.add(user_id)
        _last_run[user_id] = now

    def job():
        try:
            run_autogen(user_id, results)
        except Exception as exc:
            print(f"  auto-generation failed: {exc}")
        finally:
            with _lock:
                _running.discard(user_id)

    if background:
        threading.Thread(target=job, daemon=True, name=f"autogen-{user_id}").start()
    else:
        job()
    return True


def ready_items(user_id: str, hours: int = 24) -> list:
    """Practice the app prepared on its own in the last `hours`: [{'concept_id','content_type','generated_at'}]."""
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        rows = get_collection("generated_content").find(
            {"user_id": user_id, "auto_generated": True, "generated_at": {"$gte": cutoff}},
            {"concept_id": 1, "content_type": 1, "generated_at": 1, "_id": 0})
    except Exception:
        return []
    newest = {}
    for r in sorted(rows, key=lambda r: r["generated_at"]):
        newest[(r["concept_id"], r["content_type"])] = r
    return list(newest.values())


def run_for_all_users() -> list:
    """One pass over every student (used by the API's background worker)."""
    out = []
    try:
        users = [u["user_id"] for u in get_collection("users").find({}, {"user_id": 1, "_id": 0})]
    except Exception:
        users = []
    for uid in users:
        if maybe_trigger(uid, background=False, force=True):
            out.append(uid)
    return out
