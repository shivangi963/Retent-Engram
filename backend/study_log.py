"""
backend/study_log.py
====================
ONE place where study becomes data.

  * In-app study (quizzes, flashcards, coding tasks, reading in My Notes, summaries)
    calls log_study(...) by itself - the student never types a score for these.
  * Study done OUTSIDE the app (learning a new concept, re-reading an old one) is logged
    from the "Log study" page with source="manual".

Event fields added on top of the old schema (old events stay valid):
    source    "manual" | "notes" | "notes_quiz" | "notes_flashcards" | "study_quiz" |
              "study_flashcards" | "study_coding" | "study_summary" | "queue_review"
    activity  manual entries only: "learned" (first time) or "revised" (re-read / revisited)
    note      optional free text (manual entries)
    doc_id, section_id   when the study was a section of an uploaded PDF
"""
from datetime import datetime, timezone

from backend.db import get_collection
from backend.ml.features import _to_dt

MANUAL = "manual"

SOURCE_LABELS = {
    "notes": "Read in My Notes",
    "notes_quiz": "Quiz - My Notes",
    "notes_flashcards": "Flashcards - My Notes",
    "flashcards": "Flashcards - My Notes",
    "study_quiz": "Quiz - Study",
    "study_flashcards": "Flashcards - Study",
    "study_coding": "Coding task - Study",
    "study_summary": "Summary - Study",
    "queue_review": "Review - Today",
}
ACTIVITY_LABELS = {"learned": "Learned something new", "revised": "Revised something I knew"}


# ------------------------------------------------------------------------ writing
def log_study(user_id: str, concept_id: str, score: float, event_type: str = "reading",
              minutes: float = 5, source: str = "notes", doc_id=None, section_id=None,
              timestamp=None, activity=None, note=None):
    """Save one study event, refresh recall scores and the streak.

    Returns False if the event could not be saved, otherwise a dict (truthy):
        {"ok": True, "recall": 93.1, "priority": "Low", "due_in_hours": 31.0, "half_life_days": 3.0}
    (the recall keys are missing if the refresh itself failed).  `timestamp` must be timezone-aware
    (any zone) or None for "now"."""
    now = datetime.now(timezone.utc)
    when = _to_dt(timestamp) if timestamp is not None else now
    try:
        event = {
            "user_id": user_id, "concept_id": concept_id, "event_type": event_type,
            "score": round(min(1.0, max(0.0, float(score))), 2),
            "response_time_min": round(float(minutes), 1), "hints_used": 0,
            "timestamp": when, "source": source,
        }
        for key, val in (("doc_id", doc_id), ("section_id", section_id), ("activity", activity), ("note", note)):
            if val:
                event[key] = val
        get_collection("events").insert_one(event)
    except Exception as exc:
        print(f"  could not save study event: {exc}")
        return False

    out = {"ok": True}
    try:                                                    # refreshes must never undo a successful save
        from backend.ml.pipeline import compute_scores_for_user
        for r in compute_scores_for_user(user_id):
            if r["concept_id"] == concept_id:
                out.update(recall=r["recall_score"], priority=r["priority"],
                           due_in_hours=r.get("due_in_hours"), half_life_days=r.get("half_life_days"))
    except Exception as exc:
        print(f"  recall refresh skipped: {exc}")
    if (now - when).total_seconds() < 12 * 3600:            # a log for last week must not extend today's streak
        try:
            from backend.scheduler import update_streak
            update_streak(user_id)
        except Exception as exc:
            print(f"  streak update skipped: {exc}")
    return out


def make_logger(user_id: str, concept_id: str, doc_id=None, section_id=None):
    """log(score, event_type, source, minutes) bound to one student and concept - what the
    practice widgets call when the student finishes something."""
    def log(score, event_type="quiz", source="study_quiz", minutes=5):
        return log_study(user_id, concept_id, score, event_type, minutes=minutes, source=source,
                         doc_id=doc_id, section_id=section_id)
    return log


# ------------------------------------------------------------------------ reading
def get_studied_sections(user_id: str, doc_id: str) -> set:
    try:
        rows = get_collection("events").find({"user_id": user_id, "doc_id": doc_id}, {"section_id": 1, "_id": 0})
        return {e["section_id"] for e in rows if e.get("section_id")}
    except Exception:
        return set()


def count_concept_events(user_id: str, concept_id: str) -> int:
    try:
        return get_collection("events").count_documents({"user_id": user_id, "concept_id": concept_id})
    except Exception:
        return 0


def recent_events(user_id: str, limit: int = 8) -> list:
    """Newest first, WITH _id (needed to remove a wrong entry)."""
    try:
        rows = list(get_collection("events").find({"user_id": user_id}).sort("timestamp", -1).limit(60))
    except Exception:
        return []
    rows = [r for r in rows if _to_dt(r.get("timestamp")) is not None]
    rows.sort(key=lambda r: _to_dt(r["timestamp"]), reverse=True)
    return rows[:limit]


def delete_event(user_id: str, event_id) -> bool:
    """Remove one event (only the owner's), then recompute that concept's recall."""
    try:
        col = get_collection("events")
        doc = col.find_one({"_id": event_id, "user_id": user_id})
        if not doc:
            return False
        col.delete_one({"_id": event_id, "user_id": user_id})
        if col.count_documents({"user_id": user_id, "concept_id": doc["concept_id"]}) == 0:
            get_collection("recall_scores").delete_one({"user_id": user_id, "concept_id": doc["concept_id"]})
        else:
            from backend.ml.pipeline import compute_scores_for_user
            compute_scores_for_user(user_id)
        return True
    except Exception as exc:
        print(f"  could not remove event: {exc}")
        return False


# ---------------------------------------------------------------------- describing
def describe_event(e: dict) -> tuple:
    """(label, logged_by_hand)"""
    src = e.get("source")
    if src == MANUAL:
        return ACTIVITY_LABELS.get(e.get("activity"), "Studied outside the app"), True
    if src in SOURCE_LABELS:
        return SOURCE_LABELS[src], False
    # events from before `source` existed were all typed in by hand
    return {"reading": "Reading", "quiz": "Quiz", "coding": "Coding", "review": "Review"}.get(
        e.get("event_type"), "Study") + " (logged by hand)", True


def to_local(ts) -> datetime:
    return _to_dt(ts).astimezone()


def describe_due(hours) -> str:
    if hours is None:
        return ""
    if hours <= 0.5:
        return "now"
    if hours < 24:
        n = max(1, round(hours))
        return f"in about {n} hour{'s' if n != 1 else ''}"
    days = hours / 24
    if days < 14:
        n = max(1, round(days))
        return f"in about {n} day{'s' if n != 1 else ''}"
    return f"in about {round(days / 7)} weeks"
