"""
backend/ml/pipeline.py
======================
events -> features -> recall % -> priority + urgency -> saved to recall_scores.

Same function names and result keys as before (so the redesigned pages keep working);
new keys: half_life_days, due_in_hours.
"""
import json
from collections import defaultdict
from datetime import datetime, timezone

from backend.config import CONCEPTS_PATH
from backend.db import get_collection
from backend.ml import scorer
from backend.ml.features import extract_features
from backend.ml.scorer import get_priority


def _difficulty_map() -> dict:
    try:
        with open(CONCEPTS_PATH, encoding="utf-8") as f:
            return {c["concept_id"]: c.get("difficulty", 3) for c in json.load(f)}
    except Exception:
        return {}


def compute_recall_smart(features: dict):
    """(recall %, model name).  Trained model if valid, memory model otherwise."""
    try:
        from backend.ml.predict import predict_recall, active_model_name
        ml = predict_recall(features)
        if ml is not None:
            return ml, active_model_name()
    except Exception as exc:
        print(f"  trained model unavailable ({exc}); using memory model")
    return scorer.compute_recall_score(features), "memory_model"


def save_recall_score(user_id, concept_id, recall_score, features,
                      model_used="memory_model", extra=None):
    doc = {
        "user_id": user_id, "concept_id": concept_id,
        "recall_score": recall_score, "priority": get_priority(recall_score),
        "last_computed": datetime.now(timezone.utc),
        "model_used": model_used, "features": features,
    }
    doc.update(extra or {})
    get_collection("recall_scores").update_one(
        {"user_id": user_id, "concept_id": concept_id}, {"$set": doc}, upsert=True)


def compute_scores_for_user(user_id: str) -> list:
    events = list(get_collection("events").find({"user_id": user_id}, {"_id": 0}))
    if not events:
        return []

    by_concept = defaultdict(list)
    for e in events:
        by_concept[e.get("concept_id", "unknown")].append(e)

    difficulty = _difficulty_map()
    try:
        from backend.scheduler import compute_urgency_score, get_urgency_level
    except Exception:
        compute_urgency_score = get_urgency_level = None

    results = []
    for concept_id, evs in by_concept.items():
        features = extract_features(evs)
        if features is None:
            continue
        recall, model_used = compute_recall_smart(features)
        info = scorer.explain(features)
        item = {
            "concept_id": concept_id, "recall_score": recall,
            "priority": get_priority(recall), "model_used": model_used,
            "features": features,
            "half_life_days": info["half_life_days"], "due_in_hours": info["due_in_hours"],
        }
        if compute_urgency_score:
            urg = compute_urgency_score(recall, features["hours_since_last"], difficulty.get(concept_id, 3))
            item["urgency_score"], item["urgency_level"] = urg, get_urgency_level(urg)
        save_recall_score(
            user_id, concept_id, recall, features, model_used,
            extra={k: item[k] for k in ("half_life_days", "due_in_hours", "urgency_score", "urgency_level") if k in item})
        results.append(item)

    results.sort(key=lambda r: r["recall_score"])
    return results


def compute_scores_for_all_users() -> dict:
    users = get_collection("users").find({}, {"user_id": 1, "_id": 0})
    return {u["user_id"]: compute_scores_for_user(u["user_id"]) for u in users}
