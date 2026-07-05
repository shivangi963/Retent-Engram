import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from collections import defaultdict
from datetime import datetime, timezone

from backend.ml.features import extract_features
from backend.ml.scorer import compute_recall_score as formula_score, get_priority
from backend.db import get_collection


PROJECT_ROOT    = os.path.join(os.path.dirname(__file__), "..", "..")
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "recall_model.pkl")


def compute_recall_smart(features: dict) -> tuple:
    # decides to use the ML model or Ebbinghaus formula.
    try:
        from backend.ml.predict import predict_recall, is_model_available

        if is_model_available():
            ml_score = predict_recall(features)

            if ml_score is not None:
                return ml_score, "xgboost"
    except ImportError:
        pass
    except Exception as e:
        print(f"ML prediction failed: {e}, falling back to formula")

    # fall back to Ebbinghaus formula 
    recall = formula_score(features)
    return recall, "ebbinghaus_formula"


def save_recall_score(user_id: str, concept_id: str, recall_score: float, features: dict, model_used: str = "ebbinghaus_formula"):
    
    # Upserts a recall score document into MongoDB recall_scores collection.

    col = get_collection("recall_scores")

    doc = {
        "user_id":       user_id,
        "concept_id":    concept_id,
        "recall_score":  recall_score,
        "priority":      get_priority(recall_score),
        "last_computed": datetime.now(timezone.utc),
        "model_used":    model_used,
        "features":      features
    }

    col.update_one(
        filter={"user_id": user_id, "concept_id": concept_id},
        update={"$set": doc},
        upsert=True
    )


def compute_scores_for_user(user_id: str) -> list:
   # Full scoring pipeline for 1 student. Called by 2_dashboard.py on load.

    events_col = get_collection("events")

    all_events = list(
        events_col.find({"user_id": user_id}, {"_id": 0})
    )

    if not all_events:
        return []

    # Group by concept
    concept_events = defaultdict(list)
    for event in all_events:
        concept_id = event.get("concept_id", "unknown")
        concept_events[concept_id].append(event)

    results = []

    for concept_id, events in concept_events.items():
        features = extract_features(events)
        if features is None:
            continue

        recall_score, model_used = compute_recall_smart(features)

        save_recall_score(user_id, concept_id, recall_score, features, model_used)

        results.append({
            "concept_id":   concept_id,
            "recall_score": recall_score,
            "priority":     get_priority(recall_score),
            "model_used":   model_used,   
            "features":     features
        })

    concepts_path = os.path.join(PROJECT_ROOT, "data", "concepts.json")
    try:
        with open(concepts_path) as f:
            concepts_list = json.load(f)
        concepts_lookup = {c["concept_id"]: c for c in concepts_list}

        from backend.scheduler import compute_urgency_score, get_urgency_level
        for r in results:
            cid        = r["concept_id"]
            hours      = r["features"].get("hours_since_last", 24.0)
            difficulty = concepts_lookup.get(cid, {}).get("difficulty", 3)
            urgency    = compute_urgency_score(r["recall_score"], hours, difficulty)
            r["urgency_score"] = urgency
            r["urgency_level"] = get_urgency_level(urgency)
    except Exception:
        pass  

    results.sort(key=lambda x: x["recall_score"])
    return results


def compute_scores_for_all_users() -> dict:
    
    #Runs the full pipeline for every user. Used for demo prep and testing.
    users_col = get_collection("users")
    all_users = list(users_col.find({}, {"user_id": 1, "_id": 0}))

    all_results = {}
    for user in all_users:
        uid = user["user_id"]
        all_results[uid] = compute_scores_for_user(uid)

    return all_results