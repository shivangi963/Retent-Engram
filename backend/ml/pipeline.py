import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from collections import defaultdict
from datetime import datetime, timezone

from backend.ml.features import extract_features
from backend.ml.scorer import compute_recall_score, get_priority
from backend.db import get_collection


def save_recall_score(user_id: str, concept_id: str, recall_score: float, features: dict):
    col = get_collection("recall_scores")

    doc = {
        "user_id":       user_id,
        "concept_id":    concept_id,
        "recall_score":  recall_score,
        "priority":      get_priority(recall_score),
        "last_computed": datetime.now(timezone.utc),
        "model_used":    "ebbinghaus_formula",   # Phase 4 changes this to "xgboost"
        "features":      features                 # stored as training data for Phase 4
    }

    # update_one with upsert=True:
    #   filter  = find the doc where user_id AND concept_id match
    #   $set    = replace ALL fields with our new doc
    #   upsert  = create if not found
    col.update_one(
        filter={"user_id": user_id, "concept_id": concept_id},
        update={"$set": doc},
        upsert=True
    )

def compute_scores_for_user(user_id: str) -> list:
    events_col = get_collection("events")

    # Fetch all events for this user
    # The {"_id": 0} projection tells MongoDB: don't include the _id field
    all_events = list(
        events_col.find({"user_id": user_id}, {"_id": 0})
    )

    # If no events at all, return empty list
    if not all_events:
        return []

    # GROUP events by concept_id using defaultdict
    # defaultdict(list) automatically creates an empty list for new keys
    # After this loop, concept_events looks like:
    #   { "os": [event1, event2], "dbms": [event3], "cn": [event4, event5, event6] }
    concept_events = defaultdict(list)
    for event in all_events:
        concept_id = event.get("concept_id", "unknown")
        concept_events[concept_id].append(event)

    results = []

    # Process each concept independently
    for concept_id, events in concept_events.items():

        # STEP A: Extract features from raw events
        features = extract_features(events)

        # Skip if no features returned (shouldn't happen but defensive programming)
        if features is None:
            continue

        # STEP B: Compute recall score from features
        recall_score = compute_recall_score(features)

        # STEP C: Save to MongoDB
        save_recall_score(user_id, concept_id, recall_score, features)

        # STEP D: Add to results list
        results.append({
            "concept_id":   concept_id,
            "recall_score": recall_score,
            "priority":     get_priority(recall_score),
            "features":     features
        })

    # Sort by recall_score ascending → lowest (most urgent) first
    results.sort(key=lambda x: x["recall_score"])

    return results


def compute_scores_for_all_users() -> dict:
    users_col = get_collection("users")

    # Only fetch the user_id field from each user document
    all_users = list(users_col.find({}, {"user_id": 1, "_id": 0}))

    all_results = {}
    for user in all_users:
        uid = user["user_id"]
        all_results[uid] = compute_scores_for_user(uid)

    return all_results