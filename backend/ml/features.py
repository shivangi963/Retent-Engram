# backend/ml/features.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime, timezone
import math
from backend.db import get_collection

# ── Constants ─────────────────────────────────────────────────────────────────

RECALL_THRESHOLD = 65.0   # below this = needs review


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(events: list) -> dict:
    """
    Takes a list of event dicts for one user-concept pair.
    Returns a dict of 6 computed features.
    """
    if not events:
        return None

    # Sort oldest → newest
    sorted_events = sorted(events, key=lambda e: e["timestamp"])
    now = datetime.now(timezone.utc)

    # 1. Hours since last review
    last_event = sorted_events[-1]
    last_ts = last_event["timestamp"]
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    hours_since_last = (now - last_ts).total_seconds() / 3600

    # 2. Total review count
    total_reviews = len(sorted_events)

    # 3. Average score across all reviews
    scores = [e.get("score", 0) for e in sorted_events]
    avg_score = sum(scores) / len(scores)

    # 4. Last score (most recent event)
    last_score = sorted_events[-1].get("score", 0)

    # 5. Success streak — how many of the last 3 reviews scored above 0.6
    recent = sorted_events[-3:]
    success_streak = sum(1 for e in recent if e.get("score", 0) >= 0.6)

    # 6. Average response time in minutes
    times = [e.get("response_time_min", e.get("response_time_sec", 0)) for e in sorted_events]
    # if stored in seconds (old Phase 0 events), convert
    avg_response_time = sum(times) / len(times)

    return {
        "hours_since_last": round(hours_since_last, 2),
        "total_reviews": total_reviews,
        "avg_score": round(avg_score, 4),
        "last_score": round(last_score, 4),
        "success_streak": success_streak,
        "avg_response_time": round(avg_response_time, 2)
    }


# ── Ebbinghaus decay formula ──────────────────────────────────────────────────

def compute_recall_score(features: dict) -> float:
    """
    Estimates recall % using a weighted Ebbinghaus-inspired formula.

    Core idea:
      - Base retention = last_score × avg_score (how well you know it)
      - Decay = e^(-hours / half_life)
      - half_life grows with more reviews and success streak
      - Result scaled to 0–100

    Returns a float between 0.0 and 100.0
    """
    if features is None:
        return 0.0

    hours = features["hours_since_last"]
    total_reviews = features["total_reviews"]
    avg_score = features["avg_score"]
    last_score = features["last_score"]
    streak = features["success_streak"]

    # Base retention: weighted blend of last score and average
    base_retention = (0.6 * last_score) + (0.4 * avg_score)

    # Half-life in hours: starts at 24h, grows with reviews and streak
    # More reviews + higher streak = slower forgetting
    half_life = 24 * (1 + math.log1p(total_reviews)) * (1 + 0.3 * streak)

    # Ebbinghaus decay: R = base × e^(-hours / half_life)
    decay = math.exp(-hours / half_life)
    recall = base_retention * decay * 100

    return round(min(max(recall, 0.0), 100.0), 2)


# ── Priority label ────────────────────────────────────────────────────────────

def get_priority(recall_score: float) -> str:
    if recall_score < 40:
        return "High"
    elif recall_score < RECALL_THRESHOLD:
        return "Medium"
    else:
        return "Low"


# ── Save recall score to MongoDB ──────────────────────────────────────────────

def save_recall_score(user_id: str, concept_id: str, recall_score: float,
                      features: dict, model_used: str = "formula"):
    """
    Upserts a recall score document into the recall_scores collection.
    Creates it if it doesn't exist, updates it if it does.
    """
    col = get_collection("recall_scores")
    doc = {
        "user_id": user_id,
        "concept_id": concept_id,
        "recall_score": recall_score,
        "priority": get_priority(recall_score),
        "last_computed": datetime.now(timezone.utc),
        "model_used": model_used,
        "features": features      # store raw features too, useful for Phase 4 training
    }
    col.update_one(
        {"user_id": user_id, "concept_id": concept_id},
        {"$set": doc},
        upsert=True
    )


# ── Main pipeline: run for one user ──────────────────────────────────────────

def compute_scores_for_user(user_id: str):
    """
    Fetches all events for a user, groups by concept,
    computes features + recall score for each, saves to recall_scores.
    Returns a list of result dicts for display.
    """
    events_col = get_collection("events")
    all_events = list(events_col.find({"user_id": user_id}, {"_id": 0}))

    if not all_events:
        return []

    # Group events by concept_id
    from collections import defaultdict
    concept_events = defaultdict(list)
    for e in all_events:
        concept_events[e["concept_id"]].append(e)

    results = []
    for concept_id, events in concept_events.items():
        features = extract_features(events)
        if features is None:
            continue
        recall_score = compute_recall_score(features)
        save_recall_score(user_id, concept_id, recall_score, features)
        results.append({
            "concept_id": concept_id,
            "recall_score": recall_score,
            "priority": get_priority(recall_score),
            "features": features
        })

    # Sort by urgency: lowest recall first
    results.sort(key=lambda x: x["recall_score"])
    return results


# ── Convenience: run for all users ───────────────────────────────────────────

def compute_scores_for_all_users():
    """
    Runs compute_scores_for_user for every user in the users collection.
    Called on dashboard load in Phase 3.
    """
    users_col = get_collection("users")
    all_users = list(users_col.find({}, {"user_id": 1, "_id": 0}))
    for user in all_users:
        compute_scores_for_user(user["user_id"])