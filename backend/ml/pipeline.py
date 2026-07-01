"""
backend/ml/pipeline.py  (UPDATED for Phase 4)
================================================
WHAT CHANGED FROM PHASE 3
--------------------------
Phase 3 pipeline.py called scorer.py's Ebbinghaus formula directly.

Phase 4 pipeline.py now:
  1. First checks if a trained ML model exists (models/recall_model.pkl)
  2. If YES → uses predict.py to get ML prediction
  3. If NO  → falls back to scorer.py's Ebbinghaus formula

This is called the "graceful degradation" pattern:
  - App always works (formula is always the fallback)
  - App gets better automatically once training is done
  - The model_used field in MongoDB tells you which was used

NOTHING ELSE CHANGES:
  - feature extraction:   still features.py
  - saving to MongoDB:    same upsert logic
  - return value shape:   same list of dicts
  - dashboard call:       still compute_scores_for_user(user_id)

PHASE 5 NOTE:
  Phase 5 adds urgency_score to each result dict.
  That will be added in pipeline.py too, using the recall_score + features.
"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from collections import defaultdict
from datetime import datetime, timezone

from backend.ml.features import extract_features
from backend.ml.scorer import compute_recall_score as formula_score, get_priority
from backend.db import get_collection


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT    = os.path.join(os.path.dirname(__file__), "..", "..")
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "recall_model.pkl")


# =============================================================================
# SMART SCORING: ML model if available, formula if not
# =============================================================================

def compute_recall_smart(features: dict) -> tuple:
    """
    Decides whether to use the ML model or Ebbinghaus formula.

    DECISION LOGIC:
      1. Try to import predict_recall from predict.py
      2. Check if model file exists (is_model_available())
      3. If yes: call predict_recall(features) → get ML score
         - If prediction returns None (error): fall back to formula
      4. If no model: call formula_score(features) from scorer.py

    RETURN VALUE:
      (recall_score, model_used)
      model_used is stored in MongoDB so dashboard can show which model ran

    EXAMPLES:
      Before training: returns (72.3, "ebbinghaus_formula")
      After training:  returns (68.1, "xgboost")

    Args:
        features: dict from extract_features()

    Returns:
        tuple: (recall_score: float, model_used: str)
    """
    # ── Try ML model first ────────────────────────────────────────────────────
    try:
        from backend.ml.predict import predict_recall, is_model_available

        if is_model_available():
            ml_score = predict_recall(features)

            if ml_score is not None:
                # ML model worked — use its prediction
                return ml_score, "xgboost"
            # ml_score is None = prediction failed, fall through to formula
    except ImportError:
        # predict.py not yet created (shouldn't happen after Phase 4)
        pass
    except Exception as e:
        # Any other error (model file corrupted, etc.)
        print(f"⚠️  ML prediction failed: {e}, falling back to formula")

    # ── Fall back to Ebbinghaus formula ───────────────────────────────────────
    recall = formula_score(features)
    return recall, "ebbinghaus_formula"


# =============================================================================
# SAVE TO MONGODB
# =============================================================================

def save_recall_score(user_id: str, concept_id: str,
                      recall_score: float, features: dict,
                      model_used: str = "ebbinghaus_formula"):
    """
    Upserts a recall score document into MongoDB recall_scores collection.

    UPDATED FROM PHASE 3:
      Now includes model_used parameter (previously hardcoded to "formula").
      This lets the dashboard show whether ML or formula was used.

    MONGODB DOCUMENT SHAPE:
      {
        "user_id":       "shivangi_01",
        "concept_id":    "os",
        "recall_score":  68.1,
        "priority":      "Medium",
        "last_computed": datetime(...),
        "model_used":    "xgboost",      ← NEW: was always "ebbinghaus_formula"
        "features":      { ...6 features... }
      }
    """
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


# =============================================================================
# MAIN PIPELINE: Run for one user
# =============================================================================

def compute_scores_for_user(user_id: str) -> list:
    """
    Full scoring pipeline for one student. Called by 2_dashboard.py on load.

    IDENTICAL INTERFACE TO PHASE 3 — callers don't need to change.

    WHAT'S NEW IN PHASE 4:
      - compute_recall_smart() replaces direct formula call
      - model_used is now dynamic ("xgboost" or "ebbinghaus_formula")
      - each result dict includes "model_used" key

    RETURNS:
      [
        {
          "concept_id":   "os",
          "recall_score": 68.1,
          "priority":     "Medium",
          "model_used":   "xgboost",    ← new in Phase 4
          "features":     {...}
        },
        ...
      ]
      Sorted by recall_score ascending (most urgent first).
    """
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

        # CHANGED FROM PHASE 3: use smart scoring instead of direct formula call
        recall_score, model_used = compute_recall_smart(features)

        # Save to MongoDB with model_used
        save_recall_score(user_id, concept_id, recall_score, features, model_used)

        results.append({
            "concept_id":   concept_id,
            "recall_score": recall_score,
            "priority":     get_priority(recall_score),
            "model_used":   model_used,   # ← new in Phase 4
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
        pass  # urgency is optional in pipeline — queue page handles it separately

    results.sort(key=lambda x: x["recall_score"])
    return results


# =============================================================================
# CONVENIENCE: Run for all users
# =============================================================================

def compute_scores_for_all_users() -> dict:
    """
    Runs the full pipeline for every user. Used for demo prep and testing.
    """
    users_col = get_collection("users")
    all_users = list(users_col.find({}, {"user_id": 1, "_id": 0}))

    all_results = {}
    for user in all_users:
        uid = user["user_id"]
        all_results[uid] = compute_scores_for_user(uid)

    return all_results