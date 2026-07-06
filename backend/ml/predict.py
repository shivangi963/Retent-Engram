"""
backend/ml/predict.py
=======================
PHASE 4 — ML Model Inference

PURPOSE
-------
This file loads the trained model (saved by train.py) and uses it
to predict recall probability for a given set of features.

It is called by pipeline.py, which replaced the Ebbinghaus formula call
with this ML-based prediction.

GRACEFUL DEGRADATION
--------------------
If the model file doesn't exist (e.g. training hasn't been run yet),
predict.py returns None. pipeline.py detects this and falls back to
the Ebbinghaus formula from scorer.py. This means:

  - Before training: app uses formula (Phase 2 behavior)
  - After training:  app uses ML model (Phase 4 behavior)

The user doesn't need to do anything — it just gets better automatically.

MODEL CACHING
-------------
Loading a pickle file on every prediction is slow.
We cache the loaded model in a module-level variable (_cached_model).
After the first load, subsequent calls to predict_recall() return
the prediction without re-reading the file from disk.

HOW PREDICTION WORKS
---------------------
  1. Load model from models/recall_model.pkl (once, cached)
  2. Build a feature vector [f1, f2, f3, f4, f5, f6]
     IN THE SAME ORDER as train.py's FEATURE_COLUMNS
  3. model.predict_proba([[f1,...,f6]]) → [[P(forgot), P(recalled)]]
  4. Take P(recalled) = column index 1
  5. Multiply by 100 → recall score 0–100

IMPORTANT: COLUMN ORDER MUST MATCH
------------------------------------
train.py uses FEATURE_COLUMNS = [
  "hours_since_last", "total_reviews", "avg_score",
  "last_score", "success_streak", "avg_response_time"
]

predict.py builds the feature vector in the SAME order.
If you add a new feature in Phase 5/6, add it to BOTH files
in the same position, then retrain.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import pickle
import numpy as np
from datetime import datetime


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT    = os.path.join(os.path.dirname(__file__), "..", "..")
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "recall_model.pkl")
METADATA_PATH   = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

# FEATURE ORDER — must exactly match train.py's FEATURE_COLUMNS
FEATURE_ORDER = [
    "hours_since_last",
    "total_reviews",
    "avg_score",
    "last_score",
    "success_streak",
    "avg_response_time"
]


# =============================================================================
# MODEL CACHE
# =============================================================================

# Module-level cache: starts as None, gets set on first load
# This is a simple, effective caching pattern for single-process apps
_cached_model = None
_model_loaded_at = None   # timestamp of when we loaded (for debugging)


# =============================================================================
# CHECK IF MODEL EXISTS
# =============================================================================

def is_model_available() -> bool:
    """
    Returns True if the trained model file exists on disk.

    Called by pipeline.py before deciding whether to use ML or formula.

    Returns:
        bool: True if models/recall_model.pkl exists
    """
    return os.path.exists(BEST_MODEL_PATH)


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model():
    """
    Loads the trained model from disk. Caches it after first load.

    CACHING EXPLAINED:
      global _cached_model — we modify the module-level variable
      First call:      _cached_model is None → load from disk → cache it
      Subsequent calls: _cached_model is not None → return cached model

    WHY CACHE?
      Loading a .pkl file involves reading disk + deserializing bytes.
      For XGBoost this can take 50–200ms per call.
      Caching brings it to < 1ms after the first load.
      In Streamlit, this function is called on every page load,
      so caching makes a real difference in dashboard responsiveness.

    Returns:
        The loaded sklearn Pipeline or XGBoost model
        None if file doesn't exist

    Raises:
        FileNotFoundError: if model file is missing (caller should handle)
    """
    global _cached_model, _model_loaded_at

    # Return cached model if already loaded
    if _cached_model is not None:
        return _cached_model

    # Load from disk
    if not is_model_available():
        return None   # model not trained yet

    try:
        with open(BEST_MODEL_PATH, "rb") as f:
            _cached_model = pickle.load(f)
        _model_loaded_at = datetime.now().isoformat()
        print(f"✅  ML model loaded from {BEST_MODEL_PATH}")
        return _cached_model

    except Exception as e:
        print(f"⚠️  Failed to load model: {e}")
        _cached_model = None
        return None


def reload_model():
    """
    Forces a fresh reload of the model from disk (clears cache).

    WHEN TO CALL THIS:
      - After running train.py (new model was saved)
      - If you suspect the cached model is stale

    Called automatically if pipeline.py detects a new model file
    newer than the cached one.
    """
    global _cached_model, _model_loaded_at
    _cached_model = None
    _model_loaded_at = None
    return load_model()


# =============================================================================
# BUILD FEATURE VECTOR
# =============================================================================

def features_to_vector(features: dict) -> np.ndarray:
    """
    Converts a features dict from features.py into a numpy array
    in the exact order the model expects.

    CRITICAL: Order must match FEATURE_ORDER above AND train.py's FEATURE_COLUMNS.

    EXAMPLE:
      features = {
        "hours_since_last": 18.5,
        "total_reviews": 3,
        "avg_score": 0.70,
        "last_score": 0.80,
        "success_streak": 2,
        "avg_response_time": 20.0
      }

      vector = [18.5, 3, 0.70, 0.80, 2, 20.0]
      shape: (1, 6) after reshape — model expects 2D array

    Args:
        features: dict from backend/ml/features.py extract_features()

    Returns:
        np.ndarray of shape (1, 6) — 2D array required by sklearn/xgboost
    """
    vector = [features[col] for col in FEATURE_ORDER]
    return np.array(vector).reshape(1, -1)   # reshape to [[f1, f2, ..., f6]]


# =============================================================================
# PREDICT RECALL PROBABILITY
# =============================================================================

def predict_recall(features: dict) -> float | None:
    """
    Uses the trained ML model to predict recall probability.

    RETURN VALUE INTERPRETATION:
      0.0  → model is certain the student has completely forgotten
      50.0 → model is uncertain (50/50)
      100.0 → model is certain the student perfectly remembers

    NOTE: XGBoost's predict_proba returns the probability of CLASS 1
    (where 1 = recalled, 0 = forgot, as defined in data_prep.py).
    We multiply by 100 to get a percentage.

    RETURNS None (not 0.0) if model is unavailable.
    This lets pipeline.py distinguish between:
      - "model predicted 0% recall" (genuine low score)
      - "model wasn't available" (should fall back to formula)

    Args:
        features: dict from extract_features() in features.py
                  Must have all 6 keys in FEATURE_ORDER

    Returns:
        float: recall score 0.0–100.0
        None: if model not available or prediction failed
    """
    model = load_model()

    if model is None:
        return None   # signal to caller: use fallback

    try:
        # Build feature vector: dict → numpy array shape (1, 6)
        X = features_to_vector(features)

        # Get probability estimates
        # predict_proba returns [[P(class=0), P(class=1)]]
        # We want P(class=1) = probability of "recalled"
        proba = model.predict_proba(X)   # shape: (1, 2)
        recall_probability = proba[0][1] # P(recalled) for the first (only) sample

        # Convert 0–1 probability to 0–100 score
        recall_score = recall_probability * 100

        # Clamp to valid range (floating point can slightly exceed bounds)
        recall_score = max(0.0, min(100.0, recall_score))

        return round(recall_score, 2)

    except Exception as e:
        print(f"⚠️  Prediction failed: {e}")
        return None   # caller falls back to formula


# =============================================================================
# GET MODEL METADATA
# =============================================================================

def get_model_metadata() -> dict:
    """
    Loads and returns the metadata JSON saved by train.py.

    USED BY:
      - 2_dashboard.py: shows "Model: XGBoost (AUC: 0.82)" in footer
      - Tests: verifying model performance metrics

    Returns:
        dict with keys: active_model, trained_at, xgboost, logistic_regression
        Empty dict if metadata file doesn't exist
    """
    if not os.path.exists(METADATA_PATH):
        return {}

    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


# =============================================================================
# QUICK TEST — run this file directly to verify model works
# =============================================================================

if __name__ == "__main__":
    print("\nTesting predict.py...\n")

    if not is_model_available():
        print("❌  No trained model found.")
        print("   Run: python scripts/run_training.py  first.")
    else:
        # Test with a sample feature set
        sample_features = {
            "hours_since_last":   24.0,   # reviewed 1 day ago
            "total_reviews":      3,
            "avg_score":          0.70,
            "last_score":         0.80,
            "success_streak":     2,
            "avg_response_time":  20.0
        }

        score = predict_recall(sample_features)
        print(f"Sample features: {sample_features}")
        print(f"Predicted recall: {score}%")

        meta = get_model_metadata()
        if meta:
            print(f"\nModel metadata:")
            print(f"  Active model:  {meta.get('active_model', 'unknown')}")
            print(f"  Trained at:    {meta.get('trained_at', 'unknown')}")
            active = meta.get("active_model", "xgboost")
            model_metrics = meta.get(active, {})
            print(f"  AUC:           {model_metrics.get('auc', 'N/A')}")
            print(f"  Brier Score:   {model_metrics.get('brier', 'N/A')}")