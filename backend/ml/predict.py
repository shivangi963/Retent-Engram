import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import pickle
import numpy as np
from datetime import datetime


PROJECT_ROOT    = os.path.join(os.path.dirname(__file__), "..", "..")
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "recall_model.pkl")
METADATA_PATH   = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

FEATURE_ORDER = [
    "hours_since_last",
    "total_reviews",
    "avg_score",
    "last_score",
    "success_streak",
    "avg_response_time"
]


_cached_model = None
_model_loaded_at = None   # timestamp of when we loaded (for debugging)


def is_model_available() -> bool:
    return os.path.exists(BEST_MODEL_PATH)

def load_model():
    
    #Loads the trained model from disk. Caches it after first load.

    global _cached_model, _model_loaded_at

    if _cached_model is not None:
        return _cached_model

    # Load from disk
    if not is_model_available():
        return None   # not trained yet

    try:
        with open(BEST_MODEL_PATH, "rb") as f:
            _cached_model = pickle.load(f)
        _model_loaded_at = datetime.now().isoformat()
        print(f"ML model loaded from {BEST_MODEL_PATH}")
        return _cached_model

    except Exception as e:
        print(f"Failed to load model: {e}")
        _cached_model = None
        return None


def reload_model():
   
    global _cached_model, _model_loaded_at
    _cached_model = None
    _model_loaded_at = None
    return load_model()

def features_to_vector(features: dict) -> np.ndarray:
    vector = [features[col] for col in FEATURE_ORDER]
    return np.array(vector).reshape(1, -1)   # reshape to [[f1, f2, ..., f6]]


def predict_recall(features: dict) -> float | None:
    
  #  Uses the trained ML model to predict recall probability.
    
    model = load_model()

    if model is None:
        return None   # signal to caller: use fallback

    try:
        #  dict to numpy array shape (1, 6)
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
        print(f"Prediction failed: {e}")
        return None   # caller falls back to formula


def get_model_metadata() -> dict:
    if not os.path.exists(METADATA_PATH):
        return {}

    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}



if __name__ == "__main__":
    print("\nTesting predict.py...\n")

    if not is_model_available():
        print(" No trained model found.")
        print(" Run: python scripts/run_training.py  first.")
    else:
        # testing with a sample feature set
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