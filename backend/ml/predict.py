"""
backend/ml/predict.py
=====================
Loads the trained model - but only if training marked it `gate_passed`.
A pickle left over from an older, broken training run has no such flag in
model_metadata.json, so it is ignored and the memory model is used instead.

predict_recall() returns None (=> caller falls back to the memory model) when
  * there is no valid model, or
  * the gap since the last review is longer than any gap the model was trained on
    (tree models cannot extrapolate; the memory model keeps decaying correctly).
"""
import json
import pickle
import numpy as np

from backend.config import MODEL_PATH, MODEL_META_PATH
from backend.ml.data_prep import FEATURE_COLUMNS

_cache = {"stamp": None, "model": None, "meta": {}}


def _load():
    if not (MODEL_PATH.exists() and MODEL_META_PATH.exists()):
        _cache.update(stamp=None, model=None, meta={})
        return None, {}
    stamp = (MODEL_PATH.stat().st_mtime, MODEL_META_PATH.stat().st_mtime)
    if _cache["stamp"] == stamp:
        return _cache["model"], _cache["meta"]
    model, meta = None, {}
    try:
        meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
        if meta.get("gate_passed") and meta.get("feature_columns") == FEATURE_COLUMNS:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
    except Exception as exc:
        print(f"  could not load model: {exc}")
        model = None
    _cache.update(stamp=stamp, model=model, meta=meta)
    return model, meta


def is_model_available() -> bool:
    return _load()[0] is not None


def active_model_name() -> str:
    model, meta = _load()
    return meta.get("active_model", "memory_model") if model is not None else "memory_model"


def get_model_metadata() -> dict:
    return _load()[1]


def predict_recall(features: dict):
    """Recall in percent (0-100) or None."""
    model, meta = _load()
    if model is None or not features:
        return None
    if features["hours_since_last"] > meta.get("train_max_hours", 0) * 1.05:
        return None
    try:
        x = np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=float)
        p = float(model.predict_proba(x)[0][1])
        return round(min(100.0, max(0.0, p * 100.0)), 2)
    except Exception as exc:
        print(f"  prediction failed: {exc}")
        return None
