"""
backend/ml/predict.py
=====================
Loads the ACTIVE trained model (written by train.py) and predicts recall.

Which file is loaded is decided by model_metadata.json:
    artifact_type "sklearn"  recall_model.pkl      logistic regression / XGBoost / gradient boosting
    artifact_type "weibull"  recall_survival.json  survival model, recall = S(hours since last review | history)
    artifact_type "lstm"     recall_lstm.pt        PyTorch sequence model, reads the review history
A model is used only if training marked it `gate_passed` (so a pickle left over from an older, broken
run is ignored).  predict_recall() returns None - and the caller falls back to the decay-formula memory
model - when there is no valid model, or when a tree model is asked about a gap longer than any it
trained on (trees cannot extrapolate; the survival and LSTM curves can).
"""
import json
import pickle

import numpy as np

from backend.config import MODEL_PATH, MODEL_META_PATH
from backend.ml.data_prep import FEATURE_COLUMNS

_cache = {"stamp": None, "model": None, "meta": {}}


def _read():
    meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    if not meta.get("gate_passed") or meta.get("feature_columns") != FEATURE_COLUMNS:
        return None, meta
    kind = meta.get("artifact_type", "sklearn")                 # older metadata had no artifact_type
    path = MODEL_PATH.parent / meta.get("artifact", MODEL_PATH.name)
    if not path.exists():
        return None, meta
    if kind == "sklearn":
        with open(path, "rb") as f:
            return pickle.load(f), meta
    if kind == "weibull":
        from backend.ml.survival import WeibullSurvival
        return WeibullSurvival.load(path), meta
    if kind == "lstm":
        from backend.ml.seq_model import SequenceRecallModel
        return SequenceRecallModel.load(path), meta
    return None, meta


def _load():
    if not MODEL_META_PATH.exists():
        _cache.update(stamp=None, model=None, meta={})
        return None, {}
    try:
        meta_stamp = MODEL_META_PATH.stat().st_mtime
        files = sorted(p.stat().st_mtime for p in MODEL_PATH.parent.glob("recall_*") if p.suffix in (".pkl", ".json", ".pt"))
        stamp = (meta_stamp, tuple(files))
        if _cache["stamp"] == stamp:
            return _cache["model"], _cache["meta"]
        model, meta = _read()
    except Exception as exc:                                      # corrupt file, torch missing, ...
        print(f"  could not load model: {exc}")
        model, meta, stamp = None, {}, None
    _cache.update(stamp=stamp, model=model, meta=meta)
    return model, meta


def is_model_available() -> bool:
    return _load()[0] is not None


def active_model_name() -> str:
    model, meta = _load()
    return meta.get("active_model", "memory_model") if model is not None else "memory_model"


def get_model_metadata() -> dict:
    return _load()[1]


def predict_recall(features: dict, events=None):
    """Recall in percent (0-100) or None.  `events` = this learner's events for the concept
    (only the LSTM needs them)."""
    model, meta = _load()
    if model is None or not features:
        return None
    kind = meta.get("artifact_type", "sklearn")
    try:
        if kind == "sklearn":
            if features["hours_since_last"] > meta.get("train_max_hours", 0) * 1.05:
                return None
            p = float(model.predict_proba(np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=float))[0][1])
        elif kind == "weibull":
            from backend.ml.survival import survival_covariates
            p = float(model.survival(survival_covariates(features), features["hours_since_last"])[0])
        elif kind == "lstm":
            if not events:
                return None
            from backend.ml.sequences import encode_events, MAX_LEN
            seq = encode_events(events)[-MAX_LEN:]
            if len(seq) == 0:
                return None
            p = float(model.predict([seq], [features["hours_since_last"]])[0])
        else:
            return None
        return round(min(100.0, max(0.0, p * 100.0)), 2)
    except Exception as exc:
        print(f"  prediction failed: {exc}")
        return None
