"""
backend/ml/sequences.py
=======================
One view of the data for ALL models, so they are compared on exactly the same examples.

An example = "the history of one learner on one concept up to review i" plus
   * tabular features   (6 numbers; logistic regression, XGBoost, survival)
   * the sequence       (the reviews themselves, in order; the PyTorch LSTM)
   * gap                (hours until review i+1 - how long the learner actually waited)
   * label              (1 if review i+1 scored >= 0.6)

Per review the sequence holds 4 numbers:
   score, success flag (score >= 0.6), log(1 + hours since the previous review), log(1 + minutes spent)
"""
import numpy as np
import pandas as pd

from backend.ml.features import extract_features, _to_dt, _score, _minutes, FEATURE_NAMES, SUCCESS_THRESHOLD

STEP_DIM = 4
MAX_LEN = 20                                   # the model looks at the last 20 reviews
FEATURE_COLUMNS = list(FEATURE_NAMES)


def encode_events(events: list) -> np.ndarray:
    """(n, 4) float32 array, oldest review first.  Events without a usable timestamp are dropped."""
    dated = sorted(((_to_dt(e.get("timestamp")), e) for e in events if _to_dt(e.get("timestamp")) is not None),
                   key=lambda x: x[0])
    rows, prev = [], None
    for ts, e in dated:
        gap_h = 0.0 if prev is None else max((ts - prev).total_seconds() / 3600.0, 0.0)
        score = _score(e)
        mins = _minutes(e) or 0.0
        rows.append([score, 1.0 if score >= SUCCESS_THRESHOLD else 0.0,
                     np.log1p(gap_h) / 8.0, np.log1p(max(mins, 0.0)) / 5.0])
        prev = ts
    return np.asarray(rows, dtype="float32").reshape(-1, STEP_DIM)


def build_examples(pairs: list, group_prefix: str = ""):
    """pairs: [(user_id, concept_id, events)] -> (DataFrame, list_of_sequences).
    DataFrame columns: FEATURE_COLUMNS + label + group + gap (row i <-> sequence i)."""
    rows, seqs = [], []
    for user_id, concept_id, events in pairs:
        ordered = sorted((e for e in events if _to_dt(e.get("timestamp")) is not None),
                         key=lambda e: _to_dt(e["timestamp"]))
        if len(ordered) < 2:
            continue
        enc = encode_events(ordered)
        for i in range(len(ordered) - 1):
            nxt = ordered[i + 1]
            feats = extract_features(ordered[: i + 1], as_of=nxt["timestamp"])
            if feats is None:
                continue
            rows.append({**feats, "label": int(float(nxt.get("score", 0) or 0) >= SUCCESS_THRESHOLD),
                         "group": f"{group_prefix}{user_id}", "gap": feats["hours_since_last"]})
            seqs.append(enc[max(0, i + 1 - MAX_LEN): i + 1])
    return pd.DataFrame(rows), seqs
