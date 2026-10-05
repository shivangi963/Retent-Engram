"""
backend/ml/data_prep.py
=======================
Builds labelled training rows from event history.

Row i  = "what we knew right after review i"  ->  label = did review i+1 go well?
Features are computed AS OF the time of review i+1, so `hours_since_last` is the
gap the learner actually waited (this matches Section 4.8 of the report:
"the time gap measured to the next review").
"""
from collections import defaultdict
import pandas as pd

from backend.ml.features import extract_features, _to_dt, FEATURE_NAMES

RECALL_THRESHOLD = 0.6                      # next score >= 0.6  ->  "recalled"
FEATURE_COLUMNS = list(FEATURE_NAMES)
LABEL_COLUMN = "label"
GROUP_COLUMN = "_user_id"                   # used to keep one learner out of both train and test


def extract_labeled_rows_for_concept(events: list) -> list:
    ordered = [e for e in events if _to_dt(e.get("timestamp")) is not None]
    ordered.sort(key=lambda e: _to_dt(e["timestamp"]))
    rows = []
    for i in range(len(ordered) - 1):
        nxt = ordered[i + 1]
        feats = extract_features(ordered[: i + 1], as_of=nxt["timestamp"])
        if feats is None:
            continue
        next_score = float(nxt.get("score", 0) or 0)
        rows.append({
            **feats,
            LABEL_COLUMN: int(next_score >= RECALL_THRESHOLD),
            GROUP_COLUMN: str(nxt.get("user_id", "")),
            "_concept_id": nxt.get("concept_id", ""),
        })
    return rows


def build_training_dataset_from_mongodb() -> pd.DataFrame:
    """Real rows from MongoDB.  Returns an empty frame (never raises) if the DB is
    unreachable or there is not enough history yet."""
    try:
        from backend.db import get_collection
        all_events = list(get_collection("events").find({}, {"_id": 0}))
    except Exception as exc:                                    # DB down, etc.
        print(f"  Could not read events from MongoDB ({exc}); using synthetic data only.")
        return pd.DataFrame()

    if not all_events:
        print("  No events in MongoDB yet.")
        return pd.DataFrame()

    grouped = defaultdict(list)
    for e in all_events:
        grouped[(e.get("user_id", ""), e.get("concept_id", ""))].append(e)

    rows, skipped = [], 0
    for events in grouped.values():
        r = extract_labeled_rows_for_concept(events)
        rows.extend(r) if r else None
        skipped += 0 if r else 1

    print(f"  Real data: {len(all_events)} events, {len(grouped)} learner-concept pairs "
          f"-> {len(rows)} labelled rows ({skipped} pairs have < 2 events)")
    return pd.DataFrame(rows)


def get_X_y(df: pd.DataFrame):
    if df.empty:
        raise ValueError("empty training frame")
    return df[FEATURE_COLUMNS].values, df[LABEL_COLUMN].values
