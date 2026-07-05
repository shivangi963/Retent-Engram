import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime, timezone
from collections import defaultdict
import pandas as pd

from backend.db import get_collection
from backend.ml.features import extract_features


# A score >= this threshold at the NEXT review = "recalled" (label = 1)
# A score <  this threshold                    = "forgot"  (label = 0)
RECALL_THRESHOLD = 0.6


def _make_aware(dt: datetime) -> datetime:
    """Adds UTC timezone to naive datetimes from MongoDB."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def extract_labeled_rows_for_concept(events: list) -> list:
    
#Takes all events for one user-concept pair and produces labeled training rows.
    if len(events) < 2:
        # Can't make labels with only 1 event
        return []

    # in chronological order
    sorted_events = sorted(events, key=lambda e: e["timestamp"])

    rows = []

    #len-1: 0, 1, 2, ..., N-2
    for i in range(len(sorted_events) - 1):

        #all events from index 0 up to and including index i
        history = sorted_events[: i + 1]   # [0, 1, ..., i]

        # The NEXT event
        next_event = sorted_events[i + 1]

        # Compute features from the history (calls features.py)
        features = extract_features(history)

        if features is None:
            continue 

        # Determine label recall at next review
        next_score = next_event.get("score", 0)
        label = 1 if next_score >= RECALL_THRESHOLD else 0

        row = {
            # The 6 features (inputs to the model)
            "hours_since_last":   features["hours_since_last"],
            "total_reviews":      features["total_reviews"],
            "avg_score":          features["avg_score"],
            "last_score":         features["last_score"],
            "success_streak":     features["success_streak"],
            "avg_response_time":  features["avg_response_time"],

            # prediction 
            "label": label,

            # Metadata for debugging
            "_user_id":       next_event.get("user_id", ""),
            "_concept_id":    next_event.get("concept_id", ""),
            "_next_score":    next_score,
            "_review_index":  i
        }

        rows.append(row)

    return rows


def build_training_dataset_from_mongodb() -> pd.DataFrame:
# Queries MongoDB, groups events by (user_id, concept_id), and builds a complete training dataset

    events_col = get_collection("events")

    all_events = list(events_col.find({}, {"_id": 0}))

    if not all_events:
        print("No events found in MongoDB. Run the app and log some events first.")
        return pd.DataFrame()

    print(f"Fetched {len(all_events)} total events from MongoDB")

    grouped = defaultdict(list)
    for event in all_events:
        key = (event.get("user_id", ""), event.get("concept_id", ""))
        grouped[key].append(event)

    print(f"Found {len(grouped)} unique user-concept pairs")

    all_rows = []
    skipped = 0

    for (user_id, concept_id), events in grouped.items():
        rows = extract_labeled_rows_for_concept(events)
        if rows:
            all_rows.extend(rows)
        else:
            skipped += 1  # not enough events for this pair

    print(f"Extracted {len(all_rows)} training rows")
    print(f"Skipped {skipped} pairs (< 2 events each)")

    if not all_rows:
        print("No training rows extracted. Need more event data.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Print class balance (how many 1s vs 0s)
    label_counts = df["label"].value_counts()
    total = len(df)
    print(f"\n Label distribution:")
    print(f"    Recalled  (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total*100:.1f}%)")
    print(f"    Forgot    (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total*100:.1f}%)")

    return df


FEATURE_COLUMNS = [
    "hours_since_last",
    "total_reviews",
    "avg_score",
    "last_score",
    "success_streak",
    "avg_response_time"
]

LABEL_COLUMN = "label"


def get_X_y(df: pd.DataFrame):
 #   Splits a training DataFrame into features (X) and labels (y).
 #   X = numpy array of shape (n_samples, 6)
 #   y = numpy array of shape (n_samples,) with values 0 or 1

    if df.empty:
        raise ValueError("DataFrame is empty — cannot split into X and y")

    # Check all required columns exist
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLUMNS].values    # shape: (n_samples, 6)
    y = df[LABEL_COLUMN].values        # shape: (n_samples,)

    return X, y