"""
backend/ml/data_prep.py
========================
PHASE 4 — Training Data Preparation

PURPOSE
-------
Phase 2 computed recall scores using a formula.
Phase 4 replaces that formula with a trained ML model.

But to TRAIN the ML model, we need LABELED DATA:
  - Input  (X): 6 features describing the student's history
  - Output (y): 1 if the student RECALLED at next review, 0 if they FORGOT

This file creates that labeled dataset from real MongoDB events.

THE CORE IDEA — "NEXT EVENT LABELING"
--------------------------------------
For every event a student has logged, we look at the NEXT event
on the same concept. The next event's score tells us whether the
student remembered or forgot between the two sessions.

Example for one student on "OS":
  Event 1: score=0.8, timestamp=Day1
  Event 2: score=0.6, timestamp=Day3   ← label for Event 1 = 1 (recalled, score ≥ 0.6)
  Event 3: score=0.3, timestamp=Day7   ← label for Event 2 = 0 (forgot, score < 0.6)
  Event 4: ...                         ← label for Event 3 = ?

  So Event 1 becomes training row: features_from_Event1 → label=1
     Event 2 becomes training row: features_from_Event2 → label=0
     Event 3 becomes training row: features_from_Event3 → label=?

Why the last event has no label:
  The last event has no "next" review yet. We don't know if the
  student remembered. So we skip the last event for training.

FEATURE COMPUTATION PER ROW
-----------------------------
For Event i, features are computed using ALL events up to and
including Event i (not including Event i+1):
  - hours_since_last: gap between Event i and Event i-1
  - total_reviews: count of events 1..i
  - avg_score: average of events 1..i
  - last_score: score of Event i
  - success_streak: last 3 of events 1..i
  - avg_response_time: average of events 1..i

MINIMUM DATA REQUIREMENT
--------------------------
For a training row to be valid, we need at least 2 events
for a concept (so we have at least one event with a "next" event).
Concepts with only 1 event are skipped.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime, timezone
from collections import defaultdict
import pandas as pd

from backend.db import get_collection
from backend.ml.features import extract_features


# =============================================================================
# CONSTANTS
# =============================================================================

# A score >= this threshold at the NEXT review = "recalled" (label = 1)
# A score <  this threshold                    = "forgot"  (label = 0)
RECALL_THRESHOLD = 0.6


# =============================================================================
# HELPER — Make datetime timezone-aware
# =============================================================================

def _make_aware(dt: datetime) -> datetime:
    """Adds UTC timezone to naive datetimes from MongoDB."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# =============================================================================
# STEP 1 — Extract labeled rows for one user-concept pair
# =============================================================================

def extract_labeled_rows_for_concept(events: list) -> list:
    """
    Takes all events for one user-concept pair and produces labeled training rows.

    ALGORITHM:
      1. Sort events by timestamp (oldest first)
      2. For each event at index i (except the last):
         a. Take events[0..i] as the "history so far"
         b. Compute features from this history
         c. Look at events[i+1] — the NEXT review
         d. Label = 1 if events[i+1].score >= 0.6 else 0
         e. Append (features, label) to results
      3. Skip the last event (no next review to use as label)

    MINIMUM EVENTS NEEDED:
      At least 2 events → produces 1 training row
      3 events → 2 rows
      N events → N-1 rows

    EXAMPLE:
      events = [
        {score: 0.5, timestamp: Day1},
        {score: 0.8, timestamp: Day3},   ← label for row1 = 1 (0.8 ≥ 0.6)
        {score: 0.3, timestamp: Day10},  ← label for row2 = 0 (0.3 < 0.6)
      ]
      → 2 training rows produced

    Args:
        events: list of event dicts for ONE user-concept pair

    Returns:
        list of dicts, each with 6 features + "label" field
    """
    if len(events) < 2:
        # Can't make labels with only 1 event
        return []

    # Sort oldest → newest so we process in chronological order
    sorted_events = sorted(events, key=lambda e: e["timestamp"])

    rows = []

    # Loop through all events EXCEPT the last one
    # range(len-1) means: indices 0, 1, 2, ..., N-2
    # The last index (N-1) is skipped because it has no "next" event
    for i in range(len(sorted_events) - 1):

        # "History so far" = all events from index 0 up to and including index i
        # This is what the model "knows" at the time of event i
        history = sorted_events[: i + 1]   # Python slice: [0, 1, ..., i]

        # The NEXT event after event i — this gives us the label
        next_event = sorted_events[i + 1]

        # Compute features from the history (calls features.py)
        features = extract_features(history)

        if features is None:
            continue   # skip if feature extraction failed

        # Determine label: did student recall at next review?
        next_score = next_event.get("score", 0)
        label = 1 if next_score >= RECALL_THRESHOLD else 0

        # Build training row: all 6 features + the label
        row = {
            # The 6 features (inputs to the model)
            "hours_since_last":   features["hours_since_last"],
            "total_reviews":      features["total_reviews"],
            "avg_score":          features["avg_score"],
            "last_score":         features["last_score"],
            "success_streak":     features["success_streak"],
            "avg_response_time":  features["avg_response_time"],

            # The label (what we're trying to predict)
            "label": label,

            # Metadata (not used in training, useful for debugging)
            "_user_id":       next_event.get("user_id", ""),
            "_concept_id":    next_event.get("concept_id", ""),
            "_next_score":    next_score,
            "_review_index":  i
        }

        rows.append(row)

    return rows


# =============================================================================
# STEP 2 — Extract labeled rows for ALL users and concepts
# =============================================================================

def build_training_dataset_from_mongodb() -> pd.DataFrame:
    """
    Queries MongoDB, groups events by (user_id, concept_id), and builds
    a complete training dataset.

    HOW IT WORKS:
      1. Fetch all events from MongoDB
      2. Group by (user_id, concept_id)
      3. For each group, call extract_labeled_rows_for_concept()
      4. Combine all rows into one DataFrame

    WHAT THE DATAFRAME LOOKS LIKE:
      hours_since_last | total_reviews | avg_score | last_score | success_streak | avg_response_time | label
      -----------------|---------------|-----------|------------|----------------|-------------------|------
      18.5             | 3             | 0.70      | 0.80       | 2              | 20.0              | 1
      48.0             | 2             | 0.60      | 0.50       | 1              | 25.0              | 0
      ...

    Returns:
        pd.DataFrame with columns:
            hours_since_last, total_reviews, avg_score, last_score,
            success_streak, avg_response_time, label
        Returns empty DataFrame if no valid training data found.
    """
    events_col = get_collection("events")

    # Fetch ALL events from MongoDB (no user filter — we want all users)
    all_events = list(events_col.find({}, {"_id": 0}))

    if not all_events:
        print("⚠️  No events found in MongoDB. Run the app and log some events first.")
        return pd.DataFrame()

    print(f"📥  Fetched {len(all_events)} total events from MongoDB")

    # Group events by (user_id, concept_id)
    # The key is a tuple: ("shivangi_01", "os")
    grouped = defaultdict(list)
    for event in all_events:
        key = (event.get("user_id", ""), event.get("concept_id", ""))
        grouped[key].append(event)

    print(f"📊  Found {len(grouped)} unique user-concept pairs")

    # Extract labeled rows from each group
    all_rows = []
    skipped = 0

    for (user_id, concept_id), events in grouped.items():
        rows = extract_labeled_rows_for_concept(events)
        if rows:
            all_rows.extend(rows)
        else:
            skipped += 1  # not enough events for this pair

    print(f"✅  Extracted {len(all_rows)} training rows")
    print(f"⏭️   Skipped {skipped} pairs (< 2 events each)")

    if not all_rows:
        print("⚠️  No training rows extracted. Need more event data.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Print class balance (how many 1s vs 0s)
    label_counts = df["label"].value_counts()
    total = len(df)
    print(f"\n📈  Label distribution:")
    print(f"    Recalled  (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/total*100:.1f}%)")
    print(f"    Forgot    (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/total*100:.1f}%)")

    return df


# =============================================================================
# STEP 3 — Get feature columns (used by train.py and predict.py)
# =============================================================================

# This list defines the EXACT ORDER of features for the model.
# CRITICAL: train.py and predict.py MUST use the same column order.
# If you add a new feature later, append it to this list AND retrain.
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
    """
    Splits a training DataFrame into features (X) and labels (y).

    X = numpy array of shape (n_samples, 6)
    y = numpy array of shape (n_samples,) with values 0 or 1

    WHY THIS FUNCTION?
      Both train.py and any analysis code need to split the same way.
      Having one function ensures X and y are always prepared identically.

    Args:
        df: DataFrame from build_training_dataset_from_mongodb()

    Returns:
        tuple: (X, y) numpy arrays
    """
    if df.empty:
        raise ValueError("DataFrame is empty — cannot split into X and y")

    # Check all required columns exist
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLUMNS].values    # shape: (n_samples, 6)
    y = df[LABEL_COLUMN].values        # shape: (n_samples,)

    return X, y