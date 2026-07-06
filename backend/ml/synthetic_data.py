"""
backend/ml/synthetic_data.py
==============================
PHASE 4 — Synthetic Training Data Generator

THE PROBLEM
-----------
A student who just installed the app has very few events.
With only 10–20 real events, XGBoost can't learn meaningful patterns.
We need at least 200–500 training rows for reliable model performance.

THE SOLUTION — SYNTHETIC DATA
------------------------------
We SIMULATE realistic student behavior using the Ebbinghaus forgetting
curve as the ground truth. This gives us hundreds of labeled rows
that reflect real forgetting patterns, even before users log many events.

HOW SIMULATION WORKS
---------------------
For each simulated student:
  1. Assign random forgetting rate (some students forget fast, some slow)
  2. Simulate N review sessions for M concepts over T days
  3. At each review, use Ebbinghaus to compute true recall probability
  4. If true recall > 0.5 → student scores well (score ∼ 0.7–1.0)
  5. If true recall ≤ 0.5 → student scores poorly  (score ∼ 0.0–0.5)
  6. After simulation, apply extract_labeled_rows_for_concept() to get labels

WHY THIS IS VALID
------------------
The synthetic data is generated using the SAME mathematical model
that our Phase 2 formula uses. So the ML model will learn the same
patterns — but in a more flexible, non-linear way.

More importantly, even if synthetic data's distribution differs
slightly from real data, the model trained on it will still be
much better than pure formula-based scoring.

COMBINATION STRATEGY
---------------------
We COMBINE real + synthetic data:
  - Real data captures actual student-specific quirks
  - Synthetic data provides the bulk of training signal
  Combined, they give us a robust, well-calibrated model.

WHAT YOU CAN TUNE
------------------
  N_STUDENTS:    more students = more diverse patterns
  N_CONCEPTS:    more concepts per student = more rows
  N_REVIEWS:     more reviews per concept = longer event sequences
  NOISE:         how much randomness in simulated scores (0 = perfect, 1 = random)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from backend.ml.data_prep import (
    extract_labeled_rows_for_concept,
    FEATURE_COLUMNS,
    LABEL_COLUMN
)


# =============================================================================
# SIMULATION PARAMETERS — tune these if you need more/less data
# =============================================================================

N_STUDENTS      = 30    # number of simulated students
N_CONCEPTS      = 8     # concepts per student
N_REVIEWS       = 6     # review sessions per concept per student
SCORE_NOISE     = 0.15  # how much randomness in scores (0 = deterministic)
RANDOM_SEED     = 42    # for reproducibility (same seed = same data every run)


# =============================================================================
# SIMULATE ONE STUDENT'S REVIEW HISTORY
# =============================================================================

def simulate_student_concept_events(
    student_id: str,
    concept_id: str,
    forgetting_rate: float,
    n_reviews: int
) -> list:
    """
    Simulates a student reviewing one concept N times over several days.

    HOW INTER-REVIEW GAPS WORK:
      Reviews happen at random intervals (1–7 days between sessions).
      This mimics real students who don't always review on schedule.

    HOW SCORES ARE DETERMINED:
      True recall at review time = Ebbinghaus formula
      Simulated score = true_recall + noise (clamped to 0–1)
      If true_recall > 0.5: student tends to score well (0.6–1.0)
      If true_recall ≤ 0.5: student tends to score poorly (0.1–0.5)

    FORGETTING RATE EXPLAINED:
      High forgetting_rate (e.g. 0.9) → student forgets quickly
        → memory half-life ≈ 10 hours
      Low forgetting_rate (e.g. 0.1) → student forgets slowly
        → memory half-life ≈ 100 hours
      We vary this across students to create diverse patterns.

    Args:
        student_id:     unique string for this student
        concept_id:     which concept (e.g. "os", "dbms")
        forgetting_rate: float 0–1, higher = forgets faster
        n_reviews:      how many sessions to simulate

    Returns:
        list of event dicts (same format as real MongoDB events)
    """
    events = []

    # Start time: random point between 30 and 90 days ago
    start_offset_days = random.randint(30, 90)
    current_time = datetime.now(timezone.utc) - timedelta(days=start_offset_days)

    # Track the last review time and current score for Ebbinghaus computation
    last_review_time = None
    current_recall = 0.0   # 0% before first review

    for review_idx in range(n_reviews):

        if review_idx == 0:
            # First review: student starts with no memory, scores based on difficulty
            score = random.uniform(0.3, 0.7)  # first exposure score
        else:
            # Subsequent reviews: score based on Ebbinghaus recall at this moment
            hours_elapsed = (current_time - last_review_time).total_seconds() / 3600

            # Memory half-life: determined by forgetting_rate
            # Lower rate → longer half-life → slower forgetting
            # half_life range: 12h (rate=0.9) to 120h (rate=0.1)
            half_life = 12 + (1 - forgetting_rate) * 108   # linear map

            # Ebbinghaus: R = current_recall × e^(-t × ln2 / half_life)
            # Note: ln2 = 0.693, and this gives exactly R=0.5 at t=half_life
            decay = math.exp(-hours_elapsed * 0.693 / half_life)
            true_recall = current_recall * decay

            # Add noise: score = true_recall ± noise
            noise = random.gauss(0, SCORE_NOISE)  # Gaussian noise
            score = true_recall + noise
            score = max(0.0, min(1.0, score))  # clamp to [0, 1]

        # Response time: good students take less time (negative correlation with score)
        # Base: 30 minutes. Better score → faster response (less time needed)
        base_time = 30
        time_variation = random.gauss(0, 8)  # ±8 minutes randomness
        response_time = max(5, base_time - (score * 15) + time_variation)

        # Hints: worse scores → more hints used
        hints = max(0, int((1 - score) * 5 + random.gauss(0, 1)))

        # Build the event dict (same format as 1_log_event.py creates)
        event = {
            "user_id":            student_id,
            "concept_id":         concept_id,
            "event_type":         random.choice(["reading", "quiz", "coding"]),
            "score":              round(score, 4),
            "response_time_min":  round(response_time, 1),
            "hints_used":         hints,
            "timestamp":          current_time
        }
        events.append(event)

        # Update state for next iteration
        last_review_time = current_time
        current_recall = score  # new recall = what they just scored

        # Advance time by 1–7 days for next review
        gap_hours = random.randint(24, 168)   # 1–7 days
        current_time = current_time + timedelta(hours=gap_hours)

    return events


# =============================================================================
# MAIN GENERATOR — Simulate all students
# =============================================================================

def generate_synthetic_dataset(
    n_students: int = N_STUDENTS,
    n_concepts: int = N_CONCEPTS,
    n_reviews: int   = N_REVIEWS,
    random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generates a complete synthetic training dataset by simulating
    multiple students reviewing multiple concepts.

    WHAT VARIES ACROSS STUDENTS:
      - forgetting_rate: 0.1 (slow forgetter) to 0.9 (fast forgetter)
      - This creates diverse patterns for the model to learn from

    WHAT VARIES ACROSS CONCEPTS (per student):
      - Initial score on first exposure
      - Random inter-review gaps

    RETURNS:
      DataFrame with same columns as data_prep.py's real data:
        hours_since_last | total_reviews | avg_score | last_score |
        success_streak | avg_response_time | label

    Args:
        n_students:  number of simulated students
        n_concepts:  concepts per student
        n_reviews:   review sessions per concept
        random_seed: for reproducibility

    Returns:
        pd.DataFrame of labeled training rows
    """
    # Set random seed so synthetic data is identical every run
    random.seed(random_seed)
    np.random.seed(random_seed)

    concept_ids = [
        "os", "dbms", "cn", "dsa", "python_oop",
        "process_mgmt", "memory_mgmt", "sql",
        "recursion", "binary_search"
    ]

    all_rows = []
    total_events_simulated = 0

    for student_idx in range(n_students):
        student_id = f"synthetic_student_{student_idx:03d}"

        # Assign this student a forgetting rate
        # Distribute evenly: some fast forgetters, some slow
        forgetting_rate = random.uniform(0.1, 0.9)

        # Assign n_concepts random concepts to this student
        # (different students study different subsets)
        student_concepts = random.sample(concept_ids, min(n_concepts, len(concept_ids)))

        for concept_id in student_concepts:
            # Simulate event history for this student-concept pair
            events = simulate_student_concept_events(
                student_id=student_id,
                concept_id=concept_id,
                forgetting_rate=forgetting_rate,
                n_reviews=n_reviews
            )
            total_events_simulated += len(events)

            # Extract labeled training rows from the simulated events
            # (same function used on real data — this is the key design choice)
            rows = extract_labeled_rows_for_concept(events)
            all_rows.extend(rows)

    print(f"🤖  Simulated {n_students} students × {n_concepts} concepts × {n_reviews} reviews")
    print(f"📅  Total simulated events: {total_events_simulated}")
    print(f"🏷️   Labeled training rows:  {len(all_rows)}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Print class balance
    label_counts = df["label"].value_counts()
    total = len(df)
    print(f"\n📊  Synthetic label distribution:")
    print(f"    Recalled (1): {label_counts.get(1, 0)} ({label_counts.get(1,0)/total*100:.1f}%)")
    print(f"    Forgot   (0): {label_counts.get(0, 0)} ({label_counts.get(0,0)/total*100:.1f}%)")

    return df


# =============================================================================
# COMBINE REAL + SYNTHETIC
# =============================================================================

def get_combined_training_data(
    real_df: pd.DataFrame,
    synthetic_multiplier: int = 3
) -> pd.DataFrame:
    """
    Combines real MongoDB data with synthetic data.

    STRATEGY:
      - If real data has N rows, generate synthetic_multiplier × N synthetic rows
      - If real data is empty, generate a fixed default amount (1000 rows)
      - Real data is always included when available

    WHY MULTIPLY BY 3?
      We want real data to have some influence on the model, but not
      so much that rare edge cases in real data overfit the model.
      A 1:3 real:synthetic ratio works well in practice.

    Args:
        real_df:               DataFrame from data_prep.py (may be empty)
        synthetic_multiplier:  how many synthetic rows per real row

    Returns:
        pd.DataFrame: combined dataset ready for training
    """
    # How many synthetic rows to generate
    if real_df.empty:
        print("\n No real data found. Using synthetic data only.")
        # Generate enough for a decent model
        n_students_needed = 50
    else:
        real_count = len(real_df)
        # Scale synthetic to be synthetic_multiplier × real data size
        # More real data → less synthetic needed (relative)
        # Math: n_students × n_concepts × (n_reviews-1) ≈ rows
        # So n_students ≈ target_rows / (n_concepts × (n_reviews-1))
        target_synthetic = real_count * synthetic_multiplier
        n_students_needed = max(20, target_synthetic // (N_CONCEPTS * (N_REVIEWS - 1)))
        print(f"\n📊  Real data: {real_count} rows → generating ~{target_synthetic} synthetic rows")

    # Generate synthetic data
    synthetic_df = generate_synthetic_dataset(n_students=n_students_needed)

    # Combine: real first, then synthetic
    # (pandas concat stacks DataFrames vertically)
    if real_df.empty:
        combined = synthetic_df
    else:
        combined = pd.concat([real_df, synthetic_df], ignore_index=True)

    # Keep only the columns needed for training (drop metadata columns like _user_id)
    keep_cols = FEATURE_COLUMNS + [LABEL_COLUMN]
    combined = combined[keep_cols].dropna()   # drop any rows with missing values

    print(f"\n Combined training set: {len(combined)} rows total")
    return combined