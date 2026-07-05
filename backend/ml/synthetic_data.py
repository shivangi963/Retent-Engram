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


N_STUDENTS      = 30    # number of simulated students
N_CONCEPTS      = 8     # concepts per student
N_REVIEWS       = 6     # review sessions per concept per student
SCORE_NOISE     = 0.15  # how much randomness in scores (0 = deterministic)
RANDOM_SEED     = 42    # for reproducibility (same seed = same data every run)


def simulate_student_concept_events(
    student_id: str,
    concept_id: str,
    forgetting_rate: float,
    n_reviews: int
) -> list:
    events = []

    # Start time random point between 30 and 90 days ago
    start_offset_days = random.randint(30, 90)
    current_time = datetime.now(timezone.utc) - timedelta(days=start_offset_days)

    # track the last review time and current score for Ebbinghaus computation
    last_review_time = None
    current_recall = 0.0   # 0% before first review

    for review_idx in range(n_reviews):

        if review_idx == 0:
            # First review
            score = random.uniform(0.3, 0.7)  # first exposure score
        else:
            # Subsequent reviews that is score based on Ebbinghaus recall at this moment
            hours_elapsed = (current_time - last_review_time).total_seconds() / 3600

            # Memory half-life to determined by forgetting_rate
            # Lower rate means longer half-life means slower forgetting
            # half_life range for 12h (rate=0.9) to 120h (rate=0.1)
            half_life = 12 + (1 - forgetting_rate) * 108   # linear map

            # Ebbinghaus R = current_recall × e^(-t × ln2 / half_life)
            decay = math.exp(-hours_elapsed * 0.693 / half_life)
            true_recall = current_recall * decay

            # Add noise: score = true_recall ± noise
            noise = random.gauss(0, SCORE_NOISE)  # Gaussian noise
            score = true_recall + noise
            score = max(0.0, min(1.0, score))  # clamp to [0, 1]


        # Base: 30 minutes. Better score means faster response (less time needed)
        base_time = 30
        time_variation = random.gauss(0, 8)  # ±8 minutes randomness
        response_time = max(5, base_time - (score * 15) + time_variation)

        # Hints: worse scores means more hints used
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

        last_review_time = current_time
        current_recall = score  # new recall = what they just scored

        # Advance time by 1–7 days for next review
        gap_hours = random.randint(24, 168)   # 1–7 days
        current_time = current_time + timedelta(hours=gap_hours)

    return events


def generate_synthetic_dataset(
    n_students: int = N_STUDENTS,
    n_concepts: int = N_CONCEPTS,
    n_reviews: int   = N_REVIEWS,
    random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    
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
        # Distribute evenly
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

    print(f" Simulated {n_students} students × {n_concepts} concepts × {n_reviews} reviews")
    print(f" Total simulated events: {total_events_simulated}")
    print(f" Labeled training rows:  {len(all_rows)}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Print class balance
    label_counts = df["label"].value_counts()
    total = len(df)
    print(f"\n Synthetic label distribution:")
    print(f"    Recalled (1): {label_counts.get(1, 0)} ({label_counts.get(1,0)/total*100:.1f}%)")
    print(f"    Forgot   (0): {label_counts.get(0, 0)} ({label_counts.get(0,0)/total*100:.1f}%)")

    return df


def get_combined_training_data(
    real_df: pd.DataFrame,
    synthetic_multiplier: int = 3
) -> pd.DataFrame:
    # How many synthetic rows to generate
    if real_df.empty:
        print("\nNo real data found. Using synthetic data only.")
        # Generate enough for a decent model
        n_students_needed = 50
    else:
        real_count = len(real_df)
        # Scale synthetic to be synthetic_multiplier × real data size
        # More real data means less synthetic needed (relative)
        # Math: n_students × n_concepts × (n_reviews-1) = rows
        # So n_students = target_rows / (n_concepts × (n_reviews-1))
        target_synthetic = real_count * synthetic_multiplier
        n_students_needed = max(20, target_synthetic // (N_CONCEPTS * (N_REVIEWS - 1)))
        print(f"\n Real data: {real_count} rows → generating ~{target_synthetic} synthetic rows")

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