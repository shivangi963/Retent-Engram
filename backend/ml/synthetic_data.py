"""
backend/ml/synthetic_data.py
============================
Bootstrap training data for when you have few real events.

Simulated learner:
  * memory has a half-life H (hours):  recall after a gap g  =  0.5 ** (g / H)
  * a successful review (score >= 0.6) grows H, and grows it MORE when the review
    happened after a longer gap (the spacing effect); a failed review shrinks it
  * gaps between reviews span 6 hours to 60 days, so the model sees long gaps too
    (tree models cannot extrapolate beyond the gaps they were trained on)

IMPORTANT (say this in the viva): AUC measured on this data tells you how well the
model learned THIS simulator, not how well it predicts a real student.  Real numbers
come from real logged sessions / the classmate pilot.
"""
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

from backend.ml.data_prep import (
    extract_labeled_rows_for_concept, FEATURE_COLUMNS, LABEL_COLUMN, GROUP_COLUMN,
)

CONCEPT_DIFFICULTY = {
    "os": 4, "dbms": 3, "cn": 4, "dsa": 5, "python_oop": 2,
    "process_mgmt": 4, "memory_mgmt": 4, "sql": 3, "recursion": 3, "binary_search": 2,
}
MAX_GAP_HOURS = 24 * 60


def _clip(x, lo=0.0, hi=1.0):
    return float(min(hi, max(lo, x)))


def _simulate_pair(rng, user_id, concept_id, difficulty, h_mult, skill, n_reviews):
    now = datetime.now(timezone.utc)
    t = now - timedelta(days=int(rng.integers(150, 500)))
    h0 = 72.0 * h_mult * (1.3 - 0.12 * difficulty) / 0.94    # half-life after first exposure (hours)
    h = h0
    events = []
    for i in range(n_reviews):
        if i == 0:
            score = _clip(skill + rng.normal(0, 0.12) - 0.03 * difficulty, 0.15, 1.0)
        else:
            if rng.random() < 0.15:                           # occasional very long gap
                gap = float(np.exp(rng.uniform(np.log(6), np.log(MAX_GAP_HOURS))))
            else:                                             # typical: around 2-3 days
                gap = float(np.clip(np.exp(rng.normal(np.log(60), 1.0)), 4, MAX_GAP_HOURS))
            t = t + timedelta(hours=gap)
            recall = 0.5 ** (gap / h)
            score = _clip(recall * (0.75 + 0.25 * skill) + rng.normal(0, 0.10))
            if score >= 0.6:
                h *= 1.6 + 1.4 * (1.0 - recall)               # spacing effect
            else:
                h = max(h0 * 0.8, h * 0.55)
        events.append({
            "user_id": user_id, "concept_id": concept_id,
            "event_type": str(rng.choice(["reading", "quiz", "coding"])),
            "score": round(score, 3),
            "response_time_min": round(float(max(3, rng.normal(25 - 10 * score, 6))), 1),
            "hints_used": int(max(0, round((1 - score) * 4 + rng.normal(0, 1)))),
            "timestamp": t,
        })
    return events


def generate_synthetic_dataset(n_students=150, concepts_per_student=6,
                               min_reviews=4, max_reviews=9, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    concepts = list(CONCEPT_DIFFICULTY)
    rows = []
    for s in range(n_students):
        uid = f"syn_{s:04d}"
        h_mult = float(np.exp(rng.normal(0, 0.45)))           # fast vs slow forgetters
        skill = float(rng.beta(6, 3))
        for cid in rng.choice(concepts, size=min(concepts_per_student, len(concepts)), replace=False):
            n = int(rng.integers(min_reviews, max_reviews + 1))
            ev = _simulate_pair(rng, uid, str(cid), CONCEPT_DIFFICULTY[str(cid)], h_mult, skill, n)
            rows.extend(extract_labeled_rows_for_concept(ev))
    return pd.DataFrame(rows)


def get_combined_training_data(real_df: pd.DataFrame, **kw) -> pd.DataFrame:
    syn = generate_synthetic_dataset(**kw)
    print(f"  Synthetic data: {len(syn)} labelled rows, {syn[LABEL_COLUMN].mean():.0%} positive")
    if real_df is None or real_df.empty:
        return syn
    real = real_df.copy()
    real[GROUP_COLUMN] = "real_" + real[GROUP_COLUMN].astype(str)
    cols = FEATURE_COLUMNS + [LABEL_COLUMN, GROUP_COLUMN]
    return pd.concat([real[cols], syn[cols]], ignore_index=True)
