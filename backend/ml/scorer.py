"""
backend/ml/scorer.py
====================
The transparent memory model (no training needed).  It is both the default scorer
and the safety net: the app falls back to it whenever the trained model is missing,
fails its sanity checks, or is asked about a gap longer than it was trained on.

    strength(t) = strength_after_review * 0.5 ** (hours / half_life)
    recall %    = chance you would pass a quick check (>= 60%) on it right now
                = logistic((strength - 0.6) / 0.10)

  * strength_after_review: how well it went last time (blend of last and average score)
  * half_life: grows with every successful review (spaced repetition) and with a streak

"Recall %" therefore means the same thing whether it comes from this model or from the
trained classifier (whose label is "next score >= 0.6"), so the dashboard never jumps
when the app switches between them.

The constants are heuristics - the report lists "learn them from data via half-life
regression" as the next step.  They are the first block below so they are easy to tune.
"""
import math

# ---- tunable constants -----------------------------------------------------
FIRST_HALF_LIFE_DAYS = 1.5     # half-life after ONE decent review
GROWTH_EXPONENT      = 1.15    # how fast the half-life grows with effective reviews
STREAK_BONUS         = 0.15    # +15% half-life per success in the last 3
LAST_SCORE_WEIGHT    = 0.6
AVG_SCORE_WEIGHT     = 0.4
MIN_STRENGTH         = 0.55    # even a poor review leaves something right after studying
LINK_CENTER          = 0.60    # strength at which the chance of passing is 50%
LINK_SCALE           = 0.10    # smaller = sharper switch from "know it" to "forgot it"
HIGH_THRESHOLD       = 40.0    # below -> High priority
MEDIUM_THRESHOLD     = 65.0    # below -> Medium priority (also the review threshold)


def initial_strength(features: dict) -> float:
    blend = LAST_SCORE_WEIGHT * features["last_score"] + AVG_SCORE_WEIGHT * features["avg_score"]
    return min(1.0, MIN_STRENGTH + (1.0 - MIN_STRENGTH) * blend)


def half_life_hours(features: dict) -> float:
    effective = max(features["total_reviews"] * features["avg_score"], 0.0)
    days = FIRST_HALF_LIFE_DAYS * (1.0 + effective) ** GROWTH_EXPONENT
    days *= 1.0 + STREAK_BONUS * features["success_streak"]
    return days * 24.0


def strength(features: dict) -> float:
    return initial_strength(features) * 0.5 ** (features["hours_since_last"] / half_life_hours(features))


def compute_recall_score(features) -> float:
    """Chance (0-100) of recalling the concept right now.  None -> 0."""
    if not features:
        return 0.0
    p = 1.0 / (1.0 + math.exp(-(strength(features) - LINK_CENTER) / LINK_SCALE))
    return round(100.0 * p, 2)


def hours_until_threshold(features: dict, threshold: float = MEDIUM_THRESHOLD) -> float:
    """Hours from NOW until recall drops below `threshold` (0 if it already has)."""
    theta = min(max(threshold / 100.0, 0.01), 0.99)
    s_needed = LINK_CENTER + LINK_SCALE * math.log(theta / (1.0 - theta))   # strength that gives `threshold`
    s0 = initial_strength(features)
    if s0 <= s_needed:
        return 0.0
    total = half_life_hours(features) * math.log2(s0 / s_needed)            # hours after the last review
    return max(0.0, total - features["hours_since_last"])


def explain(features: dict) -> dict:
    """Plain-language numbers for the UI ('your memory halves every ~3 days')."""
    return {
        "recall": compute_recall_score(features),
        "half_life_days": round(half_life_hours(features) / 24.0, 1),
        "due_in_hours": round(hours_until_threshold(features), 1),
    }


def get_priority(recall_score: float) -> str:
    if recall_score < HIGH_THRESHOLD:
        return "High"
    if recall_score < MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"
