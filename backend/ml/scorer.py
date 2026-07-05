import math

# recall thresholds to decide priority label
HIGH_THRESHOLD   = 40.0   # below 40 = High priority (urgent review needed)
MEDIUM_THRESHOLD = 65.0   # below 65 = Medium priority (review soon)
# above 65 = Low priority (still well remembered)

# Formula weights
LAST_SCORE_WEIGHT = 0.6   # recent performance matters more
AVG_SCORE_WEIGHT  = 0.4   # but history still counts

# Base stability in hours (how long a single review lasts on average)
BASE_STABILITY_HOURS = 24.0

# Streak bonus multiplier per success
STREAK_BONUS = 0.3

def compute_base_retention(last_score: float, avg_score: float) -> float:
    return (LAST_SCORE_WEIGHT * last_score) + (AVG_SCORE_WEIGHT * avg_score)


def compute_stability(total_reviews: int, success_streak: int) -> float:
    
    review_factor = math.log1p(total_reviews)                    # log(1 + n)
    streak_factor = 1 + (STREAK_BONUS * success_streak)         # 1 + 0.3 × streak
    stability = BASE_STABILITY_HOURS * review_factor * streak_factor
    return round(stability, 4)


def apply_decay(base_retention: float, hours_since_last: float, stability: float) -> float:
    if stability <= 0:
        stability = 0.001   # against division by zero

    decay_exponent = -hours_since_last / stability   # negative exponent
    decay_factor = math.exp(decay_exponent)          # e^(negative number) → 0 to 1
    current_retention = base_retention * decay_factor
    return current_retention



def compute_recall_score(features: dict) -> float:
    if features is None:
        return 0.0

    hours         = features["hours_since_last"]
    total_reviews = features["total_reviews"]
    avg_score     = features["avg_score"]
    last_score    = features["last_score"]
    streak        = features["success_streak"]

    base_retention = compute_base_retention(last_score, avg_score)
    stability      = compute_stability(total_reviews, streak)
    retention      = apply_decay(base_retention, hours, stability)

    recall_score = retention * 100
    recall_score = min(100.0, max(0.0, recall_score))   # clamp to valid range
    return round(recall_score, 2)


def get_priority(recall_score: float) -> str:
    if recall_score < HIGH_THRESHOLD:
        return "High"
    elif recall_score < MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "Low"