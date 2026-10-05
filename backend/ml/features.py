"""
backend/ml/features.py
======================
Turns a learner's event history for ONE concept into 6 numbers.

THE FIX IN THIS VERSION
-----------------------
`as_of` is the moment the prediction is made for.

  * In the app (inference):  as_of = None  -> "now".
  * In training:             as_of = timestamp of the NEXT review.

Before, training also used "now", so for a row built from an old event the
`hours_since_last` feature was "hours since that event until today" (thousands of
hours) instead of "the gap before the next review".  The model therefore could not
learn that waiting longer makes you forget - it even learned the opposite.
"""
from datetime import datetime, timezone

FEATURE_NAMES = [
    "hours_since_last", "total_reviews", "avg_score",
    "last_score", "success_streak", "avg_response_time",
]
SUCCESS_THRESHOLD = 0.6


def _to_dt(value):
    """datetime | ISO string -> aware UTC datetime (None if unusable).
    Phase-0 events stored the timestamp as an ISO *string*, newer ones as datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


_make_aware = _to_dt          # old name, kept so existing imports keep working


def _score(event) -> float:
    try:
        return min(1.0, max(0.0, float(event.get("score", 0) or 0)))
    except (TypeError, ValueError):
        return 0.0


def _minutes(event):
    if event.get("response_time_min") is not None:
        return float(event["response_time_min"])
    if event.get("response_time_sec") is not None:
        return float(event["response_time_sec"]) / 60.0
    return None


def extract_features(events: list, as_of=None):
    """events: all events for one user+concept (any order).  Returns dict or None."""
    dated = [(_to_dt(e.get("timestamp")), e) for e in events]
    dated = [(t, e) for t, e in dated if t is not None]
    if not dated:
        return None
    dated.sort(key=lambda x: x[0])
    ordered = [e for _, e in dated]
    last_ts = dated[-1][0]

    ref = _to_dt(as_of) if as_of is not None else datetime.now(timezone.utc)
    hours = max((ref - last_ts).total_seconds() / 3600.0, 0.0)

    scores = [_score(e) for e in ordered]
    times = [m for m in (_minutes(e) for e in ordered) if m is not None]

    return {
        "hours_since_last":  round(hours, 2),
        "total_reviews":     len(ordered),
        "avg_score":         round(sum(scores) / len(scores), 4),
        "last_score":        round(scores[-1], 4),
        "success_streak":    sum(1 for s in scores[-3:] if s >= SUCCESS_THRESHOLD),
        "avg_response_time": round(sum(times) / len(times), 2) if times else 0.0,
    }
