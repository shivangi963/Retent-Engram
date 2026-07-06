import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.time_utils import make_aware, utc_now


def get_hours_since_last(sorted_events: list, reference_time=None) -> float:

    last_event = sorted_events[-1]          # most recent event
    last_ts = make_aware(last_event["timestamp"])

    if reference_time is None:
        reference_time = utc_now()          # live inference: "as of now"
    else:
        reference_time = make_aware(reference_time)

    delta = reference_time - last_ts        # timedelta object
    hours = delta.total_seconds() / 3600    # seconds to hours
    return round(hours, 2)


def get_total_reviews(events: list) -> int:
    return len(events)


def get_avg_score(events: list) -> float:
    scores = [e.get("score", 0) for e in events]
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def get_last_score(sorted_events: list) -> float:
    return round(sorted_events[-1].get("score", 0), 4)


def get_success_streak(sorted_events: list, threshold: float = 0.6) -> int:
    recent_3 = sorted_events[-3:]                           # last 3 event
    return sum(1 for e in recent_3 if e.get("score", 0) >= threshold)


def get_avg_response_time(events: list) -> float:
    times = []
    for e in events:
        if "response_time_min" in e:
            times.append(e["response_time_min"])          # minutes
        elif "response_time_sec" in e:
            times.append(e["response_time_sec"] / 60)     #seconds to minutes

    if not times:
        return 0.0
    return round(sum(times) / len(times), 2)


ACTIVE_RECALL_TYPES = {"quiz", "coding", "review"}


def get_active_recall_ratio(events: list) -> float:
    
    if not events:
        return 0.0
    active = sum(1 for e in events if e.get("event_type") in ACTIVE_RECALL_TYPES)
    return round(active / len(events), 4)


def extract_features(events: list, reference_time=None) -> dict | None:
    
    if not events:
        return None

    sorted_events = sorted(
        events,
        key=lambda e: e["timestamp"]
    )

    features = {
        "hours_since_last":     get_hours_since_last(sorted_events, reference_time),
        "total_reviews":        get_total_reviews(events),
        "avg_score":            get_avg_score(events),
        "last_score":           get_last_score(sorted_events),
        "success_streak":       get_success_streak(sorted_events),
        "avg_response_time":    get_avg_response_time(events),
        "active_recall_ratio":  get_active_recall_ratio(events),
    }

    return features