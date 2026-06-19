import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime, timezone

def _make_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def get_hours_since_last(sorted_events: list) -> float:
    last_event = sorted_events[-1]          # most recent event
    last_ts = _make_aware(last_event["timestamp"])
    now = datetime.now(timezone.utc)
    delta = now - last_ts                   # timedelta object
    hours = delta.total_seconds() / 3600   # convert seconds → hours
    return round(hours, 2)



def get_total_reviews(events: list) -> int:
    return len(events)


def get_avg_score(events: list) -> float:
    scores = [e.get("score", 0) for e in events]   # get score, default 0 if missing
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)



def get_last_score(sorted_events: list) -> float:
    return round(sorted_events[-1].get("score", 0), 4)


def get_success_streak(sorted_events: list, threshold: float = 0.6) -> int:
    recent_3 = sorted_events[-3:]                           # last 3 events (or fewer)
    return sum(1 for e in recent_3 if e.get("score", 0) >= threshold)


def get_avg_response_time(events: list) -> float:
    times = []
    for e in events:
        if "response_time_min" in e:
            times.append(e["response_time_min"])          # already in minutes
        elif "response_time_sec" in e:
            times.append(e["response_time_sec"] / 60)     # convert seconds → minutes
        # if neither field exists, skip this event (don't append 0, that skews avg)

    if not times:
        return 0.0
    return round(sum(times) / len(times), 2)


def extract_features(events: list) -> dict | None:
    if not events:
        return None   # can't compute features with zero data

    # Sort events by timestamp, oldest first
    # This is critical — features like last_score depend on order
    sorted_events = sorted(
        events,
        key=lambda e: e["timestamp"]  # sort by timestamp field
    )

    # Call each feature function and collect results
    features = {
        "hours_since_last":   get_hours_since_last(sorted_events),
        "total_reviews":      get_total_reviews(events),
        "avg_score":          get_avg_score(events),
        "last_score":         get_last_score(sorted_events),
        "success_streak":     get_success_streak(sorted_events),
        "avg_response_time":  get_avg_response_time(events)
    }

    return features