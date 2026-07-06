
from datetime import datetime, timezone


def make_aware(dt: datetime | None) -> datetime | None:
    """
    Converts a naive datetime to UTC-aware. Leaves already-aware
    datetimes untouched. Passes None through unchanged.

    Args:
        dt: a datetime (naive or aware), or None

    Returns:
        UTC-aware datetime, or None if dt was None
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def utc_now() -> datetime:
    """Returns the current UTC time, timezone-aware.

    Use this everywhere instead of datetime.utcnow() (naive, and
    deprecated since Python 3.12) or datetime.now() (local time).
    """
    return datetime.now(timezone.utc)