"""Integer-microsecond time helpers.

Reference time is UTC microseconds since the Unix epoch.
Station local time is also stored as integer microseconds, but it is a *clock
reading*, not a reference instant.  It must never be used as if it were UTC
unless a calibration map says so.
"""

from __future__ import annotations

from datetime import datetime, timezone

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def to_us(value: datetime | str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = parse_iso(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(round(value.timestamp() * 1_000_000))


def parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_local_us(value: str) -> int:
    """Parse a station-clock ISO-8601 reading into integer microseconds.

    The reading is a counter on the station clock.  Any time-zone designator
    is rejected instead of being silently converted: converting a local
    reading to UTC is exactly the mistake this service exists to avoid.
    """
    text = value.strip()
    if "T" not in text:
        raise ValueError("station-local timestamps must be ISO-8601 with a time component")
    time_part = text[10:]
    if time_part.endswith(("Z", "z")) or "+" in time_part or "-" in time_part:
        raise ValueError("station-local timestamps must not carry a time zone or UTC offset")
    dt = datetime.fromisoformat(text)
    return int(round((dt - datetime(1970, 1, 1)).total_seconds() * 1_000_000))


def from_us(value: int | None) -> datetime | None:
    if value is None:
        return None
    return EPOCH + timedelta_us(value)


def timedelta_us(value: int):
    from datetime import timedelta

    return timedelta(microseconds=value)


def iso(value: int | None) -> str | None:
    """Format integer microseconds as ISO-8601 UTC with microseconds."""
    if value is None:
        return None
    dt = from_us(value)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def seconds_to_us(value: float) -> int:
    return int(round(value * 1_000_000))


def us_to_seconds(value: int | float | None) -> float | None:
    return None if value is None else float(value) / 1_000_000.0


def now_us() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1_000_000)
