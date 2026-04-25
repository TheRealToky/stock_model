"""Date / timezone helpers shared across ingestion and querying.

OpenSky's arrival/departure endpoints take ``begin`` / ``end`` as UNIX
seconds (UTC).  Splitting a user-supplied date range into daily UTC
intervals is the most common operation in this package.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Iterator


def utc_day_bounds(day: datetime) -> tuple[datetime, datetime]:
    """Return the UTC midnight bounds of a given day.

    ``start`` is 00:00:00 UTC of *day* and ``end`` is 23:59:59 UTC of
    the same day.  Both returned datetimes are tz-aware.
    """
    d = day.astimezone(timezone.utc).date()
    start = datetime.combine(d, time.min, tzinfo=timezone.utc)
    end = datetime.combine(d, time(23, 59, 59), tzinfo=timezone.utc)
    return start, end


def daily_utc_intervals(
    start: datetime,
    end: datetime,
) -> Iterator[tuple[datetime, datetime]]:
    """Yield (day_start, day_end) UTC pairs covering [start, end].

    * Both inputs are coerced to UTC.
    * The iterator always yields at least one interval (the day of
      *start*) even when *start == end*.
    * The final interval is clipped at *end* so we don't query the
      future.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")

    cursor_day = start.astimezone(timezone.utc).date()
    last_day = end.astimezone(timezone.utc).date()

    while cursor_day <= last_day:
        day_start, day_end = utc_day_bounds(
            datetime.combine(cursor_day, time.min, tzinfo=timezone.utc)
        )
        # Clip the first interval to the user's start, and the last
        # interval to the user's end.
        if cursor_day == start.date():
            day_start = max(day_start, start)
        if cursor_day == last_day:
            day_end = min(day_end, end)
        yield day_start, day_end
        cursor_day = cursor_day + timedelta(days=1)


def to_unix_seconds(dt: datetime) -> int:
    """Convert a tz-aware datetime to integer UNIX seconds (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp())
