"""Timezone-aware time helpers. All timestamps in this repo are Pacific time."""

from datetime import datetime
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")


def now_pacific() -> datetime:
    return datetime.now(tz=PACIFIC)
