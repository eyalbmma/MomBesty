import time
from typing import Optional

SECONDS_IN_DAY = 86400

def now_ts() -> int:
    return int(time.time())

def compute_day_index(
    postpartum_start_ts: Optional[int],
    fallback_start_ts: int,
    now: Optional[int] = None,
) -> int:
    """
    מחזיר יום מאז לידה (1-based).
    אם אין תאריך לידה – סופר מה-fallback (לרוב created_at של הפרופיל).
    """
    if now is None:
        now = now_ts()

    start_ts = postpartum_start_ts or fallback_start_ts
    delta_seconds = max(0, now - start_ts)
    day_index = (delta_seconds // SECONDS_IN_DAY) + 1

    return day_index
