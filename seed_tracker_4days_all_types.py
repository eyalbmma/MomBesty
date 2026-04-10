# seed_tracker_4days_all_types.py
# Seeds tracker entries for ALL screens (baby + mother) for the last 4 days (including today)
# Usage: python seed_tracker_4days_all_types.py
#
# If you run locally, change BASE_URL accordingly.

import time
import json
import urllib.request
import urllib.error
import random
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

BASE_URL = "https://mombesty.onrender.com"

USER_ID = "1722ecb2-6b3a-4f72-9b4d-e7d1b580058a"
BABY_ID = "default"

DAY = 24 * 60 * 60
DAYS_BACK = 4  # today + 3 days back

# Make runs deterministic (optional)
random.seed(42)

def _url(path: str) -> str:
    return BASE_URL.rstrip("/") + path

def request_json(method: str, path: str, payload: Optional[Dict[str, Any]] = None, retries: int = 3) -> Any:
    data = None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(_url(path), data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            # small backoff
            time.sleep(0.6 * attempt)

    raise RuntimeError(f"Request failed after {retries} attempts: {method} {path} err={last_err}")

def post_json(path: str, payload: Dict[str, Any]) -> Any:
    return request_json("POST", path, payload=payload, retries=3)

def get_json(path: str) -> Any:
    return request_json("GET", path, payload=None, retries=3)

def clamp_past(ts: int, now: int) -> int:
    return ts if ts <= now else now

def day_offsets_seconds() -> List[int]:
    """
    Offsets within a 'day' relative to a base anchor.
    We anchor each day around 'now - day_index*DAY' and subtract offsets to spread events earlier.
    """
    return [
        45 * 60,     # 00:45 ago (relative)
        2 * 60 * 60, # 02:00
        4 * 60 * 60, # 04:00
        6 * 60 * 60, # 06:00
        8 * 60 * 60, # 08:00
        10 * 60 * 60,# 10:00
        12 * 60 * 60,# 12:00
        14 * 60 * 60,# 14:00
        16 * 60 * 60,# 16:00
        18 * 60 * 60,# 18:00
        20 * 60 * 60,# 20:00
        22 * 60 * 60,# 22:00
    ]

def create_entry(entry_type: str, occurred_at: int, **fields: Any) -> Optional[int]:
    payload: Dict[str, Any] = {
        "user_id": USER_ID,
        "baby_id": BABY_ID,
        "type": entry_type,
        "occurred_at": occurred_at,
    }
    payload.update(fields)
    res = post_json("/tracker/entries", payload)
    return res.get("id") if isinstance(res, dict) else None

def choose(lst: List[Any]) -> Any:
    return lst[random.randrange(0, len(lst))]

def maybe_note(prefix: str, day_index: int) -> str:
    # Keep notes short; you can remove notes entirely if you prefer
    return f"{prefix} d-{day_index}"

def seed_one_day(day_index: int, now: int) -> List[int]:
    created: List[int] = []

    # Anchor this day around now - day_index*DAY
    anchor = now - day_index * DAY

    # Build candidate times (all in the past)
    offsets = day_offsets_seconds()
    times = [clamp_past(anchor - off, now) for off in offsets]
    times = sorted(set(times), reverse=True)  # latest first

    # Helper to pick a time index safely
    def t(i: int) -> int:
        return times[min(i, len(times) - 1)]

    # -----------------------
    # BABY TYPES (core)
    # -----------------------

    # feed x2
    created.append(create_entry(
        "feed", t(1),
        method=choose(["bottle", "breast"]),
        amount_ml=random.randint(60, 160),
        note=maybe_note("seed feed", day_index),
    ) or -1)

    created.append(create_entry(
        "feed", t(7),
        method=choose(["bottle", "breast"]),
        amount_ml=random.randint(60, 160),
        note=maybe_note("seed feed2", day_index),
    ) or -1)

    # diaper x2
    created.append(create_entry(
        "diaper", t(2),
        diaper_kind=choose(["pee", "poo", "both"]),
        note=maybe_note("seed diaper", day_index),
    ) or -1)

    created.append(create_entry(
        "diaper", t(9),
        diaper_kind=choose(["pee", "poo", "both"]),
        note=maybe_note("seed diaper2", day_index),
    ) or -1)

    # sleep x2 (durations in minutes)
    created.append(create_entry(
        "sleep", t(3),
        duration_min=random.randint(25, 140),
        note=maybe_note("seed sleep", day_index),
    ) or -1)

    created.append(create_entry(
        "sleep", t(10),
        duration_min=random.randint(20, 120),
        note=maybe_note("seed sleep2", day_index),
    ) or -1)

    # breastfeed OR pump (choose one each day)
    if random.random() < 0.5:
        created.append(create_entry(
            "breastfeed", t(4),
            note=maybe_note("seed breastfeed", day_index),
        ) or -1)
    else:
        # IMPORTANT: your updated UX saves pump duration_min and maybe note/method.
        created.append(create_entry(
            "pump", t(4),
            duration_min=random.randint(5, 25),
            note=maybe_note("seed pump", day_index),
            # method is free text in your system; keep it short
            method=choose(["ימין", "שמאל", "שתיהן"]),
        ) or -1)

    # bath (often once per day or every other day)
    if day_index % 2 == 0:
        created.append(create_entry(
            "bath", t(8),
            note=maybe_note("seed bath", day_index),
        ) or -1)

    # medicine (some days)
    if random.random() < 0.6:
        created.append(create_entry(
            "medicine", t(6),
            note=maybe_note("seed medicine", day_index),
            method=choose(["ויטמין D", "ברזל", "אחר"]),
        ) or -1)

    # vaccine (rare; make only on the oldest day so it appears once)
    if day_index == (DAYS_BACK - 1):
        created.append(create_entry(
            "vaccine", t(11),
            note="seed vaccine",
            method=choose(["טיפת חלב", "חיסון שגרה"]),
        ) or -1)

    # -----------------------
    # MOTHER TYPES (new buttons)
    # -----------------------

    # me_time (a short self-care moment)
    created.append(create_entry(
        "me_time", t(5),
        note=choose(["הליכה קצרה", "מקלחת רגועה", "10 דקות שקט", "קפה רגע לבד"]) + f" d-{day_index}",
    ) or -1)

    # period (usually not every day; seed once or twice)
    if day_index in (1, 3):
        created.append(create_entry(
            "period", t(0),
            note=choose(["קל", "בינוני", "חזק"]) + f" d-{day_index}",
        ) or -1)

    # postpartum_bleeding (can appear on multiple days)
    if day_index in (0, 2, 3):
        created.append(create_entry(
            "postpartum_bleeding", t(0),
            note=choose(["קל", "בינוני"]) + f" d-{day_index}",
        ) or -1)

    # alcohol (optional; seed only once so it doesn't look weird)
    if day_index == 2:
        created.append(create_entry(
            "alcohol", t(11),
            note=choose(["כוס יין קטנה", "בירה קטנה"]) + f" d-{day_index}",
        ) or -1)

    # Remove invalid placeholders (-1) if any request failed to return id
    created = [i for i in created if isinstance(i, int) and i > 0]
    return created

def main() -> None:
    now = int(time.time())
    print("SERVER NOW (client):", now)
    print(f"Seeding ALL types for last {DAYS_BACK} days for user_id={USER_ID} baby_id={BABY_ID}")
    print("Base URL:", BASE_URL)

    created_ids: List[int] = []

    for day_index in range(DAYS_BACK):
        ids = seed_one_day(day_index, now)
        created_ids.extend(ids)
        print(f"Day d-{day_index}: created {len(ids)} entries")

    print(f"\nCreated total {len(created_ids)} entries")
    print("Last 12 IDs:", created_ids[-12:])

    # Verify per 24h window (like app pagination)
    print("\nVerify per 24h window (like app pagination):")
    for w in range(DAYS_BACK):
        end = now - w * DAY
        start = end - DAY
        path = f"/tracker/entries?user_id={USER_ID}&baby_id={BABY_ID}&start={start}&end={end}&limit=200"
        data = get_json(path)
        if not isinstance(data, list):
            print(f"Window {w}: unexpected response type: {type(data)}")
            continue
        counts = Counter([e.get("type") for e in data])
        print(f"Window {w}: {len(data)} entries | {dict(counts)}")

    print("\nDone.")

if __name__ == "__main__":
    main()
