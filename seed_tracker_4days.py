# seed_tracker_4days.py
# מכניס נתוני יומן (tracker) ל-4 ימים אחורה עבור user_id נתון
# שימוש: python seed_tracker_4days.py
# אם צריך להריץ על localhost: שנה BASE_URL ל-http://localhost:8000 (או מה שרלוונטי)

import time
import json
import urllib.request
import random
from collections import Counter

BASE_URL = "https://mombesty.onrender.com"
USER_ID = "acea002e-1102-44f9-8db3-c50b0ea83f8c"
BABY_ID = "default"

DAY = 24 * 60 * 60
DAYS_BACK = 4  # היום + 3 ימים אחורה

def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_json(path):
    req = urllib.request.Request(BASE_URL + path, method="GET")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def create_entry(entry_type, occurred_at, **fields):
    payload = {
        "user_id": USER_ID,
        "baby_id": BABY_ID,
        "type": entry_type,
        "occurred_at": occurred_at,
        **fields,
    }
    return post_json("/tracker/entries", payload)

def clamp_times(times, now):
    # ודא שאין timestamps "בעתיד" ביחס ל-now
    return [t for t in times if t <= now]

now = int(time.time())
print("SERVER NOW:", now)
print(f"Seeding data for last {DAYS_BACK} days for user_id={USER_ID}...")

created_ids = []

for day_index in range(DAYS_BACK):
    # day_index=0 => היום, 1 => אתמול וכו'
    day_base = now - day_index * DAY

    # 6 אירועים ביום, מפוזרים "ריאלית"
    times = clamp_times([
        day_base - 1 * 60 * 60,   # -1h
        day_base - 4 * 60 * 60,   # -4h
        day_base - 7 * 60 * 60,   # -7h
        day_base - 10 * 60 * 60,  # -10h
        day_base - 14 * 60 * 60,  # -14h
        day_base - 18 * 60 * 60,  # -18h
    ], now)

    if len(times) < 3:
        # תיאורטית לא אמור לקרות, אבל ליתר בטחון
        continue

    # feed
    created_ids.append(create_entry(
        "feed", times[0],
        method=random.choice(["bottle", "breast"]),
        amount_ml=random.randint(70, 140),
        note=f"seed feed d-{day_index}"
    ).get("id"))

    # sleep
    created_ids.append(create_entry(
        "sleep", times[1],
        duration_min=random.randint(25, 120),
        note=f"seed sleep d-{day_index}"
    ).get("id"))

    # diaper
    created_ids.append(create_entry(
        "diaper", times[2],
        diaper_kind=random.choice(["pee", "poo", "both"]),
        note=f"seed diaper d-{day_index}"
    ).get("id"))

    # breastfeed או pump
    if random.random() < 0.5:
        created_ids.append(create_entry(
            "breastfeed", times[3] if len(times) > 3 else times[-1],
            note=f"seed breastfeed d-{day_index}"
        ).get("id"))
    else:
        created_ids.append(create_entry(
            "pump", times[3] if len(times) > 3 else times[-1],
            amount_ml=random.randint(40, 140),
            note=f"seed pump d-{day_index}"
        ).get("id"))

    # sleep 2
    if len(times) > 4:
        created_ids.append(create_entry(
            "sleep", times[4],
            duration_min=random.randint(30, 150),
            note=f"seed sleep 2 d-{day_index}"
        ).get("id"))

    # feed 2
    if len(times) > 5:
        created_ids.append(create_entry(
            "feed", times[5],
            method=random.choice(["bottle", "breast"]),
            amount_ml=random.randint(60, 130),
            note=f"seed feed 2 d-{day_index}"
        ).get("id"))

print(f"Created {len(created_ids)} entries")
print("Last 10 IDs:", created_ids[-10:])

# אימות: בדיוק כמו שהאפליקציה מביאה חלונות 24 שעות
print("\nVerify per 24h window (like app pagination):")
for w in range(DAYS_BACK):
    end = now - w * DAY
    start = end - DAY
    path = f"/tracker/entries?user_id={USER_ID}&baby_id={BABY_ID}&start={start}&end={end}&limit=200"
    data = get_json(path)
    counts = Counter([e.get("type") for e in data])
    print(f"Window {w}: start={start} end={end} -> {len(data)} entries | {dict(counts)}")

print("Done.")
