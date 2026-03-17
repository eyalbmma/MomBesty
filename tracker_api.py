import sqlite3
import time
from typing import Optional, List, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

DB_PATH = "/data/rag.db"

router = APIRouter(prefix="/tracker", tags=["tracker"])


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def now_ts() -> int:
    return int(time.time())


# =========================
# Types
# =========================
EntryType = Literal[
    # Baby
    "feed",
    "diaper",
    "sleep",
    "breastfeed",
    "pump",
    "medicine",
    "vaccine",
    "bath",
    # Mom
    "me_time",              # 🫂 רגע בשבילי
    "period",               # 🩸 וסת
    "postpartum_bleeding",  # 🩸 דימום אחרי לידה
    "alcohol",              # 🥂/🍻 אלכוהול
]


class EntryCreate(BaseModel):
    user_id: str = Field(..., min_length=3)
    baby_id: str = Field("default")  # MVP: תינוק אחד. אפשר להרחיב בהמשך.
    type: EntryType

    # זמן האירוע (אם לא נשלח — עכשיו)
    occurred_at: Optional[int] = None

    # שדות גמישים לפי סוג:
    # - feed: method, amount_ml, duration_min?, note
    # - diaper: diaper_kind, note
    # - sleep: duration_min, note
    # - breastfeed: duration_min, method?, note
    # - pump: amount_ml?, duration_min?, method?, note
    # - medicine: method (שם התרופה), amount_ml? (מינון), note
    # - vaccine: method (שם החיסון), note
    # - bath: duration_min?, note
    # - me_time: note (מומלץ), method? (אופציונלי: "נשימה/מקלחת/הליכה")
    # - period: method? (אופציונלי: "יום 1/2/3" או "קל/בינוני/חזק"), note?
    # - postpartum_bleeding: method? (צבע/שלב), note?
    # - alcohol: method? ("יין/בירה"), amount_ml? (כמות), note?
    method: Optional[str] = None
    amount_ml: Optional[int] = None
    diaper_kind: Optional[str] = None
    duration_min: Optional[int] = None
    note: Optional[str] = None


class EntryOut(BaseModel):
    id: int
    user_id: str
    baby_id: str
    type: EntryType
    occurred_at: int

    method: Optional[str]
    amount_ml: Optional[int]
    diaper_kind: Optional[str]
    duration_min: Optional[int]
    sleep_started_at: Optional[int]
    sleep_ended_at: Optional[int]
    is_active_sleep: int
    note: Optional[str]

    status: str
    created_at: int


class SleepStartRequest(BaseModel):
    user_id: str = Field(..., min_length=3)
    baby_id: str = Field("default")
    note: Optional[str] = None


class SleepStopRequest(BaseModel):
    user_id: str = Field(..., min_length=3)
    baby_id: str = Field("default")


class ActiveSleepOut(BaseModel):
    has_active: bool
    entry: Optional[EntryOut] = None
    elapsed_seconds: Optional[int] = None


def _require(condition: bool, msg: str):
    if not condition:
        raise HTTPException(status_code=400, detail=msg)


def _normalize_baby_id(baby_id: Optional[str]) -> str:
    # Normalize baby_id to ensure consistent querying (handles empty string/None)
    norm = (baby_id or "default").strip()
    return norm or "default"


def _fetch_active_sleep(cur, user_id: str, baby_id: str):
    cur.execute(
        """
        SELECT id, user_id, baby_id, type, occurred_at, method, amount_ml, diaper_kind, duration_min,
               sleep_started_at, sleep_ended_at, is_active_sleep, note, status, created_at
        FROM tracker_entries
        WHERE user_id = ?
          AND baby_id = ?
          AND type = 'sleep'
          AND status = 'active'
          AND is_active_sleep = 1
        ORDER BY sleep_started_at DESC, id DESC
        """,
        (user_id, baby_id),
    )
    return cur.fetchall()


@router.post("/sleep/start", response_model=ActiveSleepOut)
def start_sleep(payload: SleepStartRequest):
    con = db()
    cur = con.cursor()

    baby_id = _normalize_baby_id(payload.baby_id)

    active_rows = _fetch_active_sleep(cur, payload.user_id, baby_id)
    if active_rows:
        con.close()
        raise HTTPException(status_code=400, detail="כבר קיימת שינה פעילה עבור המשתמש/תינוק")

    started_at = now_ts()
    created_at = started_at

    cur.execute(
        """
        INSERT INTO tracker_entries
        (user_id, baby_id, type, occurred_at, method, amount_ml, diaper_kind, duration_min, sleep_started_at, sleep_ended_at, is_active_sleep, note, status, created_at)
        VALUES (?, ?, 'sleep', ?, NULL, NULL, NULL, NULL, ?, NULL, 1, ?, 'active', ?)
        """,
        (payload.user_id, baby_id, started_at, started_at, payload.note, created_at),
    )
    con.commit()
    entry_id = cur.lastrowid

    cur.execute(
        """
        SELECT id, user_id, baby_id, type, occurred_at, method, amount_ml, diaper_kind, duration_min,
               sleep_started_at, sleep_ended_at, is_active_sleep, note, status, created_at
        FROM tracker_entries
        WHERE id = ?
        """,
        (entry_id,),
    )
    row = cur.fetchone()

    if row:
        print(
            "[sleep_start] Created sleep entry | "
            f"id={row['id']} user_id={row['user_id']} baby_id={row['baby_id']} "
            f"type={row['type']} status={row['status']} is_active_sleep={row['is_active_sleep']} "
            f"sleep_started_at={row['sleep_started_at']} sleep_ended_at={row['sleep_ended_at']}"
        )
    else:
        print(
            "[sleep_start] Failed to read created sleep entry immediately after insert | "
            f"user_id={payload.user_id} baby_id={baby_id}"
        )
    con.close()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to read created sleep entry")

    started_at = row["sleep_started_at"] or row["occurred_at"]
    elapsed_seconds = 0 if started_at else None

    return ActiveSleepOut(
        has_active=True,
        entry=dict(row) if row else None,
        elapsed_seconds=elapsed_seconds,
    )


@router.post("/sleep/stop", response_model=EntryOut)
def stop_sleep(payload: SleepStopRequest):
    con = db()
    cur = con.cursor()

    baby_id = _normalize_baby_id(payload.baby_id)
    active_rows = _fetch_active_sleep(cur, payload.user_id, baby_id)
    if active_rows:
        print(
            f"[sleep_stop] Found active rows for user={payload.user_id} baby_id={baby_id} "
            f"ids={[r['id'] for r in active_rows]}"
        )
    if len(active_rows) == 0:
        print(
            "[sleep_stop] No active sleep found | "
            f"user_id={payload.user_id} baby_id={baby_id} type='sleep' is_active_sleep=1 status='active' sleep_ended_at IS NULL/ANY"
        )
        con.close()
        raise HTTPException(status_code=404, detail="לא נמצאה שינה פעילה לסגירה")
    if len(active_rows) > 1:
        print(
            f"[sleep_stop] Multiple active sleeps found for user={payload.user_id} baby_id={baby_id} "
            f"ids={[r['id'] for r in active_rows]}"
        )
        con.close()
        raise HTTPException(status_code=409, detail="נמצאו מספר שינות פעילות - נדרש טיפול ידני")

    active = active_rows[0]
    started_at = active["sleep_started_at"] or active["occurred_at"]
    if not started_at:
        con.close()
        raise HTTPException(status_code=500, detail="Missing sleep_started_at for active sleep")

    ended_at = now_ts()
    duration_seconds = max(0, ended_at - started_at)
    duration_min = int(duration_seconds // 60)

    cur.execute(
        """
        UPDATE tracker_entries
        SET sleep_ended_at = ?, duration_min = ?, is_active_sleep = 0
        WHERE id = ?
        """,
        (ended_at, duration_min, active["id"]),
    )
    con.commit()

    cur.execute(
        """
        SELECT id, user_id, baby_id, type, occurred_at, method, amount_ml, diaper_kind, duration_min,
               sleep_started_at, sleep_ended_at, is_active_sleep, note, status, created_at
        FROM tracker_entries
        WHERE id = ?
        """,
        (active["id"],),
    )
    row = cur.fetchone()
    con.close()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to read stopped sleep entry")

    return dict(row)


@router.get("/sleep/active", response_model=ActiveSleepOut)
def get_active_sleep(user_id: str = Query(..., min_length=3), baby_id: str = Query("default")):
    con = db()
    cur = con.cursor()
    baby_id_norm = _normalize_baby_id(baby_id)
    rows = _fetch_active_sleep(cur, user_id, baby_id_norm)
    if rows:
        print(
            f"[sleep_active] Found active rows for user={user_id} baby_id={baby_id_norm} "
            f"ids={[r['id'] for r in rows]}"
        )
    else:
        print(f"[sleep_active] No active rows for user={user_id} baby_id={baby_id_norm}")
    con.close()

    if not rows:
        return ActiveSleepOut(has_active=False, entry=None, elapsed_seconds=None)

    active = rows[0]
    started_at = active["sleep_started_at"] or active["occurred_at"]
    elapsed_seconds = max(0, now_ts() - started_at) if started_at else None

    return ActiveSleepOut(
        has_active=True,
        entry=dict(active),
        elapsed_seconds=elapsed_seconds,
    )

@router.post("/entries", response_model=EntryOut)
def create_entry(payload: EntryCreate):
    con = db()
    cur = con.cursor()

    occurred_at = payload.occurred_at or now_ts()
    created_at = now_ts()

    # =========================
    # Lightweight validation (MVP)
    # =========================
    if payload.type == "diaper":
        _require(bool(payload.diaper_kind), "diaper_kind הוא שדה חובה עבור חיתול")

    if payload.type == "sleep":
        _require(payload.duration_min is not None and payload.duration_min >= 0, "duration_min הוא שדה חובה עבור שינה")

    if payload.type == "medicine":
        _require(bool(payload.method and payload.method.strip()), "method (שם התרופה) הוא שדה חובה עבור תרופה")

    if payload.type == "vaccine":
        _require(bool(payload.method and payload.method.strip()), "method (שם החיסון) הוא שדה חובה עבור חיסון")

    if payload.type == "breastfeed":
        # אופציונלי, אבל אם נשלח duration_min - שלא יהיה שלילי
        if payload.duration_min is not None:
            _require(payload.duration_min >= 0, "duration_min חייב להיות >= 0")

    if payload.type in ("feed", "pump", "bath", "alcohol"):
        # אם נשלחו ערכים מספריים – שלא יהיו שליליים
        if payload.amount_ml is not None:
            _require(payload.amount_ml >= 0, "amount_ml חייב להיות >= 0")
        if payload.duration_min is not None:
            _require(payload.duration_min >= 0, "duration_min חייב להיות >= 0")

    # Mom types: me_time / period / postpartum_bleeding
    # לא מחייבים שדות (כדי להשאיר MVP גמיש), אבל מוודאים שאם יש מספרים הם תקינים
    if payload.type in ("me_time", "period", "postpartum_bleeding"):
        if payload.amount_ml is not None:
            _require(payload.amount_ml >= 0, "amount_ml חייב להיות >= 0")
        if payload.duration_min is not None:
            _require(payload.duration_min >= 0, "duration_min חייב להיות >= 0")

    cur.execute(
        """
        INSERT INTO tracker_entries
        (user_id, baby_id, type, occurred_at, method, amount_ml, diaper_kind, duration_min, sleep_started_at, sleep_ended_at, is_active_sleep, note, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 'active', ?)
        """,
        (
            payload.user_id,
            payload.baby_id,
            payload.type,
            occurred_at,
            payload.method,
            payload.amount_ml,
            payload.diaper_kind,
            payload.duration_min,
            None,
            None,
            payload.note,
            created_at,
        ),
    )
    con.commit()
    entry_id = cur.lastrowid

    cur.execute(
        """
        SELECT id, user_id, baby_id, type, occurred_at, method, amount_ml, diaper_kind, duration_min, sleep_started_at, sleep_ended_at, is_active_sleep, note, status, created_at
        FROM tracker_entries
        WHERE id = ?
        """,
        (entry_id,),
    )
    row = cur.fetchone()
    con.close()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to read created entry")

    return dict(row)


@router.get("/entries", response_model=List[EntryOut])
def list_entries(
    user_id: str = Query(..., min_length=3),
    baby_id: str = Query("default"),
    type: Optional[EntryType] = Query(None),
    start: Optional[int] = Query(None, description="occurred_at >= start"),
    end: Optional[int] = Query(None, description="occurred_at <= end"),
    limit: int = Query(50, ge=1, le=200),
):
    con = db()
    cur = con.cursor()

    clauses = ["user_id = ?", "baby_id = ?", "status = 'active'"]
    params = [user_id, baby_id]

    if type is not None:
        clauses.append("type = ?")
        params.append(type)
    if start is not None:
        clauses.append("occurred_at >= ?")
        params.append(start)
    if end is not None:
        clauses.append("occurred_at <= ?")
        params.append(end)

    where_sql = " AND ".join(clauses)
    sql = f"""
        SELECT id, user_id, baby_id, type, occurred_at, method, amount_ml, diaper_kind, duration_min, sleep_started_at, sleep_ended_at, is_active_sleep, note, status, created_at
        FROM tracker_entries
        WHERE {where_sql}
        ORDER BY occurred_at DESC, id DESC
        LIMIT ?
    """
    params.append(limit)

    cur.execute(sql, tuple(params))
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


@router.delete("/entries/{entry_id}")
def delete_entry(
    entry_id: int,
    user_id: str = Query(..., min_length=3),
    baby_id: str = Query("default"),
):
    con = db()
    cur = con.cursor()

    cur.execute("SELECT user_id, baby_id, status FROM tracker_entries WHERE id = ?", (entry_id,))
    row = cur.fetchone()
    if not row:
        con.close()
        raise HTTPException(status_code=404, detail="Entry not found")

    if row["user_id"] != user_id:
        con.close()
        raise HTTPException(status_code=403, detail="Not allowed")

    # MVP: תינוק אחד, אבל נשמור עקביות אם נשלח baby_id
    if row["baby_id"] != baby_id:
        con.close()
        raise HTTPException(status_code=403, detail="Not allowed (baby mismatch)")

    cur.execute("UPDATE tracker_entries SET status = 'deleted' WHERE id = ?", (entry_id,))
    con.commit()
    con.close()
    return {"ok": True}
