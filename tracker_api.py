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


EntryType = Literal["feed", "diaper", "sleep", "breastfeed", "pump", "medicine", "vaccine", "bath"]


class EntryCreate(BaseModel):
    user_id: str = Field(..., min_length=3)
    baby_id: str = Field("default")  # MVP: תינוק אחד. אפשר להרחיב בהמשך.
    type: EntryType

    # זמן האירוע (אם לא נשלח — עכשיו)
    occurred_at: Optional[int] = None

    # פרטים גמישים לפי סוג:
    # feed: method (breast/bottle), amount_ml
    # diaper: kind (pee/poo/both)
    # sleep: duration_min (או end-start בהמשך)
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
    note: Optional[str]

    status: str
    created_at: int


@router.post("/entries", response_model=EntryOut)
def create_entry(payload: EntryCreate):
    con = db()
    cur = con.cursor()

    occurred_at = payload.occurred_at or now_ts()
    created_at = now_ts()

    # ולידציה קלה לפי type (MVP)
    if payload.type == "feed":
        # אפשר להשאיר רופף, אבל נוודא שלא מכניסים שדות לא קשורים בצורה מוזרה
        pass
    if payload.type == "diaper":
        pass
    if payload.type == "sleep":
        pass

    cur.execute("""
        INSERT INTO tracker_entries
        (user_id, baby_id, type, occurred_at, method, amount_ml, diaper_kind, duration_min, note, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
    """, (
        payload.user_id,
        payload.baby_id,
        payload.type,
        occurred_at,
        payload.method,
        payload.amount_ml,
        payload.diaper_kind,
        payload.duration_min,
        payload.note,
        created_at
    ))
    con.commit()
    entry_id = cur.lastrowid

    cur.execute("""
        SELECT id, user_id, baby_id, type, occurred_at, method, amount_ml, diaper_kind, duration_min, note, status, created_at
        FROM tracker_entries
        WHERE id = ?
    """, (entry_id,))
    row = cur.fetchone()
    con.close()
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
        SELECT id, user_id, baby_id, type, occurred_at, method, amount_ml, diaper_kind, duration_min, note, status, created_at
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
def delete_entry(entry_id: int, user_id: str = Query(..., min_length=3)):
    con = db()
    cur = con.cursor()

    cur.execute("SELECT user_id, status FROM tracker_entries WHERE id = ?", (entry_id,))
    row = cur.fetchone()
    if not row:
        con.close()
        raise HTTPException(status_code=404, detail="Entry not found")

    if row["user_id"] != user_id:
        con.close()
        raise HTTPException(status_code=403, detail="Not allowed")

    cur.execute("UPDATE tracker_entries SET status = 'deleted' WHERE id = ?", (entry_id,))
    con.commit()
    con.close()
    return {"ok": True}
