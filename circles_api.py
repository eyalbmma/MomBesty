import os
import os
from pydantic import BaseModel, Field
import sqlite3
import time
from typing import Optional, List, Dict, Any


from fastapi import APIRouter, HTTPException, Query

DB_PATH = "/data/rag.db"
router = APIRouter(prefix="/circles", tags=["circles"])


def db():
    con = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def now_ts() -> int:
    return int(time.time())


# ---------- Helpers to attach areas/categories ----------
def _load_pro_meta(con: sqlite3.Connection, pro_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not pro_ids:
        return {}

    cur = con.cursor()
    placeholders = ",".join(["?"] * len(pro_ids))

    # categories
    cur.execute(
        f"SELECT pro_id, category FROM circles_pro_categories WHERE pro_id IN ({placeholders})",
        pro_ids,
    )
    cat_rows = cur.fetchall()

    # areas
    cur.execute(
        f"SELECT pro_id, area_id FROM circles_pro_areas WHERE pro_id IN ({placeholders})",
        pro_ids,
    )
    area_rows = cur.fetchall()

    out: Dict[str, Dict[str, Any]] = {pid: {"categories": [], "areaIds": []} for pid in pro_ids}
    for r in cat_rows:
        out[r["pro_id"]]["categories"].append(r["category"])
    for r in area_rows:
        out[r["pro_id"]]["areaIds"].append(r["area_id"])
    return out


def _load_group_areas(con: sqlite3.Connection, group_ids: List[str]) -> Dict[str, List[str]]:
    if not group_ids:
        return {}
    cur = con.cursor()
    placeholders = ",".join(["?"] * len(group_ids))
    cur.execute(
        f"SELECT group_id, area_id FROM circles_group_areas WHERE group_id IN ({placeholders})",
        group_ids,
    )
    rows = cur.fetchall()
    out: Dict[str, List[str]] = {gid: [] for gid in group_ids}
    for r in rows:
        out[r["group_id"]].append(r["area_id"])
    return out


def _load_event_areas(con: sqlite3.Connection, event_ids: List[str]) -> Dict[str, List[str]]:
    if not event_ids:
        return {}
    cur = con.cursor()
    placeholders = ",".join(["?"] * len(event_ids))
    cur.execute(
        f"SELECT event_id, area_id FROM circles_event_areas WHERE event_id IN ({placeholders})",
        event_ids,
    )
    rows = cur.fetchall()
    out: Dict[str, List[str]] = {eid: [] for eid in event_ids}
    for r in rows:
        out[r["event_id"]].append(r["area_id"])
    return out


# ---------- Endpoints ----------
@router.get("/areas")
def list_areas():
    con = db()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, name, order_index
            FROM circles_areas
            ORDER BY order_index ASC, name ASC
            """
        )
        rows = cur.fetchall()
        return {"ok": True, "areas": [dict(r) for r in rows]}
    finally:
        con.close()


@router.get("/pros")
def list_pros(
    q: Optional[str] = Query(None),
    area_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    mode: Optional[str] = Query(None, description="online | inPerson"),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0, le=5000),
):
    """
    Filters:
      - q: search name + short_bio (LIKE)
      - area_id: pro has area in circles_pro_areas
      - category: pro has category in circles_pro_categories
      - mode: online | inPerson
    """
    con = db()
    try:
        cur = con.cursor()
        where = ["p.is_active=1"]
        params: List[Any] = []

        if q:
            needle = f"%{q.strip()}%"
            where.append("(p.name LIKE ? OR p.short_bio LIKE ?)")
            params.extend([needle, needle])

        if mode == "online":
            where.append("p.is_online=1")
        elif mode == "inPerson":
            where.append("p.is_in_person=1")
        elif mode is not None:
            raise HTTPException(status_code=400, detail="mode must be online|inPerson")

        # Filtering by area/category via EXISTS to avoid duplicates
        if area_id:
            where.append(
                "EXISTS (SELECT 1 FROM circles_pro_areas pa WHERE pa.pro_id=p.id AND pa.area_id=?)"
            )
            params.append(area_id)

        if category:
            where.append(
                "EXISTS (SELECT 1 FROM circles_pro_categories pc WHERE pc.pro_id=p.id AND pc.category=?)"
            )
            params.append(category)

        where_sql = " AND ".join(where)

        cur.execute(
            f"""
            SELECT
              p.id, p.name, p.short_bio,
              p.is_online, p.is_in_person,
              p.phone_whatsapp, p.website_url, p.instagram_url
            FROM circles_pros p
            WHERE {where_sql}
            ORDER BY p.name ASC
            LIMIT ? OFFSET ?
            """,
            (*params, int(limit), int(offset)),
        )
        rows = cur.fetchall()

        pro_ids = [r["id"] for r in rows]
        meta = _load_pro_meta(con, pro_ids)

        pros = []
        for r in rows:
            pid = r["id"]
            pros.append(
                {
                    "id": pid,
                    "name": r["name"],
                    "shortBio": r["short_bio"],
                    "isOnline": bool(int(r["is_online"] or 0)),
                    "isInPerson": bool(int(r["is_in_person"] or 0)),
                    "phoneWhatsApp": r["phone_whatsapp"],
                    "websiteUrl": r["website_url"],
                    "instagramUrl": r["instagram_url"],
                    "categories": meta.get(pid, {}).get("categories", []),
                    "areaIds": meta.get(pid, {}).get("areaIds", []),
                }
            )

        return {"ok": True, "pros": pros, "limit": limit, "offset": offset}
    finally:
        con.close()


@router.get("/pros/{pro_id}")
def get_pro(pro_id: str):
    con = db()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT
              id, name, short_bio, is_online, is_in_person,
              phone_whatsapp, website_url, instagram_url, is_active
            FROM circles_pros
            WHERE id=?
            """,
            (pro_id,),
        )
        r = cur.fetchone()
        if not r or int(r["is_active"] or 0) != 1:
            raise HTTPException(status_code=404, detail="Pro not found")

        meta = _load_pro_meta(con, [pro_id]).get(pro_id, {"categories": [], "areaIds": []})

        return {
            "ok": True,
            "pro": {
                "id": r["id"],
                "name": r["name"],
                "shortBio": r["short_bio"],
                "isOnline": bool(int(r["is_online"] or 0)),
                "isInPerson": bool(int(r["is_in_person"] or 0)),
                "phoneWhatsApp": r["phone_whatsapp"],
                "websiteUrl": r["website_url"],
                "instagramUrl": r["instagram_url"],
                "categories": meta["categories"],
                "areaIds": meta["areaIds"],
            },
        }
    finally:
        con.close()


@router.get("/groups")
def list_groups(
    q: Optional[str] = Query(None),
    area_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0, le=5000),
):
    con = db()
    try:
        cur = con.cursor()
        where = ["g.is_active=1"]
        params: List[Any] = []

        if q:
            needle = f"%{q.strip()}%"
            where.append("(g.name LIKE ? OR g.description LIKE ?)")
            params.extend([needle, needle])

        if area_id:
            where.append(
                "EXISTS (SELECT 1 FROM circles_group_areas ga WHERE ga.group_id=g.id AND ga.area_id=?)"
            )
            params.append(area_id)

        where_sql = " AND ".join(where)

        cur.execute(
            f"""
            SELECT id, name, description, join_url
            FROM circles_groups g
            WHERE {where_sql}
            ORDER BY g.name ASC
            LIMIT ? OFFSET ?
            """,
            (*params, int(limit), int(offset)),
        )
        rows = cur.fetchall()

        group_ids = [r["id"] for r in rows]
        areas_map = _load_group_areas(con, group_ids)

        groups = []
        for r in rows:
            gid = r["id"]
            groups.append(
                {
                    "id": gid,
                    "name": r["name"],
                    "description": r["description"],
                    "joinUrl": r["join_url"],
                    "areaIds": areas_map.get(gid, []),
                }
            )

        return {"ok": True, "groups": groups, "limit": limit, "offset": offset}
    finally:
        con.close()


@router.get("/events")
def list_events(
    area_id: Optional[str] = Query(None),
    from_ts: Optional[int] = Query(None, description="default now (hide past)"),
    include_past: bool = Query(False),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0, le=5000),
):
    con = db()
    try:
        cur = con.cursor()
        where = ["e.is_active=1"]
        params: List[Any] = []

        if not include_past:
            start = int(from_ts) if from_ts is not None else now_ts()
            where.append("e.starts_at >= ?")
            params.append(start)

        if area_id:
            where.append(
                "EXISTS (SELECT 1 FROM circles_event_areas ea WHERE ea.event_id=e.id AND ea.area_id=?)"
            )
            params.append(area_id)

        where_sql = " AND ".join(where)

        cur.execute(
            f"""
            SELECT id, title, starts_at, description, signup_url
            FROM circles_events e
            WHERE {where_sql}
            ORDER BY e.starts_at ASC
            LIMIT ? OFFSET ?
            """,
            (*params, int(limit), int(offset)),
        )
        rows = cur.fetchall()

        event_ids = [r["id"] for r in rows]
        areas_map = _load_event_areas(con, event_ids)

        events = []
        for r in rows:
            eid = r["id"]
            events.append(
                {
                    "id": eid,
                    "title": r["title"],
                    "startsAt": int(r["starts_at"]),
                    "description": r["description"],
                    "signupUrl": r["signup_url"],
                    "areaIds": areas_map.get(eid, []),
                }
            )

        return {"ok": True, "events": events, "limit": limit, "offset": offset}
    finally:
        con.close()



from fastapi import Header

ADMIN_SEED_KEY = os.getenv("CIRCLES_ADMIN_KEY", "change-me")

@router.post("/admin/seed")
def admin_seed(x_admin_key: str = Header(default="")):
    if x_admin_key != ADMIN_SEED_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    con = db()
    try:
        cur = con.cursor()
        ts = now_ts()

        # 1) Areas
        cur.executemany(
            "INSERT OR IGNORE INTO circles_areas(id, name, order_index) VALUES(?,?,?)",
            [
                ("tel_aviv", "תל אביב", 1),
                ("gush_dan", "גוש דן", 2),
                ("jerusalem", "ירושלים", 3),
                ("sharon", "השרון", 4),
                ("haifa", "חיפה", 5),
                ("beer_sheva", "באר שבע", 6),
                ("north", "צפון", 7),
                ("south", "דרום", 8),
            ],
        )

        # 2) Pros
        cur.executemany(
            """
            INSERT OR IGNORE INTO circles_pros(
              id,name,short_bio,is_online,is_in_person,phone_whatsapp,website_url,instagram_url,is_active,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                ("pro1", "לירון כהן", "יועצת הנקה | גישה רגועה ותכל'ס", 1, 1, "0500000000", "https://example.com/pro1", None, 1, ts, ts),
                ("pro2", "נועה לוי", "יועצת שינה | בלי אשמה, עם תוכנית", 1, 0, None, None, "https://instagram.com/example2", 1, ts, ts),
                ("pro3", "אורית פרידמן", "דולה אחרי לידה | תמיכה בבית", 0, 1, "0500000001", None, None, 1, ts, ts),
                ("pro4", "מיכל רז", "מלווה התפתחותית | תרגילים פשוטים בבית", 1, 1, "0500000002", "https://example.com/pro4", None, 1, ts, ts),
            ],
        )

        cur.executemany(
            "INSERT OR IGNORE INTO circles_pro_categories(pro_id, category) VALUES(?,?)",
            [
                ("pro1", "LactationConsultant"),
                ("pro2", "SleepConsultant"),
                ("pro3", "Doula"),
                ("pro4", "DevelopmentCoach"),
            ],
        )

        cur.executemany(
            "INSERT OR IGNORE INTO circles_pro_areas(pro_id, area_id) VALUES(?,?)",
            [
                ("pro1", "tel_aviv"),
                ("pro1", "jerusalem"),
                ("pro2", "sharon"),
                ("pro3", "beer_sheva"),
                ("pro4", "haifa"),
            ],
        )

        # 3) Groups
        cur.executemany(
            """
            INSERT OR IGNORE INTO circles_groups(
              id,name,description,join_url,is_active,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?)
            """,
            [
                ("g1", "מעגל אימהות – תל אביב", "מפגש שבועי קטן ומכיל", "https://chat.whatsapp.com/ta", 1, ts, ts),
                ("g2", "מעגל אימהות – ירושלים", "שיח עדין, בלי לחץ", "https://chat.whatsapp.com/jlm", 1, ts, ts),
                ("g3", "קבוצת ווטסאפ – השרון", "תמיכה ושאלות בלי הצפה", "https://chat.whatsapp.com/sharon", 1, ts, ts),
            ],
        )

        cur.executemany(
            "INSERT OR IGNORE INTO circles_group_areas(group_id, area_id) VALUES(?,?)",
            [
                ("g1", "tel_aviv"),
                ("g2", "jerusalem"),
                ("g3", "sharon"),
            ],
        )

        # 4) Events
        cur.executemany(
            """
            INSERT OR IGNORE INTO circles_events(
              id,title,starts_at,description,signup_url,is_active,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            [
                ("e1", "שעת סיפור לתינוקות", ts + 3 * 86400, "מפגש קצר ונעים", "https://example.com/event1", 1, ts, ts),
                ("e2", "מפגש אימהות בעגלה", ts + 7 * 86400, "הליכה קלה ושיחה", "https://example.com/event2", 1, ts, ts),
                ("e3", "קפה בוקר לאימהות", ts + 10 * 86400, "שעה של נשימה וחיבור", "https://example.com/event3", 1, ts, ts),
            ],
        )

        cur.executemany(
            "INSERT OR IGNORE INTO circles_event_areas(event_id, area_id) VALUES(?,?)",
            [
                ("e1", "tel_aviv"),
                ("e2", "sharon"),
                ("e3", "jerusalem"),
            ],
        )

        con.commit()
        return {"ok": True, "seeded": True}
    finally:
        con.close()

# =========================
# Admin auth
# =========================
ADMIN_KEY = os.getenv("CIRCLES_ADMIN_KEY", "change-me")

def _require_admin(x_admin_key: str):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")


# =========================
# Admin Models (Pros)
# =========================
class ProCreateReq(BaseModel):
    id: str = Field(..., min_length=2, max_length=64)   # למשל "pro5"
    name: str = Field(..., min_length=2, max_length=120)
    shortBio: str | None = None
    isOnline: bool = False
    isInPerson: bool = True
    phoneWhatsApp: str | None = None
    websiteUrl: str | None = None
    instagramUrl: str | None = None
    categories: list[str] = Field(default_factory=list)
    areaIds: list[str] = Field(default_factory=list)


class ProPatchReq(BaseModel):
    name: str | None = None
    shortBio: str | None = None
    isOnline: bool | None = None
    isInPerson: bool | None = None
    phoneWhatsApp: str | None = None
    websiteUrl: str | None = None
    instagramUrl: str | None = None
    isActive: bool | None = None  # אופציונלי


class ProSetListReq(BaseModel):
    values: list[str] = Field(default_factory=list)


# =========================
# Admin Helpers
# =========================
def _pro_exists(con: sqlite3.Connection, pro_id: str) -> bool:
    cur = con.cursor()
    cur.execute("SELECT 1 FROM circles_pros WHERE id=?", (pro_id,))
    return cur.fetchone() is not None


def _assert_areas_exist(con: sqlite3.Connection, area_ids: list[str]):
    if not area_ids:
        return
    cur = con.cursor()
    placeholders = ",".join(["?"] * len(area_ids))
    cur.execute(f"SELECT id FROM circles_areas WHERE id IN ({placeholders})", area_ids)
    got = {r["id"] for r in cur.fetchall()}
    missing = [a for a in area_ids if a not in got]
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown areaIds: {missing}")


def _normalize_list(xs: list[str]) -> list[str]:
    out = []
    seen = set()
    for x in xs or []:
        x = (x or "").strip()
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# =========================
# Admin Endpoints (Pros)
# =========================

@router.post("/admin/pros")
def admin_create_pro(req: ProCreateReq, x_admin_key: str = Header(default="")):
    _require_admin(x_admin_key)

    pro_id = (req.id or "").strip()
    name = (req.name or "").strip()
    if not pro_id:
        raise HTTPException(status_code=400, detail="id is required")
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    categories = _normalize_list(req.categories)
    area_ids = _normalize_list(req.areaIds)

    con = db()
    try:
        cur = con.cursor()

        # אין כפילויות
        if _pro_exists(con, pro_id):
            raise HTTPException(status_code=409, detail="pro id already exists")

        _assert_areas_exist(con, area_ids)

        ts = now_ts()
        cur.execute(
            """
            INSERT INTO circles_pros(
              id,name,short_bio,is_online,is_in_person,
              phone_whatsapp,website_url,instagram_url,
              is_active,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                pro_id,
                name,
                req.shortBio,
                1 if req.isOnline else 0,
                1 if req.isInPerson else 0,
                req.phoneWhatsApp,
                req.websiteUrl,
                req.instagramUrl,
                1,
                ts,
                ts,
            ),
        )

        # categories
        if categories:
            cur.executemany(
                "INSERT OR IGNORE INTO circles_pro_categories(pro_id, category) VALUES(?,?)",
                [(pro_id, c) for c in categories],
            )

        # areas
        if area_ids:
            cur.executemany(
                "INSERT OR IGNORE INTO circles_pro_areas(pro_id, area_id) VALUES(?,?)",
                [(pro_id, a) for a in area_ids],
            )

        con.commit()

        # החזרה בפורמט של ה-GET
        meta = _load_pro_meta(con, [pro_id]).get(pro_id, {"categories": [], "areaIds": []})
        return {
            "ok": True,
            "pro": {
                "id": pro_id,
                "name": name,
                "shortBio": req.shortBio,
                "isOnline": bool(req.isOnline),
                "isInPerson": bool(req.isInPerson),
                "phoneWhatsApp": req.phoneWhatsApp,
                "websiteUrl": req.websiteUrl,
                "instagramUrl": req.instagramUrl,
                "categories": meta["categories"],
                "areaIds": meta["areaIds"],
            },
        }
    finally:
        con.close()


@router.patch("/admin/pros/{pro_id}")
def admin_patch_pro(pro_id: str, req: ProPatchReq, x_admin_key: str = Header(default="")):
    _require_admin(x_admin_key)

    pro_id = (pro_id or "").strip()
    if not pro_id:
        raise HTTPException(status_code=400, detail="pro_id is required")

    con = db()
    try:
        cur = con.cursor()
        cur.execute("SELECT 1 FROM circles_pros WHERE id=?", (pro_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Pro not found")

        fields = []
        params: list[Any] = []

        if req.name is not None:
            nm = (req.name or "").strip()
            if not nm:
                raise HTTPException(status_code=400, detail="name cannot be empty")
            fields.append("name=?")
            params.append(nm)

        if req.shortBio is not None:
            fields.append("short_bio=?")
            params.append(req.shortBio)

        if req.isOnline is not None:
            fields.append("is_online=?")
            params.append(1 if req.isOnline else 0)

        if req.isInPerson is not None:
            fields.append("is_in_person=?")
            params.append(1 if req.isInPerson else 0)

        if req.phoneWhatsApp is not None:
            fields.append("phone_whatsapp=?")
            params.append(req.phoneWhatsApp)

        if req.websiteUrl is not None:
            fields.append("website_url=?")
            params.append(req.websiteUrl)

        if req.instagramUrl is not None:
            fields.append("instagram_url=?")
            params.append(req.instagramUrl)

        if req.isActive is not None:
            fields.append("is_active=?")
            params.append(1 if req.isActive else 0)

        # nothing to update
        if not fields:
            return {"ok": True, "updated": False}

        fields.append("updated_at=?")
        params.append(now_ts())

        params.append(pro_id)

        cur.execute(
            f"UPDATE circles_pros SET {', '.join(fields)} WHERE id=?",
            tuple(params),
        )
        con.commit()

        # החזרה בפורמט GET
        cur.execute(
            """
            SELECT id,name,short_bio,is_online,is_in_person,phone_whatsapp,website_url,instagram_url,is_active
            FROM circles_pros WHERE id=?
            """,
            (pro_id,),
        )
        r = cur.fetchone()
        meta = _load_pro_meta(con, [pro_id]).get(pro_id, {"categories": [], "areaIds": []})

        return {
            "ok": True,
            "updated": True,
            "pro": {
                "id": r["id"],
                "name": r["name"],
                "shortBio": r["short_bio"],
                "isOnline": bool(int(r["is_online"] or 0)),
                "isInPerson": bool(int(r["is_in_person"] or 0)),
                "phoneWhatsApp": r["phone_whatsapp"],
                "websiteUrl": r["website_url"],
                "instagramUrl": r["instagram_url"],
                "categories": meta["categories"],
                "areaIds": meta["areaIds"],
                "isActive": bool(int(r["is_active"] or 0)),
            },
        }
    finally:
        con.close()


@router.put("/admin/pros/{pro_id}/categories")
def admin_set_pro_categories(pro_id: str, req: ProSetListReq, x_admin_key: str = Header(default="")):
    _require_admin(x_admin_key)

    pro_id = (pro_id or "").strip()
    categories = _normalize_list(req.values)

    con = db()
    try:
        cur = con.cursor()
        if not _pro_exists(con, pro_id):
            raise HTTPException(status_code=404, detail="Pro not found")

        # replace-all
        cur.execute("DELETE FROM circles_pro_categories WHERE pro_id=?", (pro_id,))
        if categories:
            cur.executemany(
                "INSERT OR IGNORE INTO circles_pro_categories(pro_id, category) VALUES(?,?)",
                [(pro_id, c) for c in categories],
            )

        cur.execute("UPDATE circles_pros SET updated_at=? WHERE id=?", (now_ts(), pro_id))
        con.commit()

        meta = _load_pro_meta(con, [pro_id]).get(pro_id, {"categories": [], "areaIds": []})
        return {"ok": True, "pro_id": pro_id, "categories": meta["categories"]}
    finally:
        con.close()


@router.put("/admin/pros/{pro_id}/areas")
def admin_set_pro_areas(pro_id: str, req: ProSetListReq, x_admin_key: str = Header(default="")):
    _require_admin(x_admin_key)

    pro_id = (pro_id or "").strip()
    area_ids = _normalize_list(req.values)

    con = db()
    try:
        cur = con.cursor()
        if not _pro_exists(con, pro_id):
            raise HTTPException(status_code=404, detail="Pro not found")

        _assert_areas_exist(con, area_ids)

        # replace-all
        cur.execute("DELETE FROM circles_pro_areas WHERE pro_id=?", (pro_id,))
        if area_ids:
            cur.executemany(
                "INSERT OR IGNORE INTO circles_pro_areas(pro_id, area_id) VALUES(?,?)",
                [(pro_id, a) for a in area_ids],
            )

        cur.execute("UPDATE circles_pros SET updated_at=? WHERE id=?", (now_ts(), pro_id))
        con.commit()

        meta = _load_pro_meta(con, [pro_id]).get(pro_id, {"categories": [], "areaIds": []})
        return {"ok": True, "pro_id": pro_id, "areaIds": meta["areaIds"]}
    finally:
        con.close()


@router.delete("/admin/pros/{pro_id}")
def admin_delete_pro(pro_id: str, x_admin_key: str = Header(default="")):
    _require_admin(x_admin_key)

    pro_id = (pro_id or "").strip()

    con = db()
    try:
        cur = con.cursor()
        cur.execute("SELECT 1 FROM circles_pros WHERE id=?", (pro_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Pro not found")

        # soft delete
        cur.execute(
            "UPDATE circles_pros SET is_active=0, updated_at=? WHERE id=?",
            (now_ts(), pro_id),
        )
        con.commit()
        return {"ok": True, "deleted": True, "pro_id": pro_id}
    finally:
        con.close()
