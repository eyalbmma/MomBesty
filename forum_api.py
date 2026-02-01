import time
import sqlite3
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

DB_PATH = "rag.db"

router = APIRouter(prefix="/forum", tags=["forum"])

# ---- Forum rate limits (seconds) ----
POST_COOLDOWN_SEC = 60
COMMENT_COOLDOWN_SEC = 20
REPORT_COOLDOWN_SEC = 30
EMPATHY_COOLDOWN_SEC = 2

MAX_POST_LEN = 800
MIN_POST_LEN = 20
MAX_COMMENT_LEN = 400
MIN_COMMENT_LEN = 5


def db_connect():
    con = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


# --------- Request Models ---------
class CreatePostReq(BaseModel):
    user_id: str
    content: str
    is_anonymous: bool = True
    display_name: Optional[str] = None  # ✅ כינוי


class CreateCommentReq(BaseModel):
    user_id: str
    content: str
    display_name: Optional[str] = None  # ✅ כינוי


class ReportReq(BaseModel):
    reporter_user_id: str
    target_type: str  # 'post' / 'comment'
    target_id: int
    reason: Optional[str] = None


class EmpathyReq(BaseModel):
    user_id: str


# --------- Rate limit helper ---------
def check_rate_limit(user_id: str, action: str, cooldown_sec: int) -> bool:
    now = int(time.time())
    con = db_connect()
    cur = con.cursor()

    cur.execute(
        "SELECT last_ts FROM forum_rate_limits WHERE user_id=? AND action=?",
        (user_id, action),
    )
    row = cur.fetchone()

    if row and now - int(row["last_ts"]) < cooldown_sec:
        con.close()
        return False

    cur.execute(
        "INSERT OR REPLACE INTO forum_rate_limits(user_id, action, last_ts) VALUES(?,?,?)",
        (user_id, action, now),
    )
    con.commit()
    con.close()
    return True


# --------- Posts ---------
@router.post("/posts")
def create_post(req: CreatePostReq):
    content = (req.content or "").strip()
    display_name = (req.display_name or "").strip() or None

    if len(content) < MIN_POST_LEN or len(content) > MAX_POST_LEN:
        return {"ok": False, "error": "אורך פוסט לא תקין"}

    if not check_rate_limit(req.user_id, "post", POST_COOLDOWN_SEC):
        return {"ok": False, "error": "נא להמתין לפני פרסום נוסף"}

    now = int(time.time())
    con = db_connect()
    cur = con.cursor()

    cur.execute(
        """
        INSERT INTO forum_posts(user_id, content, is_anonymous, display_name, created_at)
        VALUES(?,?,?,?,?)
        """,
        (req.user_id, content, int(req.is_anonymous), display_name, now),
    )
    post_id = cur.lastrowid
    con.commit()
    con.close()

    return {
        "ok": True,
        "post": {
            "id": post_id,
            "user_id": req.user_id,
            "display_name": display_name,
            "is_anonymous": bool(req.is_anonymous),
            "content": content,
            "created_at": now,
            "empathy_count": 0,
            "comments_count": 0,
        },
    }


@router.get("/posts")
def get_posts(
    limit: int = 20,
    before: Optional[int] = None,
    user_id: Optional[str] = None,  # ✅ חדש: רק הפוסטים שלי
    q: Optional[str] = None,        # ✅ חדש: חיפוש טקסט
):
    # hard cap
    limit = min(max(int(limit), 1), 50)

    con = db_connect()
    cur = con.cursor()

    where_parts = ["p.status='active'"]
    params = []

    if before is not None:
        where_parts.append("p.created_at < ?")
        params.append(int(before))

    if user_id:
        where_parts.append("p.user_id = ?")
        params.append(user_id)

    if q:
        q_clean = (q or "").strip()
        if q_clean:
            where_parts.append("p.content LIKE ?")
            params.append(f"%{q_clean}%")

    where_sql = " AND ".join(where_parts)

    cur.execute(
        f"""
        SELECT
          p.id,
          p.user_id,
          p.display_name,
          p.is_anonymous,
          p.content,
          p.empathy_count,
          p.created_at,
          (SELECT COUNT(1) FROM forum_comments c WHERE c.post_id = p.id AND c.status='active') AS comments_count
        FROM forum_posts p
        WHERE {where_sql}
        ORDER BY p.created_at DESC
        LIMIT ?
        """,
        (*params, limit),
    )

    rows = cur.fetchall()
    con.close()

    next_before = int(rows[-1]["created_at"]) if rows else None

    posts = []
    for r in rows:
        posts.append(
            {
                "id": int(r["id"]),
                "user_id": r["user_id"],
                "display_name": r["display_name"],
                "is_anonymous": bool(int(r["is_anonymous"])),
                "content": r["content"],
                "empathy_count": int(r["empathy_count"] or 0),
                "created_at": int(r["created_at"]),
                "comments_count": int(r["comments_count"] or 0),
            }
        )

    return {"posts": posts, "next_before": next_before}


# --------- Comments ---------
@router.post("/posts/{post_id}/comments")
def add_comment(post_id: int, req: CreateCommentReq):
    content = (req.content or "").strip()
    display_name = (req.display_name or "").strip() or None

    if len(content) < MIN_COMMENT_LEN or len(content) > MAX_COMMENT_LEN:
        return {"ok": False, "error": "אורך תגובה לא תקין"}

    if not check_rate_limit(req.user_id, "comment", COMMENT_COOLDOWN_SEC):
        return {"ok": False, "error": "נא להמתין לפני תגובה נוספת"}

    now = int(time.time())
    con = db_connect()
    cur = con.cursor()

    cur.execute(
        """
        INSERT INTO forum_comments(post_id, user_id, content, display_name, created_at)
        VALUES(?,?,?,?,?)
        """,
        (post_id, req.user_id, content, display_name, now),
    )
    comment_id = cur.lastrowid
    con.commit()
    con.close()

    return {
        "ok": True,
        "comment": {
            "id": comment_id,
            "post_id": post_id,
            "user_id": req.user_id,
            "display_name": display_name,
            "content": content,
            "created_at": now,
        },
    }


@router.get("/posts/{post_id}/comments")
def get_comments(
    post_id: int,
    limit: int = 50,               # ✅ חדש: כמה להחזיר בכל פעם (ברירת מחדל גבוהה כדי לא לשבור)
    after: Optional[int] = None,   # ✅ חדש: פג'ינציה קדימה (created_at > after)
):
    limit = min(max(int(limit), 1), 50)

    con = db_connect()
    cur = con.cursor()

    if after is None:
        cur.execute(
            """
            SELECT id, post_id, user_id, display_name, content, created_at
            FROM forum_comments
            WHERE post_id=? AND status='active'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (post_id, limit),
        )
    else:
        cur.execute(
            """
            SELECT id, post_id, user_id, display_name, content, created_at
            FROM forum_comments
            WHERE post_id=? AND status='active' AND created_at > ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (post_id, int(after), limit),
        )

    rows = cur.fetchall()
    con.close()

    next_after = int(rows[-1]["created_at"]) if rows else None

    return {
        "comments": [
            {
                "id": int(r["id"]),
                "post_id": int(r["post_id"]),
                "user_id": r["user_id"],
                "display_name": r["display_name"],
                "content": r["content"],
                "created_at": int(r["created_at"]),
            }
            for r in rows
        ],
        "next_after": next_after,
    }


# --------- Reports ---------
@router.post("/reports")
def report(req: ReportReq):
    if req.target_type not in ("post", "comment"):
        return {"ok": False, "error": "target_type לא תקין"}

    if not check_rate_limit(req.reporter_user_id, "report", REPORT_COOLDOWN_SEC):
        return {"ok": False, "error": "נא להמתין לפני דיווח נוסף"}

    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO forum_reports(target_type, target_id, reporter_user_id, reason, created_at)
        VALUES(?,?,?,?,?)
        """,
        (req.target_type, req.target_id, req.reporter_user_id, req.reason, int(time.time())),
    )
    con.commit()
    con.close()
    return {"ok": True}


# --------- Empathy (Heart) ---------
@router.post("/posts/{post_id}/empathy")
def add_empathy(post_id: int, req: EmpathyReq):
    user_id = req.user_id

    if not check_rate_limit(user_id, "empathy", EMPATHY_COOLDOWN_SEC):
        return {"ok": False, "error": "נא להמתין רגע"}

    now = int(time.time())
    con = db_connect()
    cur = con.cursor()

    # 1) save unique empathy (PK/UNIQUE on (post_id,user_id) recommended)
    try:
        cur.execute(
            "INSERT INTO forum_empathy(post_id, user_id, created_at) VALUES(?,?,?)",
            (post_id, user_id, now),
        )
    except sqlite3.IntegrityError:
        cur.execute("SELECT empathy_count FROM forum_posts WHERE id=?", (post_id,))
        row = cur.fetchone()
        con.close()
        return {
            "ok": True,
            "already": True,
            "empathy_count": int(row["empathy_count"]) if row else None,
        }

    # 2) increment counter
    cur.execute(
        """
        UPDATE forum_posts
        SET empathy_count = empathy_count + 1
        WHERE id = ? AND status='active'
        """,
        (post_id,),
    )

    if cur.rowcount == 0:
        cur.execute("DELETE FROM forum_empathy WHERE post_id=? AND user_id=?", (post_id, user_id))
        con.commit()
        con.close()
        return {"ok": False, "error": "פוסט לא נמצא / לא פעיל"}

    con.commit()

    cur.execute("SELECT empathy_count FROM forum_posts WHERE id=?", (post_id,))
    row = cur.fetchone()
    con.close()
    return {"ok": True, "already": False, "empathy_count": int(row["empathy_count"]) if row else None}


# --------- Deletes (soft) ---------
@router.delete("/posts/{post_id}")
def delete_post(post_id: int, user_id: str):
    con = db_connect()
    cur = con.cursor()

    cur.execute(
        """
        UPDATE forum_posts
        SET status='deleted'
        WHERE id=? AND user_id=? AND status='active'
        """,
        (post_id, user_id),
    )

    if cur.rowcount == 0:
        con.close()
        return {"ok": False, "error": "לא נמצא / לא שלך / כבר נמחק"}

    cur.execute(
        "UPDATE forum_comments SET status='deleted' WHERE post_id=? AND status='active'",
        (post_id,),
    )

    con.commit()
    con.close()
    return {"ok": True}


@router.delete("/comments/{comment_id}")
def delete_comment(comment_id: int, user_id: str):
    con = db_connect()
    cur = con.cursor()

    cur.execute(
        """
        UPDATE forum_comments
        SET status='deleted'
        WHERE id=? AND user_id=? AND status='active'
        """,
        (comment_id, user_id),
    )

    if cur.rowcount == 0:
        con.close()
        return {"ok": False, "error": "לא נמצא / לא שלך / כבר נמחק"}

    con.commit()
    con.close()
    return {"ok": True}
