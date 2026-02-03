import time
import sqlite3
from push_utils import send_expo_push

from typing import Optional, List, Any

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

DB_PATH = "rag.db"

router = APIRouter(prefix="/forum", tags=["forum"])
FORUM_API_VERSION = "render-check-2026-02-03-pushdebug"

@router.get("/__version")
def forum_version():
    return {"ok": True, "version": FORUM_API_VERSION}

# ---- Forum rate limits (seconds) ----
POST_COOLDOWN_SEC = 60
COMMENT_COOLDOWN_SEC = 20
REPORT_COOLDOWN_SEC = 30
EMPATHY_COOLDOWN_SEC = 2

MAX_POST_LEN = 800
MIN_POST_LEN = 20
MAX_COMMENT_LEN = 400
MIN_COMMENT_LEN = 5




# ---------------- DB helpers ----------------
def db_connect():
    con = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def now_ts() -> int:
    return int(time.time())


# ---------------- Request Models ----------------
class CreatePostReq(BaseModel):
    user_id: str
    content: str
    is_anonymous: bool = True
    display_name: Optional[str] = None


class CreateCommentReq(BaseModel):
    user_id: str
    content: str
    display_name: Optional[str] = None


class ReportReq(BaseModel):
    reporter_user_id: str
    target_type: str  # 'post' / 'comment'
    target_id: int
    reason: Optional[str] = None


class EmpathyReq(BaseModel):
    user_id: str


class PushTokenReq(BaseModel):
    user_id: str
    token: str  # ExponentPushToken[...]
    platform: Optional[str] = None  # 'android' / 'ios'


class MarkReadReq(BaseModel):
    user_id: str
    notification_ids: Optional[List[int]] = None
    post_id: Optional[int] = None
    all: bool = False


# ---------------- Utilities ----------------






# ---------------- Rate limit helper ----------------
def check_rate_limit(user_id: str, action: str, cooldown_sec: int) -> bool:
    user_id = (user_id or "").strip()
    if not user_id:
        return False

    now = now_ts()
    con = db_connect()
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT last_ts FROM forum_rate_limits WHERE user_id=? AND action=?",
            (user_id, action),
        )
        row = cur.fetchone()

        if row and now - int(row["last_ts"]) < cooldown_sec:
            return False

        cur.execute(
            "INSERT OR REPLACE INTO forum_rate_limits(user_id, action, last_ts) VALUES(?,?,?)",
            (user_id, action, now),
        )
        con.commit()
        return True
    finally:
        con.close()


# ---------------- Push tokens ----------------
@router.post("/push-tokens")
def upsert_push_token(req: PushTokenReq):
    user_id = (req.user_id or "").strip()
    token = (req.token or "").strip()
    platform = (req.platform or "").strip() or None
    now = now_ts()

    if not user_id:
        return {"ok": False, "error": "user_id חסר"}
    if not token:
        return {"ok": False, "error": "token חסר"}
    if not _is_expo_token(token):
        return {"ok": False, "error": "token לא נראה כמו Expo Push Token"}

    con = db_connect()
    try:
        cur = con.cursor()

        # נשמור UNIQUE(user_id, token) כך שלא יהיו כפילויות
        cur.execute(
            """
            INSERT OR IGNORE INTO forum_push_tokens(user_id, token, platform, created_at, last_seen_at)
            VALUES(?,?,?,?,?)
            """,
            (user_id, token, platform, now, now),
        )

        # אם כבר קיים - נעדכן last_seen_at / platform
        cur.execute(
            """
            UPDATE forum_push_tokens
            SET last_seen_at = ?, platform = COALESCE(?, platform)
            WHERE user_id = ? AND token = ?
            """,
            (now, platform, user_id, token),
        )

        con.commit()
        return {"ok": True}
    finally:
        con.close()


def _get_user_push_tokens(user_id: str) -> List[str]:
    con = db_connect()
    try:
        cur = con.cursor()
        cur.execute("SELECT token FROM forum_push_tokens WHERE user_id=?", (user_id,))
        rows = cur.fetchall()
        return [r["token"] for r in rows if r and r["token"]]
    finally:
        con.close()




# ---------------- Notifications ----------------
@router.get("/notifications/unread-count")
def unread_count(user_id: str):
    user_id = (user_id or "").strip()
    if not user_id:
        return {"ok": False, "error": "user_id חסר"}

    con = db_connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT COUNT(1) AS cnt
            FROM forum_notifications
            WHERE user_id=? AND read_at IS NULL
            """,
            (user_id,),
        )
        row = cur.fetchone()
        return {"ok": True, "unread": int(row["cnt"] or 0) if row else 0}
    finally:
        con.close()


@router.get("/notifications")
def list_notifications(user_id: str, limit: int = 30):
    user_id = (user_id or "").strip()
    if not user_id:
        return {"ok": False, "error": "user_id חסר"}

    limit = min(max(int(limit), 1), 50)

    con = db_connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, user_id, type, post_id, comment_id, from_user_id, created_at, read_at
            FROM forum_notifications
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()

        out = []
        for r in rows:
            out.append(
                {
                    "id": int(r["id"]),
                    "user_id": r["user_id"],
                    "type": r["type"],
                    "post_id": int(r["post_id"]) if r["post_id"] is not None else None,
                    "comment_id": int(r["comment_id"]) if r["comment_id"] is not None else None,
                    "from_user_id": r["from_user_id"],
                    "created_at": int(r["created_at"]),
                    "read_at": int(r["read_at"]) if r["read_at"] is not None else None,
                }
            )

        return {"ok": True, "notifications": out}
    finally:
        con.close()


@router.post("/notifications/mark-read")
def mark_read(req: MarkReadReq):
    user_id = (req.user_id or "").strip()
    if not user_id:
        return {"ok": False, "error": "user_id חסר"}

    now = now_ts()
    con = db_connect()
    try:
        cur = con.cursor()
        updated = 0

        if req.all:
            cur.execute(
                """
                UPDATE forum_notifications
                SET read_at = ?
                WHERE user_id=? AND read_at IS NULL
                """,
                (now, user_id),
            )
            updated = cur.rowcount

        elif req.post_id is not None:
            cur.execute(
                """
                UPDATE forum_notifications
                SET read_at = ?
                WHERE user_id=? AND post_id=? AND read_at IS NULL
                """,
                (now, user_id, int(req.post_id)),
            )
            updated = cur.rowcount

        elif req.notification_ids:
            ids = [int(x) for x in req.notification_ids if x is not None]
            if ids:
                placeholders = ",".join(["?"] * len(ids))
                cur.execute(
                    f"""
                    UPDATE forum_notifications
                    SET read_at = ?
                    WHERE user_id=? AND read_at IS NULL AND id IN ({placeholders})
                    """,
                    (now, user_id, *ids),
                )
                updated = cur.rowcount

        con.commit()
        return {"ok": True, "updated": int(updated)}
    finally:
        con.close()


def _create_notification_for_comment(owner_user_id: str, from_user_id: str, post_id: int, comment_id: int):
    now = now_ts()
    con = db_connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO forum_notifications(user_id, type, post_id, comment_id, from_user_id, created_at, read_at)
            VALUES(?,?,?,?,?,?,NULL)
            """,
            (owner_user_id, "comment_on_my_post", int(post_id), int(comment_id), from_user_id, now),
        )
        nid = cur.lastrowid
        con.commit()
        return nid, now
    finally:
        con.close()


def _get_post_owner(post_id: int) -> Optional[str]:
    con = db_connect()
    try:
        cur = con.cursor()
        cur.execute("SELECT user_id FROM forum_posts WHERE id=? AND status='active'", (int(post_id),))
        row = cur.fetchone()
        return row["user_id"] if row else None
    finally:
        con.close()


# ---------------- Posts ----------------
@router.post("/posts")
def create_post(req: CreatePostReq):
    user_id = (req.user_id or "").strip()
    content = (req.content or "").strip()
    display_name = (req.display_name or "").strip() or None

    if not user_id:
        return {"ok": False, "error": "user_id חסר"}

    if len(content) < MIN_POST_LEN or len(content) > MAX_POST_LEN:
        return {"ok": False, "error": "אורך פוסט לא תקין"}

    if not check_rate_limit(user_id, "post", POST_COOLDOWN_SEC):
        return {"ok": False, "error": "נא להמתין לפני פרסום נוסף"}

    now = now_ts()

    con = db_connect()
    try:
        cur = con.cursor()
        # ✅ חשוב: בלי comments_count/status כדי לא לשבור DB
        cur.execute(
            """
            INSERT INTO forum_posts(user_id, content, is_anonymous, display_name, created_at)
            VALUES(?,?,?,?,?)
            """,
            (user_id, content, int(req.is_anonymous), display_name, now),
        )
        post_id = cur.lastrowid
        con.commit()

        return {
            "ok": True,
            "post": {
                "id": int(post_id),
                "user_id": user_id,
                "display_name": display_name,
                "is_anonymous": bool(req.is_anonymous),
                "content": content,
                "created_at": now,
                "empathy_count": 0,
                "comments_count": 0,
            },
        }
    finally:
        con.close()


@router.get("/posts")
def get_posts(
    limit: int = 20,
    before: Optional[int] = None,
    user_id: Optional[str] = None,
    q: Optional[str] = None,
):
    limit = min(max(int(limit), 1), 50)

    con = db_connect()
    try:
        cur = con.cursor()

        where_parts = ["p.status='active'"]
        params: List[Any] = []

        if before is not None:
            where_parts.append("p.created_at < ?")
            params.append(int(before))

        if user_id:
            where_parts.append("p.user_id = ?")
            params.append((user_id or "").strip())

        if q:
            q_clean = (q or "").strip()
            if q_clean:
                where_parts.append("p.content LIKE ?")
                params.append(f"%{q_clean}%")

        where_sql = " AND ".join(where_parts)

        # ✅ comments_count מחושב ולא שדה בטבלה
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
              (SELECT COUNT(1) 
               FROM forum_comments c 
               WHERE c.post_id = p.id AND c.status='active') AS comments_count
            FROM forum_posts p
            WHERE {where_sql}
            ORDER BY p.created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        )

        rows = cur.fetchall()
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
    finally:
        con.close()


# ✅✅✅ ENDPOINT שחסר לאפליקציה: GET /forum/posts/{post_id}
@router.get("/posts/{post_id}")
def get_post_by_id(post_id: int):
    con = db_connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT
              p.id,
              p.user_id,
              p.display_name,
              p.is_anonymous,
              p.content,
              p.empathy_count,
              p.created_at,
              (SELECT COUNT(1) 
               FROM forum_comments c 
               WHERE c.post_id = p.id AND c.status='active') AS comments_count
            FROM forum_posts p
            WHERE p.id=? AND p.status='active'
            """,
            (int(post_id),),
        )
        r = cur.fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="post not found")

        return {
            "ok": True,
            "post": {
                "id": int(r["id"]),
                "user_id": r["user_id"],
                "display_name": r["display_name"],
                "is_anonymous": bool(int(r["is_anonymous"])),
                "content": r["content"],
                "empathy_count": int(r["empathy_count"] or 0),
                "comments_count": int(r["comments_count"] or 0),
                "created_at": int(r["created_at"]),
            },
        }
    finally:
        con.close()


# ---------------- Comments ----------------
@router.post("/posts/{post_id}/comments")
def add_comment(post_id: int, req: CreateCommentReq):
    user_id = (req.user_id or "").strip()
    content = (req.content or "").strip()
    display_name = (req.display_name or "").strip() or None

    if not user_id:
        return {"ok": False, "error": "user_id חסר"}

    if len(content) < MIN_COMMENT_LEN or len(content) > MAX_COMMENT_LEN:
        return {"ok": False, "error": "אורך תגובה לא תקין"}

    if not check_rate_limit(user_id, "comment", COMMENT_COOLDOWN_SEC):
        return {"ok": False, "error": "נא להמתין לפני תגובה נוספת"}

    now = now_ts()

    con = db_connect()
    try:
        cur = con.cursor()

        # ודא שהפוסט קיים ופעיל
        cur.execute("SELECT id FROM forum_posts WHERE id=? AND status='active'", (int(post_id),))
        post_row = cur.fetchone()
        if not post_row:
            return {"ok": False, "error": "פוסט לא נמצא / לא פעיל"}

        cur.execute(
            """
            INSERT INTO forum_comments(post_id, user_id, content, display_name, created_at)
            VALUES(?,?,?,?,?)
            """,
            (int(post_id), user_id, content, display_name, now),
        )
        comment_id = cur.lastrowid

        con.commit()
    finally:
        con.close()

    # --- Notifications + Push (best-effort) ---
        owner_user_id = _get_post_owner(int(post_id))
    if owner_user_id and owner_user_id != user_id:
        print(
            f"[PUSH_DEBUG] post_id={post_id} owner_user_id={owner_user_id} commenter_user_id={user_id}"
        )

        # 1) צור Notification ב-DB
        nid, ts = _create_notification_for_comment(
            owner_user_id=owner_user_id,
            from_user_id=user_id,
            post_id=int(post_id),
            comment_id=int(comment_id),
        )
        print(f"[PUSH_DEBUG] created_notification_id={nid} created_at={ts}")

        # 2) שלוף tokens של יוצר הפוסט
        tokens = _get_user_push_tokens(owner_user_id)
        print(f"[PUSH_DEBUG] owner_tokens_count={len(tokens)} tokens={tokens}")

        # 3) שלח push (גם אם אין tokens – נדע בלוג)
        resp = send_expo_push(
        tokens=tokens,
        title="תגובה חדשה לפוסט שלך",
        body=content,
        data={"screen": "ForumPost", "postId": int(post_id), "commentId": int(comment_id)},
        )
        print(f"[PUSH_DEBUG] expo_resp={resp}")
    else:
        print(
            f"[PUSH_DEBUG] skip_push post_id={post_id} owner_user_id={owner_user_id} commenter_user_id={user_id}"
        )

    return {
        "ok": True,
        "comment": {
            "id": int(comment_id),
            "post_id": int(post_id),
            "user_id": user_id,
            "display_name": display_name,
            "content": content,
            "created_at": now,
        },
    }



@router.get("/posts/{post_id}/comments")
def get_comments(post_id: int, limit: int = 50, after: Optional[int] = None):
    limit = min(max(int(limit), 1), 50)

    con = db_connect()
    try:
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
                (int(post_id), limit),
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
                (int(post_id), int(after), limit),
            )

        rows = cur.fetchall()
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
    finally:
        con.close()


# ---------------- Reports ----------------
@router.post("/reports")
def report(req: ReportReq):
    if req.target_type not in ("post", "comment"):
        return {"ok": False, "error": "target_type לא תקין"}

    if not check_rate_limit(req.reporter_user_id, "report", REPORT_COOLDOWN_SEC):
        return {"ok": False, "error": "נא להמתין לפני דיווח נוסף"}

    con = db_connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO forum_reports(target_type, target_id, reporter_user_id, reason, created_at)
            VALUES(?,?,?,?,?)
            """,
            (req.target_type, int(req.target_id), req.reporter_user_id, req.reason, now_ts()),
        )
        con.commit()
        return {"ok": True}
    finally:
        con.close()


# ---------------- Empathy ----------------
@router.post("/posts/{post_id}/empathy")
def add_empathy(post_id: int, req: EmpathyReq):
    user_id = (req.user_id or "").strip()
    if not user_id:
        return {"ok": False, "error": "user_id חסר"}

    if not check_rate_limit(user_id, "empathy", EMPATHY_COOLDOWN_SEC):
        return {"ok": False, "error": "נא להמתין רגע"}

    now = now_ts()
    con = db_connect()
    try:
        cur = con.cursor()

        # ודא שהפוסט פעיל
        cur.execute("SELECT empathy_count FROM forum_posts WHERE id=? AND status='active'", (int(post_id),))
        exists = cur.fetchone()
        if not exists:
            return {"ok": False, "error": "פוסט לא נמצא / לא פעיל"}

        try:
            cur.execute(
                "INSERT INTO forum_empathy(post_id, user_id, created_at) VALUES(?,?,?)",
                (int(post_id), user_id, now),
            )
        except sqlite3.IntegrityError:
            cur.execute("SELECT empathy_count FROM forum_posts WHERE id=?", (int(post_id),))
            row = cur.fetchone()
            return {"ok": True, "already": True, "empathy_count": int(row["empathy_count"]) if row else None}

        cur.execute(
            """
            UPDATE forum_posts
            SET empathy_count = empathy_count + 1
            WHERE id=? AND status='active'
            """,
            (int(post_id),),
        )

        con.commit()

        cur.execute("SELECT empathy_count FROM forum_posts WHERE id=?", (int(post_id),))
        row = cur.fetchone()
        return {"ok": True, "already": False, "empathy_count": int(row["empathy_count"]) if row else None}
    finally:
        con.close()


# ---------------- Deletes (soft) ----------------
@router.delete("/posts/{post_id}")
def delete_post(post_id: int, user_id: str):
    user_id = (user_id or "").strip()
    if not user_id:
        return {"ok": False, "error": "user_id חסר"}

    con = db_connect()
    try:
        cur = con.cursor()

        cur.execute(
            """
            UPDATE forum_posts
            SET status='deleted'
            WHERE id=? AND user_id=? AND status='active'
            """,
            (int(post_id), user_id),
        )

        if cur.rowcount == 0:
            return {"ok": False, "error": "לא נמצא / לא שלך / כבר נמחק"}

        cur.execute(
            "UPDATE forum_comments SET status='deleted' WHERE post_id=? AND status='active'",
            (int(post_id),),
        )

        con.commit()
        return {"ok": True}
    finally:
        con.close()


@router.delete("/comments/{comment_id}")
def delete_comment(comment_id: int, user_id: str):
    user_id = (user_id or "").strip()
    if not user_id:
        return {"ok": False, "error": "user_id חסר"}

    con = db_connect()
    try:
        cur = con.cursor()

        cur.execute(
            """
            UPDATE forum_comments
            SET status='deleted'
            WHERE id=? AND user_id=? AND status='active'
            """,
            (int(comment_id), user_id),
        )

        if cur.rowcount == 0:
            return {"ok": False, "error": "לא נמצא / לא שלך / כבר נמחק"}

        con.commit()
        return {"ok": True}
    finally:
        con.close()



@router.get("/debug/push-tokens")
def debug_push_tokens(user_id: str):
    return {"ok": True, "user_id": user_id, "tokens": _get_user_push_tokens(user_id)}
