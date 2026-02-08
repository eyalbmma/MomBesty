import sqlite3
import time
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

DB_PATH = "/data/rag.db"

router = APIRouter(prefix="/content", tags=["content"])


# ---------- DB helpers ----------
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def now_ts() -> int:
    return int(time.time())


# ---------- Models ----------
class TopicCreate(BaseModel):
    title: str
    description: Optional[str] = None
    order_index: int = 0


class TopicOut(BaseModel):
    id: int
    title: str
    description: Optional[str]
    order_index: int
    created_at: int
    default_question: Optional[str]  # ✅ חדש

class ArticleCreate(BaseModel):
    topic_id: int
    title: str
    summary: Optional[str] = None
    content: str
    source: Optional[str] = None  # למשל "לאומית", "משרד הבריאות"
    order_index: int = 0


class ArticleOut(BaseModel):
    id: int
    topic_id: int
    title: str
    summary: Optional[str]
    content: str
    source: Optional[str]
    order_index: int
    created_at: int


# ---------- Endpoints ----------
@router.get("/topics", response_model=List[TopicOut])
def list_topics():
    con = db()
    cur = con.cursor()
    cur.execute("""
        SELECT id, title, description, order_index, created_at,default_question
        FROM content_topics
        ORDER BY order_index ASC, id ASC
    """)
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


@router.post("/topics", response_model=TopicOut)
def create_topic(payload: TopicCreate):
    con = db()
    cur = con.cursor()
    ts = now_ts()
    cur.execute("""
        INSERT INTO content_topics (title, description, order_index, created_at)
        VALUES (?, ?, ?, ?)
    """, (payload.title.strip(), (payload.description or None), payload.order_index, ts))
    con.commit()
    topic_id = cur.lastrowid

    cur.execute("""
        SELECT id, title, description, order_index, created_at
        FROM content_topics
        WHERE id = ?
    """, (topic_id,))
    row = cur.fetchone()
    con.close()
    return dict(row)


@router.get("/topics/{topic_id}/articles", response_model=List[ArticleOut])
def list_articles(
    topic_id: int,
    limit: int = Query(20, ge=1, le=50),
    before: Optional[int] = Query(None, description="created_at timestamp pagination"),
):
    con = db()
    cur = con.cursor()

    # topic exists?
    cur.execute("SELECT 1 FROM content_topics WHERE id = ?", (topic_id,))
    if not cur.fetchone():
        con.close()
        raise HTTPException(status_code=404, detail="Topic not found")

    if before is None:
        cur.execute("""
            SELECT id, topic_id, title, summary, content, source, order_index, created_at
            FROM content_articles
            WHERE topic_id = ?
            ORDER BY order_index ASC, created_at DESC, id DESC
            LIMIT ?
        """, (topic_id, limit))
    else:
        cur.execute("""
            SELECT id, topic_id, title, summary, content, source, order_index, created_at
            FROM content_articles
            WHERE topic_id = ? AND created_at < ?
            ORDER BY order_index ASC, created_at DESC, id DESC
            LIMIT ?
        """, (topic_id, before, limit))

    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


@router.get("/articles/{article_id}", response_model=ArticleOut)
def get_article(article_id: int):
    con = db()
    cur = con.cursor()
    cur.execute("""
        SELECT id, topic_id, title, summary, content, source, order_index, created_at
        FROM content_articles
        WHERE id = ?
    """, (article_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="Article not found")
    return dict(row)


@router.post("/articles", response_model=ArticleOut)
def create_article(payload: ArticleCreate):
    con = db()
    cur = con.cursor()

    cur.execute("SELECT 1 FROM content_topics WHERE id = ?", (payload.topic_id,))
    if not cur.fetchone():
        con.close()
        raise HTTPException(status_code=404, detail="Topic not found")

    ts = now_ts()
    cur.execute("""
        INSERT INTO content_articles
        (topic_id, title, summary, content, source, order_index, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        payload.topic_id,
        payload.title.strip(),
        payload.summary,
        payload.content.strip(),
        payload.source,
        payload.order_index,
        ts
    ))
    con.commit()
    article_id = cur.lastrowid

    cur.execute("""
        SELECT id, topic_id, title, summary, content, source, order_index, created_at
        FROM content_articles
        WHERE id = ?
    """, (article_id,))
    row = cur.fetchone()
    con.close()
    return dict(row)


@router.get("/search", response_model=List[ArticleOut])
def search_articles(q: str = Query(..., min_length=2), limit: int = Query(20, ge=1, le=50)):
    # חיפוש פשוט ב-SQLite (LIKE). מספיק ל-MVP.
    con = db()
    cur = con.cursor()
    needle = f"%{q.strip()}%"
    cur.execute("""
        SELECT id, topic_id, title, summary, content, source, order_index, created_at
        FROM content_articles
        WHERE title LIKE ? OR summary LIKE ? OR content LIKE ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
    """, (needle, needle, needle, limit))
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]
