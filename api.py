import os
import json
import sqlite3
import time
import hashlib
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# =========================
# Config
# =========================
APP_VERSION = "render-test-11-followup-context"

DB_PATH = "rag.db"
TABLE = "rag_clean"
EMB_COL = "embedding"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ---- MVP controls ----
PILOT_MODE = True
TOP_SCORE_THRESHOLD = 0.80
LOW_SCORE_CUTOFF = 0.65
SOFT_CUTOFF = 0.45  # בפיילוט: מפעילים GPT גם בהתאמה בינונית

CACHE_TTL_SECONDS = 7 * 24 * 3600
PROMPT_VER = "v8-fastcache-followup"

# ---- Speed controls ----
HISTORY_LIMIT_DB = 6          # כמה הודעות לשלוף מה-DB
HISTORY_LIMIT_TO_GPT = 4      # כמה להכניס לפרומפט
CLIP_Q_CHARS = 220
CLIP_A_CHARS = 700
MAX_TOKENS = 220
TEMPERATURE = 0.3

# ---- Fast cache tuning ----
FAST_CACHE_SCORE_MIN = 0.72

# ---- Follow-up tuning ----
FOLLOWUP_HARD_FLOOR = 0.25    # מתחת לזה גם ב-follow-up לא נריץ GPT, אלא נשאל הבהרה חכמה
# ----------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

from forum_api import router as forum_router
from content_api import router as content_router
from tracker_api import router as tracker_router

app.include_router(forum_router)
app.include_router(content_router)
app.include_router(tracker_router)

# daily support runner (server-side)
from daily_support_sender import run_daily_support

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:19006",
        "http://127.0.0.1:19006",
        "http://192.168.1.144:8081",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/ask_final")
def options_ask_final():
    return Response(status_code=200)

@app.options("/ask")
def options_ask():
    return Response(status_code=200)

@app.options("/reload_rag")
def options_reload_rag():
    return Response(status_code=200)

# =========================
# Models
# =========================
class AskReq(BaseModel):
    question: str
    k: int = 3
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    topic: Optional[str] = None

# =========================
# Timing utilities
# =========================
def now_ms() -> int:
    return int(time.perf_counter() * 1000)

class Timer:
    def __init__(self):
        self.t0 = now_ms()
        self.marks: Dict[str, int] = {}

    def mark(self, name: str):
        self.marks[name] = now_ms() - self.t0

    def snapshot(self) -> Dict[str, int]:
        out = dict(self.marks)
        out["total_ms"] = now_ms() - self.t0
        return out

# =========================
# Text normalization
# =========================
def norm_q(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^0-9a-zA-Zא-ת\s]", "", s)
    return s

def tokenize_he(s: str) -> List[str]:
    s = norm_q(s)
    return [t for t in s.split(" ") if len(t) >= 3]

def clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"

# =========================
# Ensure rag.db exists (download from Drive if missing)
# =========================
def ensure_rag_db():
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 10_000_000:
        print("rag.db already exists, skipping download")
        return

    file_id = os.getenv("RAG_DB_FILE_ID", "").strip()
    if not file_id:
        raise RuntimeError("Missing RAG_DB_FILE_ID env var")

    print("Downloading rag.db from Google Drive via gdown...")
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, DB_PATH, quiet=False)
    print("Downloaded rag.db size:", os.path.getsize(DB_PATH))

# =========================
# SQLite helpers
# =========================
def db_connect():
    con = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def column_exists(table: str, col: str) -> bool:
    con = db_connect()
    cur = con.cursor()
    try:
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]
        return col in cols
    finally:
        con.close()

def ensure_tables():
    """
    יוצר טבלאות מערכת (cache / conversations) + טבלאות התראות לפורום +
    טבלאות daily support.
    חשוב במיוחד ב-Render כי rag.db יכול להגיע מ-Drive בלי הטבלאות החדשות.
    """
    con = db_connect()
    cur = con.cursor()

    # ---- LLM cache ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key TEXT PRIMARY KEY,
            answer TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
    """)

    # ---- Embedding cache ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS emb_cache (
            q_norm TEXT PRIMARY KEY,
            embedding TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
    """)

    # ---- Conversations ----
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            created_at INTEGER,
            updated_at INTEGER
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_conv_msgs_conv_time
        ON conversation_messages(conversation_id, created_at);
    """)

    # =========================================================
    # Forum notifications + push tokens
    # =========================================================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS forum_push_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            token TEXT NOT NULL,
            platform TEXT,
            created_at INTEGER NOT NULL,
            last_seen_at INTEGER NOT NULL,
            UNIQUE(user_id, token)
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_forum_push_tokens_user
        ON forum_push_tokens(user_id);
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS forum_notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            type TEXT NOT NULL,
            post_id INTEGER,
            comment_id INTEGER,
            from_user_id TEXT,
            created_at INTEGER NOT NULL,
            read_at INTEGER
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_forum_notifications_user_read
        ON forum_notifications(user_id, read_at, created_at);
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_forum_notifications_post
        ON forum_notifications(post_id, created_at);
    """)

    # =========================================================
    # Daily postpartum support
    # =========================================================
    cur.execute("""
        CREATE TABLE IF NOT EXISTS postpartum_profiles (
            user_id TEXT PRIMARY KEY,
            postpartum_start_ts INTEGER,
            opt_in INTEGER NOT NULL DEFAULT 1,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_support_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            interaction_hint TEXT,
            stage TEXT,
            is_active INTEGER NOT NULL DEFAULT 1
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_support_delivery_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            day_index INTEGER NOT NULL,
            message_id INTEGER,
            sent_at INTEGER NOT NULL,
            UNIQUE(user_id, day_index)
        );
    """)

    con.commit()
    con.close()

def upsert_conversation(user_id: str, conversation_id: str):
    ts = int(time.time())
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO conversations(id, user_id, created_at, updated_at)
        VALUES(?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
          user_id=excluded.user_id,
          updated_at=excluded.updated_at
        """,
        (conversation_id, user_id, ts, ts),
    )
    con.commit()
    con.close()

def add_message(conversation_id: str, role: str, content: str):
    ts = int(time.time())
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO conversation_messages(conversation_id, role, content, created_at)
        VALUES(?,?,?,?)
        """,
        (conversation_id, role, content, ts),
    )
    con.commit()
    con.close()

def get_recent_messages(conversation_id: str, limit: int = 12) -> List[Tuple[str, str]]:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT role, content
        FROM conversation_messages
        WHERE conversation_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (conversation_id, int(limit)),
    )
    rows = cur.fetchall()
    con.close()
    rows = list(reversed(rows))
    return [(r["role"] or "", r["content"] or "") for r in rows]

# =========================
# LLM answer cache
# =========================
def make_cache_key_conversational(
    question: str,
    top_ids: List[int],
    k: int,
    conversation_id: Optional[str],
    history: List[Tuple[str, str]],
) -> str:
    hist_compact = "|".join([f"{r}:{norm_q(c)[:60]}" for r, c in history[-HISTORY_LIMIT_TO_GPT:]])
    base = (
        f"{PROMPT_VER}|{CHAT_MODEL}|k={k}"
        f"|conv={conversation_id or ''}"
        f"|q={norm_q(question)}"
        f"|ids={','.join(map(str, top_ids))}"
        f"|hist={hist_compact}"
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def make_cache_key_fast(question: str, top_ids: List[int], k: int) -> str:
    base = (
        f"{PROMPT_VER}|FAST|{CHAT_MODEL}|k={k}"
        f"|q={norm_q(question)}"
        f"|ids={','.join(map(str, top_ids))}"
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def cache_get(key: str) -> Optional[str]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT answer, created_at FROM llm_cache WHERE cache_key=?", (key,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    ans, ts = row["answer"], int(row["created_at"])
    if time.time() - ts > CACHE_TTL_SECONDS:
        return None
    return ans

def cache_set(key: str, answer: str):
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO llm_cache(cache_key, answer, created_at) VALUES(?,?,?)",
        (key, answer, int(time.time()))
    )
    con.commit()
    con.close()

# =========================
# Embedding cache
# =========================
def emb_cache_get(q_norm: str) -> Optional[np.ndarray]:
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT embedding, created_at FROM emb_cache WHERE q_norm=?", (q_norm,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    emb_json, ts = row["embedding"], int(row["created_at"])
    if time.time() - ts > CACHE_TTL_SECONDS:
        return None
    try:
        return np.array(json.loads(emb_json), dtype=np.float32)
    except Exception:
        return None

def emb_cache_set(q_norm: str, vec: np.ndarray):
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO emb_cache(q_norm, embedding, created_at) VALUES(?,?,?)",
        (q_norm, json.dumps(vec.tolist(), ensure_ascii=False), int(time.time()))
    )
    con.commit()
    con.close()

def get_embedding(text: str, timing: Optional[Timer] = None) -> np.ndarray:
    qn = norm_q(text)
    v = emb_cache_get(qn)
    if v is not None:
        if timing:
            timing.mark("embed_cache_hit_ms")
        return v

    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[(text or "").replace("\n", " ").strip()]
    )
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    emb_cache_set(qn, v)
    if timing:
        timing.mark("embed_cache_miss_ms")
    return v

# =========================
# RAG in-memory
# =========================
RAG_ROWS: List[Tuple[int, str, str, str, str]] = []  # (id, question, answer, source, tags)
RAG_EMBS: Optional[np.ndarray] = None
RAG_READY: bool = False

def load_rag_to_memory():
    global RAG_ROWS, RAG_EMBS, RAG_READY

    has_is_active = column_exists(TABLE, "is_active")

    where = f"{EMB_COL} IS NOT NULL AND {EMB_COL} <> ''"
    if has_is_active:
        where += " AND (is_active IS NULL OR is_active = 1)"

    con = db_connect()
    cur = con.cursor()
    cur.execute(
        f"SELECT id, question, answer, source, tags, {EMB_COL} "
        f"FROM {TABLE} "
        f"WHERE {where}"
    )
    rows = cur.fetchall()
    con.close()

    local_rows: List[Tuple[int, str, str, str, str]] = []
    embs: List[np.ndarray] = []

    for rid, q, a, source, tags, emb_json in rows:
        try:
            ev = np.array(json.loads(emb_json), dtype=np.float32)
        except Exception:
            continue
        if ev.size == 0:
            continue
        embs.append(ev)
        local_rows.append((int(rid), q or "", a or "", source or "", tags or ""))

    if not embs:
        RAG_ROWS = []
        RAG_EMBS = None
        RAG_READY = False
        return

    embs_mat = np.vstack(embs).astype(np.float32)
    norms = np.linalg.norm(embs_mat, axis=1, keepdims=True) + 1e-12
    RAG_EMBS = (embs_mat / norms).astype(np.float32)

    RAG_ROWS = local_rows
    RAG_READY = True

def _keyword_boost(question: str, candidate_q: str, candidate_tags: str) -> float:
    qn = norm_q(question)
    cn = norm_q(candidate_q)

    if qn and (qn in cn or cn in qn):
        return 0.20

    qt = set(tokenize_he(question))
    ct = set(tokenize_he(candidate_q))
    if not qt or not ct:
        return 0.0

    reflect = len(qt.intersection(ct)) / max(1, len(qt))
    if reflect >= 0.6:
        return 0.10
    if reflect >= 0.4:
        return 0.05

    if "breast" in (candidate_tags or "").lower() and ("הנקה" in qn or "שד" in qn):
        return 0.05

    return 0.0

def retrieve_top_k(qv: np.ndarray, question: str, k: int):
    if (not RAG_READY) or (RAG_EMBS is None) or (len(RAG_ROWS) == 0):
        return []

    qv = qv.astype(np.float32)
    qv = qv / (np.linalg.norm(qv) + 1e-12)

    scores = RAG_EMBS @ qv
    k = max(1, min(int(k), scores.shape[0]))

    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]

    top = []
    for i in idx:
        rid, qq, aa, source, tags = RAG_ROWS[int(i)]
        base = float(scores[int(i)])
        boosted = min(0.999, base + _keyword_boost(question, qq, tags))
        top.append((boosted, rid, qq, aa, source, tags, base))
    return top

# =========================
# Topic-aware fallback
# =========================
def topic_fallback(question: str, top_matches: List[Tuple]) -> str:
    qn = norm_q(question)
    tags_blob = " ".join([(m[5] or "") for m in top_matches]).lower()

    if ("הנקה" in qn) or ("שד" in qn) or ("פטמה" in qn) or ("breast" in tags_blob) or ("breastfeeding" in tags_blob):
        return "כדי לדייק: בן/בת כמה התינוק, והאם יש כאב/סדקים או קושי בהיצמדות?"

    if ("שינה" in qn) or ("בכי" in qn) or ("אוכל" in qn) or ("האכלה" in qn):
        return "כדי לדייק: בן/בת כמה התינוק, ומה בדיוק קורה עכשיו ומה ניסית?"

    if ("דימום" in qn) or ("חום" in qn) or ("כאב" in qn) or ("תפרים" in qn):
        return "כדי לדייק: יש חום/דימום שמתחזק/כאב שמתגבר? ואיזה שבוע אחרי לידה את?"

    return "כדי לענות מדויק, תכתבי עוד משפט אחד של הקשר: על מי מדובר ומה הקושי?"

def topic_from_history(history: List[Tuple[str, str]]) -> str:
    blob = " ".join([norm_q(c) for _, c in history[-HISTORY_LIMIT_TO_GPT:]])
    if any(w in blob for w in ["הנקה", "שד", "פטמה", "סדק", "פטמות", "breast", "breastfeeding"]):
        return "breastfeeding"
    if any(w in blob for w in ["שינה", "התעורר", "לילה", "נרדם", "בכי", "בוכה"]):
        return "sleep"
    if any(w in blob for w in ["אוכל", "האכלה", "בקבוק", "שאיבה", "פורמולה", "ממל", "מ״ל"]):
        return "feeding"
    if any(w in blob for w in ["דימום", "תפרים", "כאב", "חום", "רחם"]):
        return "postpartum"
    return "generic"

def topic_fallback_followup(question: str, history: List[Tuple[str, str]]) -> str:
    t = topic_from_history(history)
    if t == "breastfeeding":
        return "זה נשמע בהמשך להנקה/פטמות. כדי לדייק: את מחפשת משחה לפטמות פצועות או משהו למניעה? (מילה אחת: פצועות/מניעה)"
    if t == "sleep":
        return "זה נשמע בהמשך לשינה. כדי לדייק: בן/בת כמה התינוק, והבעיה בעיקר בהירדמות או יקיצות תכופות?"
    if t == "feeding":
        return "זה נשמע בהמשך לאוכל/האכלה. כדי לדייק: בן/בת כמה התינוק, והאם מדובר בהנקה, שאיבה או בקבוק?"
    if t == "postpartum":
        return "זה נשמע בהמשך להתאוששות אחרי לידה. כדי לדייק: כמה זמן אחרי לידה את, ומה התסמין המרכזי עכשיו?"
    return "כדי לענות מדויק, תכתבי עוד משפט אחד של הקשר: על מי מדובר ומה הקושי?"

# =========================
# Fast-cache classifier
# =========================
def is_factual_question(question: str, top_score: float, history: List[Tuple[str, str]]) -> bool:
    if top_score < FAST_CACHE_SCORE_MIN:
        return False

    qn = norm_q(question)
    patterns = [
        "בן כמה", "בת כמה", "מתי", "באיזה גיל", "כמה זמן", "מתי מתחיל",
        "מה זה", "איך נקרא", "התפתחות", "שלב", "מתי תינוק"
    ]
    return any(p in qn for p in patterns)

# =========================
# Follow-up detector
# =========================
def is_followup_question(question: str, history: List[Tuple[str, str]]) -> bool:
    if not history:
        return False

    qn = norm_q(question)
    if not qn:
        return False

    if len(qn.split()) <= 5:
        return True

    starters = ["איזה", "מה", "איך", "כמה", "איפה", "זה", "כזו", "כזה", "אפשר"]
    return any(qn.startswith(s) for s in starters)

def build_augmented_question(question: str, history: List[Tuple[str, str]]) -> str:
    if not is_followup_question(question, history):
        return question

    prev_user = ""
    for role, content in reversed(history):
        if role == "user" and content:
            prev_user = content
            break

    if not prev_user:
        return question

    return f"הקשר: {clip(prev_user, 220)}\nשאלה: {question}"

# =========================
# Init
# =========================
ensure_rag_db()
ensure_tables()
load_rag_to_memory()

# =========================
# Endpoints
# =========================
@app.get("/")
def root():
    return {"ok": True, "version": APP_VERSION}

@app.get("/health")
def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "table": TABLE,
        "pilot_mode": PILOT_MODE,
        "top_score_threshold": TOP_SCORE_THRESHOLD,
        "low_score_cutoff": LOW_SCORE_CUTOFF,
        "soft_cutoff": SOFT_CUTOFF,
        "fast_cache_score_min": FAST_CACHE_SCORE_MIN,
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "rag_rows_loaded": len(RAG_ROWS),
        "rag_ready": RAG_READY,
    }

@app.get("/debug/tables")
def debug_tables():
    con = db_connect()
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    con.close()

    wanted = [
        "postpartum_profiles",
        "daily_support_messages",
        "daily_support_delivery_log",
    ]

    return {
        "ok": True,
        "has": {t: (t in tables) for t in wanted},
        "tables_count": len(tables),
    }

@app.post("/debug/daily-support")
def debug_daily_support(
    action: str,
    user_id: Optional[str] = None,
    dry_run: bool = True,
):
    """
    Single debug endpoint for daily support.
    Actions:
      - recent_token_users
      - seed_minimal_messages
      - create_test_user   (requires user_id)
      - preview            (dry_run enforced)
      - send_now           (set dry_run=false to actually send)
    """
    con = db_connect()
    cur = con.cursor()
    ts = int(time.time())

    try:
        if action == "recent_token_users":
            cur.execute(
                """
                SELECT user_id, COUNT(*) AS tokens
                FROM forum_push_tokens
                GROUP BY user_id
                ORDER BY MAX(last_seen_at) DESC, MAX(created_at) DESC
                LIMIT 20
                """
            )
            rows = cur.fetchall()
            return {
                "ok": True,
                "users": [{"user_id": r["user_id"], "tokens": int(r["tokens"])} for r in rows],
            }

        if action == "seed_minimal_messages":
            cur.execute("SELECT COUNT(*) AS c FROM daily_support_messages")
            c = int(cur.fetchone()["c"])
            if c == 0:
                cur.executemany(
                    """
                    INSERT INTO daily_support_messages (day_index, text, interaction_hint, stage, is_active)
                    VALUES (?, ?, ?, ?, 1)
                    """,
                    [
                        (1, "היום, תני לעצמך רגע אחד קטן של נשימה. את עושה המון.", "נשימה קצרה", "early"),
                        (2, "אם היום קשה—זה לא אומר שאת לא טובה. זה אומר שאת אנושית.", "חמלה עצמית", "early"),
                        (3, "נסי לבקש עזרה בדבר אחד קטן. לא חייבים לבד.", "לבקש עזרה", "early"),
                    ],
                )
                con.commit()
                return {"ok": True, "seeded": True, "count_added": 3}
            return {"ok": True, "seeded": False, "existing_count": c}

        if action == "create_test_user":
            if not user_id:
                raise HTTPException(status_code=400, detail="user_id is required for create_test_user")

            cur.execute(
                """
                INSERT INTO postpartum_profiles (user_id, postpartum_start_ts, opt_in, created_at, updated_at)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    postpartum_start_ts = excluded.postpartum_start_ts,
                    opt_in = 1,
                    updated_at = excluded.updated_at
                """,
                (user_id, ts - 86400, ts, ts),
            )
            con.commit()
            return {"ok": True, "user_id": user_id, "postpartum_day_index": 1}

        if action == "preview":
            return run_daily_support(con, dry_run=True)

        if action == "send_now":
            return run_daily_support(con, dry_run=bool(dry_run))

        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    finally:
        con.close()

@app.post("/reload_rag")
def reload_rag():
    load_rag_to_memory()
    return {"ok": True, "rag_rows_loaded": len(RAG_ROWS), "rag_ready": RAG_READY}

@app.post("/ask")
def ask(req: AskReq):
    t = Timer()

    q_for_retrieval = req.question
    qv = get_embedding(q_for_retrieval, timing=t)
    t.mark("embed_ms")

    top = retrieve_top_k(qv, q_for_retrieval, req.k)
    t.mark("retrieve_ms")

    return {
        "matches": [
            {
                "score": round(score, 4),
                "score_base": round(base, 4),
                "id": rid,
                "question": qq,
                "answer": aa,
                "source": source,
                "tags": tags,
            }
            for (score, rid, qq, aa, source, tags, base) in top
        ],
        "timing": t.snapshot(),
    }

@app.post("/ask_final")
def ask_final(req: AskReq):
    t = Timer()

    history: List[Tuple[str, str]] = []
    if req.user_id and req.conversation_id:
        upsert_conversation(req.user_id, req.conversation_id)

    if req.conversation_id:
        history = get_recent_messages(req.conversation_id, limit=HISTORY_LIMIT_DB)
        t.mark("history_ms")

    if req.conversation_id:
        add_message(req.conversation_id, "user", req.question)

    followup = is_followup_question(req.question, history)

    q_for_retrieval = build_augmented_question(req.question, history)
    qv = get_embedding(q_for_retrieval, timing=t)
    t.mark("embed_ms")

    top = retrieve_top_k(qv, q_for_retrieval, req.k)
    t.mark("retrieve_ms")

    if not top:
        ans = topic_fallback_followup(req.question, history) if followup else "לא מצאתי מידע רלוונטי במאגר כרגע. אפשר לנסח מחדש במשפט אחד?"
        return {
            "answer": ans,
            "cached": False,
            "used_gpt": False,
            "cache_type": "none",
            "followup": followup,
            "top_matches": [],
            "timing": t.snapshot(),
        }

    top_score = float(top[0][0])

    if top_score < FOLLOWUP_HARD_FLOOR:
        ans = topic_fallback_followup(req.question, history) if followup else topic_fallback(req.question, top)
        return {
            "answer": ans,
            "cached": False,
            "used_gpt": False,
            "cache_type": "none",
            "followup": followup,
            "top_score": round(top_score, 4),
            "top_matches": [{"score": round(s, 4), "score_base": round(base, 4), "id": rid, "tags": tags}
                            for (s, rid, *_r, _source, tags, base) in top],
            "timing": t.snapshot(),
        }

    if top_score < SOFT_CUTOFF and not followup:
        return {
            "answer": topic_fallback(req.question, top),
            "cached": False,
            "used_gpt": False,
            "cache_type": "none",
            "followup": followup,
            "top_score": round(top_score, 4),
            "top_matches": [{"score": round(s, 4), "score_base": round(base, 4), "id": rid, "tags": tags}
                            for (s, rid, *_r, _source, tags, base) in top],
            "timing": t.snapshot(),
        }

    if (not PILOT_MODE) and (top_score < LOW_SCORE_CUTOFF) and not followup:
        return {
            "answer": topic_fallback(req.question, top),
            "cached": False,
            "used_gpt": False,
            "cache_type": "none",
            "followup": followup,
            "top_score": round(top_score, 4),
            "top_matches": [{"score": round(s, 4), "score_base": round(base, 4), "id": rid, "tags": tags}
                            for (s, rid, *_r, _source, tags, base) in top],
            "timing": t.snapshot(),
        }

    top_ids = [rid for (_, rid, *_r) in top]

    use_fast = is_factual_question(req.question, top_score, history)
    if use_fast:
        ck = make_cache_key_fast(req.question, top_ids, req.k)
        cache_type = "fast"
    else:
        ck = make_cache_key_conversational(req.question, top_ids, req.k, req.conversation_id, history)
        cache_type = "conversational"

    cached = cache_get(ck)
    t.mark("cache_get_ms")

    if cached:
        return {
            "answer": cached,
            "cached": True,
            "used_gpt": False,
            "cache_type": cache_type,
            "followup": followup,
            "top_score": round(top_score, 4),
            "top_matches": [{"score": round(s, 4), "score_base": round(base, 4), "id": rid, "tags": tags}
                            for (s, rid, *_r, _source, tags, base) in top],
            "timing": t.snapshot(),
        }

    if (not PILOT_MODE) and (top_score >= TOP_SCORE_THRESHOLD):
        _, rid, qq, aa, source, tags, base = top[0]
        return {
            "answer": aa,
            "cached": False,
            "used_gpt": False,
            "cache_type": "none",
            "followup": followup,
            "top_score": round(top_score, 4),
            "top_matches": [{"score": round(s, 4), "score_base": round(base, 4), "id": rid, "tags": tags}
                            for (s, rid, *_r, _source, tags, base) in top],
            "timing": t.snapshot(),
        }

    context = "\n\n".join(
        [
            f"ידע {i+1} (score={score:.3f}, tags={tags}):\n"
            f"שאלה: {clip(qq, CLIP_Q_CHARS)}\n"
            f"תשובה: {clip(aa, CLIP_A_CHARS)}"
            for i, (score, _rid, qq, aa, _source, tags, _base) in enumerate(top)
        ]
    )
    t.mark("build_context_ms")

    history_text = ""
    if history and not use_fast:
        lines = []
        for role, content in history[-HISTORY_LIMIT_TO_GPT:]:
            role_h = "אמא" if role == "user" else "עוזרת"
            lines.append(f"{role_h}: {clip(content, 220)}")
        history_text = "\n".join(lines)
    t.mark("build_history_ms")

    system = (
        "את עוזרת לאימהות אחרי לידה. כתבי בעברית פשוטה וקצרה.\n"
        "חוקים: אל תאבחני. הסתמכי על מידע מהמאגר בלבד. אם חסר מידע—שאלה אחת קצרה.\n"
        "סגנון: בלי פתיח קבוע. אם יש סימני עומס/מצוקה (כאב חזק, חוסר שינה קיצוני, בלבול, הצפה) "
        "הוסיפי משפט אמפתי אחד בלבד. אחרת—גשי ישר לתשובה."
    )

    user = (
        f"שאלה: {req.question}\n"
        f"הקשר קודם: {history_text or 'אין'}\n\n"
        f"מידע מהמאגר:\n{context}\n\n"
        "עני בקצרה ובצורה מעשית (6–10 שורות). אם חסר פרט קריטי—שאלה אחת."
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    t.mark("gpt_ms")

    final_answer = resp.choices[0].message.content or ""

    if req.conversation_id:
        add_message(req.conversation_id, "assistant", final_answer)

    cache_set(ck, final_answer)
    t.mark("cache_set_ms")

    return {
        "answer": final_answer,
        "cached": False,
        "used_gpt": True,
        "cache_type": cache_type,
        "followup": followup,
        "top_score": round(top_score, 4),
        "top_matches": [{"score": round(s, 4), "score_base": round(base, 4), "id": rid, "tags": tags}
                        for (s, rid, *_r, _source, tags, base) in top],
        "timing": t.snapshot(),
    }
