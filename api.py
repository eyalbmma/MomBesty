import os
import json
import sqlite3
import time
import hashlib
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# =========================
# Config
# =========================
APP_VERSION = "render-test-6-chat-memory"

DB_PATH = "rag.db"
TABLE = "rag_clean"
EMB_COL = "embedding"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ---- MVP controls ----
PILOT_MODE = True
TOP_SCORE_THRESHOLD = 0.80
LOW_SCORE_CUTOFF = 0.65            # מתחת לזה "בדרך כלל" לא קוראים ל-GPT
SOFT_CUTOFF = 0.58                 # ✅ בפיילוט: כן קוראים ל-GPT גם בטווח 0.58–0.65
CACHE_TTL_SECONDS = 7 * 24 * 3600
PROMPT_VER = "v4-chat-memory"
# ----------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()


# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https://.*\.netlify\.app$",
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- OPTIONS Preflight ---
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


# =========================
# SQLite helpers
# =========================
def db_connect():
    # check_same_thread=False עוזר אם יש ריבוי threads בשרת
    return sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)

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
    con = db_connect()
    cur = con.cursor()

    # Cache לתשובות GPT
    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key TEXT PRIMARY KEY,
            answer TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
    """)

    # Cache ל-Embeddings
    cur.execute("""
        CREATE TABLE IF NOT EXISTS emb_cache (
            q_norm TEXT PRIMARY KEY,
            embedding TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
    """)

    # טבלאות שיחה
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
    return [(r or "", c or "") for (r, c) in rows]


# =========================
# LLM answer cache (includes conversation + history fingerprint)
# =========================
def make_cache_key(
    question: str,
    top_ids: List[int],
    k: int,
    conversation_id: Optional[str],
    history: List[Tuple[str, str]],
) -> str:
    # טביעת אצבע קצרה של ההיסטוריה כדי שה-cache לא יחזיר תשובה משיחה אחרת
    hist_compact = "|".join([f"{r}:{norm_q(c)[:80]}" for r, c in history[-10:]])
    base = (
        f"{PROMPT_VER}|{CHAT_MODEL}|k={k}"
        f"|conv={conversation_id or ''}"
        f"|q={norm_q(question)}"
        f"|ids={','.join(map(str, top_ids))}"
        f"|hist={hist_compact}"
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
    ans, ts = row
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
    emb_json, ts = row
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
    """
    ✅ לא נשבר אם אין is_active.
    """
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
    embs_mat = (embs_mat / norms).astype(np.float32)

    RAG_ROWS = local_rows
    RAG_EMBS = embs_mat
    RAG_READY = True

def retrieve_top_k(qv: np.ndarray, k: int):
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
        top.append((float(scores[int(i)]), rid, qq, aa, source, tags))
    return top


# =========================
# Init
# =========================
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
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "rag_rows_loaded": len(RAG_ROWS),
        "rag_ready": RAG_READY,
    }

@app.post("/reload_rag")
def reload_rag():
    load_rag_to_memory()
    return {"ok": True, "rag_rows_loaded": len(RAG_ROWS), "rag_ready": RAG_READY}


@app.post("/ask")
def ask(req: AskReq):
    t = Timer()

    qv = get_embedding(req.question, timing=t)
    t.mark("embed_ms")

    top = retrieve_top_k(qv, req.k)
    t.mark("retrieve_ms")

    return {
        "matches": [
            {
                "score": round(score, 4),
                "id": rid,
                "question": qq,
                "answer": aa,
                "source": source,
                "tags": tags,
            }
            for score, rid, qq, aa, source, tags in top
        ],
        "timing": t.snapshot(),
    }


@app.post("/ask_final")
def ask_final(req: AskReq):
    t = Timer()

    # 0) שימור קונטקסט שיחה
    if req.user_id and req.conversation_id:
        upsert_conversation(req.user_id, req.conversation_id)
        add_message(req.conversation_id, "user", req.question)

    # ✅ שליפת ההיסטוריה כדי שיהיה "זיכרון" (גם אחרי שבוע)
    history: List[Tuple[str, str]] = []
    if req.conversation_id:
        history = get_recent_messages(req.conversation_id, limit=12)
        t.mark("history_ms")

    # 1) Embedding
    qv = get_embedding(req.question, timing=t)
    t.mark("embed_ms")

    # 2) Retrieval
    top = retrieve_top_k(qv, req.k)
    t.mark("retrieve_ms")

    if not top:
        return {
            "answer": "לא מצאתי מידע רלוונטי במאגר כרגע. תוכלי לנסח מחדש או להוסיף פרט אחד קטן?",
            "cached": False,
            "used_gpt": False,
            "top_matches": [],
            "timing": t.snapshot(),
        }

    top_score = float(top[0][0])

    # ✅ Guardrail
    if top_score < SOFT_CUTOFF:
        return {
            "answer": (
                "אין לי מספיק התאמה מדויקת במאגר כדי לענות בביטחון. "
                "שאלה קצרה כדי לדייק: מה הדבר שהכי מטריד אותך עכשיו — מצב רוח/חרדה ובכי, "
                "עייפות וחוסר שינה, או משהו פיזי (כאב/דימום/חום)?"
            ),
            "cached": False,
            "used_gpt": False,
            "top_score": round(top_score, 4),
            "top_matches": [{"score": round(s, 4), "id": rid, "tags": tags} for s, rid, *_ , tags in top],
            "timing": t.snapshot(),
        }

    if (not PILOT_MODE) and (top_score < LOW_SCORE_CUTOFF):
        return {
            "answer": (
                "אין לי מספיק מידע מדויק במאגר כדי לענות בצורה טובה. "
                "שאלה קצרה כדי לדייק: מה הנושא העיקרי כאן — כאב/תסמין פיזי אצלך, "
                "קושי רגשי/עייפות, או משהו שקשור לתינוק (שינה/האכלה/בכי)?"
            ),
            "cached": False,
            "used_gpt": False,
            "top_score": round(top_score, 4),
            "top_matches": [{"score": round(s, 4), "id": rid, "tags": tags} for s, rid, *_ , tags in top],
            "timing": t.snapshot(),
        }

    top_ids = [rid for _, rid, *_ in top]
    ck = make_cache_key(req.question, top_ids, req.k, req.conversation_id, history)

    cached = cache_get(ck)
    t.mark("cache_get_ms")

    if cached:
        return {
            "answer": cached,
            "cached": True,
            "used_gpt": False,
            "top_score": round(top_score, 4),
            "top_matches": [{"score": round(s, 4), "id": rid, "tags": tags} for s, rid, *_ , tags in top],
            "timing": t.snapshot(),
        }

    # 4) DB-only (רק אם לא בפיילוט)
    if (not PILOT_MODE) and (top_score >= TOP_SCORE_THRESHOLD):
        _, rid, qq, aa, source, tags = top[0]
        return {
            "answer": aa,
            "cached": False,
            "used_gpt": False,
            "top_score": round(top_score, 4),
            "top_matches": [{"score": round(s, 4), "id": rid, "tags": tags} for s, rid, *_ , tags in top],
            "timing": t.snapshot(),
        }

    # 5) Context for GPT (מהמאגר)
    context = "\n\n".join(
        [
            f"ידע {i+1} (score={score:.3f}, tags={tags}):\nשאלה: {qq}\nתשובה: {aa}"
            for i, (score, _, qq, aa, _, tags) in enumerate(top)
        ]
    )
    t.mark("build_context_ms")

    # ✅ שיחת וואטסאפ: מוסיפים היסטוריה כטקסט קצר
    history_text = ""
    if history:
        lines = []
        for role, content in history[-10:]:
            role_h = "אמא" if role == "user" else "עוזרת"
            lines.append(f"{role_h}: {content}")
        history_text = "\n".join(lines)
    t.mark("build_history_ms")

    system = (
        "את עוזרת דיגיטלית לאימהות אחרי לידה, בטון חם, מכיל, אמפתי ולא שיפוטי. "
        "המטרה: לתת לאמא תחושת ביטחון, הבנה ועידוד — יחד עם מידע מדויק ורלוונטי.\n\n"
        "כללי חובה:\n"
        "• אל תאבחני ואל תקבעי 'יש לך X'.\n"
        "• השתמשי רק במידע שניתן לך ב'מידע מהמאגר'. אם אין מספיק מידע — שאלי שאלה אחת קצרה במקום לנחש.\n"
        "• קחי בחשבון את 'הקשר שיחה קודם' כדי לזכור פרטים שהאמא אמרה (אבל אל תמציאי פרטים).\n"
        "• כתבי בעברית פשוטה, קצרה וברורה.\n\n"
        "מבנה קבוע לתשובה:\n"
        "1) רגע איתך (2–3 משפטים)\n"
        "2) מה אפשר לעשות עכשיו (3–5 נקודות)\n"
        "3) מתי כדאי לפנות לעזרה (2–4 נקודות)\n"
        "4) שאלה קצרה (רק אם חסר פרט קריטי)\n"
    )

    user = (
        f"שאלת האמא (עכשיו): {req.question}\n\n"
        f"הקשר שיחה קודם (לזכור רצף/פרטים, בלי להמציא):\n"
        f"{history_text or '(אין הקשר קודם)'}\n\n"
        f"מידע מהמאגר (השתמשי רק בזה, אל תוסיפי ידע מבחוץ):\n{context}\n\n"
        "אם אין במידע מהמאגר בסיס לתשובה מדויקת — שאלי שאלה אחת קצרה כדי לדייק."
    )

    # 6) GPT call
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
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
        "top_score": round(top_score, 4),
        "top_matches": [{"score": round(s, 4), "id": rid, "tags": tags} for s, rid, *_ , tags in top],
        "timing": t.snapshot(),
    }
