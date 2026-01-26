import os, json, sqlite3, time, hashlib, re
import numpy as np
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

APP_VERSION = "render-test-3"  # עדכן כדי לוודא ש-Render עלה עם הגרסה החדשה

DB_PATH = "rag.db"
TABLE = "rag1"
EMB_COL = "embedding"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ---- MVP controls ----
PILOT_MODE = True            # כרגע רק אתה והשותפה -> תמיד GPT (אלא אם יש cache)
TOP_SCORE_THRESHOLD = 0.80   # יכנס לפעולה רק כשתשנה PILOT_MODE=False
CACHE_TTL_SECONDS = 7 * 24 * 3600  # שבוע
PROMPT_VER = "v1"            # להעלות ל-v2 אם משנים פרומפט משמעותית
# ----------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# --- CORS: מאפשר ל-React (localhost) + Netlify לקרוא לשרת ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://tranquil-gumdrop-998ac3.netlify.app",
        "https://harmonious-scone-ad9f51.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------------------------------------

# --- חשוב: Preflight OPTIONS כדי שלא נקבל 405 ---
@app.options("/ask_final")
def options_ask_final():
    return Response(status_code=200)

@app.options("/ask")
def options_ask():
    return Response(status_code=200)
# ------------------------------------------------

class AskReq(BaseModel):
    question: str
    k: int = 3


# =========================
# Timing utilities
# =========================
def now_ms() -> int:
    return int(time.perf_counter() * 1000)

class Timer:
    def __init__(self):
        self.t0 = now_ms()
        self.marks = {}

    def mark(self, name: str):
        self.marks[name] = now_ms() - self.t0

    def snapshot(self):
        out = dict(self.marks)
        out["total_ms"] = now_ms() - self.t0
        return out


def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text.replace("\n", " ").strip()]
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)


# ============================================================
# RAG in-memory cache (טעינה פעם אחת) + NORMALIZATION מראש
# ============================================================
RAG_ROWS: list[tuple[int, str, str, str, str]] = []  # (id, q, a, source, tags)
RAG_EMBS: np.ndarray | None = None                   # shape: (N, D) normalized
RAG_READY: bool = False

def load_rag_to_memory():
    """
    טוען פעם אחת את כל הרשומות עם embeddings מה-SQLite לזיכרון,
    ומנרמל מראש את embeddings (כדי שבבקשה רק נעשה dot product).
    """
    global RAG_ROWS, RAG_EMBS, RAG_READY

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        f"SELECT id, question, answer, source, tags, {EMB_COL} "
        f"FROM {TABLE} WHERE {EMB_COL} IS NOT NULL AND {EMB_COL} <> ''"
    )
    rows = cur.fetchall()
    con.close()

    local_rows = []
    embs = []

    for rid, q, a, source, tags, emb_json in rows:
        try:
            ev = np.array(json.loads(emb_json), dtype=np.float32)
        except Exception:
            continue
        if ev.size == 0:
            continue
        embs.append(ev)
        local_rows.append((rid, q, a, source, tags))

    if not embs:
        RAG_ROWS = []
        RAG_EMBS = None
        RAG_READY = False
        return

    embs_mat = np.vstack(embs).astype(np.float32)

    # ✅ normalize embeddings once (N, D)
    norms = np.linalg.norm(embs_mat, axis=1, keepdims=True) + 1e-12
    embs_mat = (embs_mat / norms).astype(np.float32)

    RAG_ROWS = local_rows
    RAG_EMBS = embs_mat
    RAG_READY = True

def retrieve_top_k(qv: np.ndarray, k: int):
    """
    cosine similarity כששני הצדדים מנורמלים = dot product.
    פלט: list[(score, rid, q, a, source, tags)]
    """
    if (not RAG_READY) or (RAG_EMBS is None) or (len(RAG_ROWS) == 0):
        return []

    # ✅ normalize query once
    qv = qv.astype(np.float32)
    qv = qv / (np.linalg.norm(qv) + 1e-12)

    # scores: (N,)
    scores = RAG_EMBS @ qv

    k = max(1, min(int(k), scores.shape[0]))

    # top-k indices without sorting full array
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]

    top = []
    for i in idx:
        rid, qq, aa, source, tags = RAG_ROWS[int(i)]
        top.append((float(scores[int(i)]), rid, qq, aa, source, tags))
    return top


# ---------- CACHE (SQLite) ----------
def ensure_cache_table():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key TEXT PRIMARY KEY,
            answer TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
    """)
    con.commit()
    con.close()

def norm_q(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^0-9a-zA-Zא-ת\s]", "", s)
    return s

def make_cache_key(question: str, top_ids: list[int], k: int) -> str:
    base = f"{PROMPT_VER}|{CHAT_MODEL}|k={k}|{norm_q(question)}|ids={','.join(map(str, top_ids))}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def cache_get(key: str):
    con = sqlite3.connect(DB_PATH)
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
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO llm_cache(cache_key, answer, created_at) VALUES(?,?,?)",
        (key, answer, int(time.time()))
    )
    con.commit()
    con.close()

ensure_cache_table()
load_rag_to_memory()
# -----------------------------------


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
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "rag_rows_loaded": len(RAG_ROWS),
        "rag_ready": RAG_READY,
    }


# Debug endpoint: מחזיר התאמות
@app.post("/ask")
def ask(req: AskReq):
    t = Timer()

    qv = get_embedding(req.question)
    t.mark("embed_ms")

    top = retrieve_top_k(qv, req.k)
    t.mark("retrieve_ms")

    print("TIMING /ask:", t.snapshot(), "rows_loaded:", len(RAG_ROWS))

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


# MVP endpoint: תשובה אחת לאמא + cache + (אופציונלי) בלי GPT לפי score כשלא בפיילוט
@app.post("/ask_final")
def ask_final(req: AskReq):
    t = Timer()

    # 1) Embedding
    qv = get_embedding(req.question)
    t.mark("embed_ms")

    # 2) Retrieval (מהיר, בזיכרון + normalized)
    top = retrieve_top_k(qv, req.k)
    t.mark("retrieve_ms")

    if not top:
        print("TIMING /ask_final (no top):", t.snapshot(), "rows_loaded:", len(RAG_ROWS))
        return {
            "answer": "לא מצאתי מידע רלוונטי במאגר כרגע. תוכלי לנסח מחדש או להוסיף פרט אחד קטן?",
            "cached": False,
            "used_gpt": False,
            "top_matches": [],
            "timing": t.snapshot(),
        }

    top_score = float(top[0][0])
    top_ids = [rid for _, rid, *_ in top]
    ck = make_cache_key(req.question, top_ids, req.k)

    # 3) Cache get
    cached = cache_get(ck)
    t.mark("cache_get_ms")

    if cached:
        print("TIMING /ask_final (cache hit):", t.snapshot(), "top_score:", round(top_score, 4))
        return {
            "answer": cached,
            "cached": True,
            "used_gpt": False,
            "top_score": round(top_score, 4),
            "top_matches": [
                {"score": round(score, 4), "id": rid, "tags": tags}
                for score, rid, _, _, _, tags in top
            ],
            "timing": t.snapshot(),
        }

    # 4) אם לא בפיילוט, אפשר להחזיר תשובת DB בלי GPT לפי score
    if (not PILOT_MODE) and (top_score >= TOP_SCORE_THRESHOLD):
        _, rid, qq, aa, source, tags = top[0]
        print("TIMING /ask_final (db only):", t.snapshot(), "top_score:", round(top_score, 4))
        return {
            "answer": aa,
            "cached": False,
            "used_gpt": False,
            "top_score": round(top_score, 4),
            "top_matches": [
                {"score": round(score, 4), "id": rid, "tags": tags}
                for score, rid, _, _, _, tags in top
            ],
            "timing": t.snapshot(),
        }

    # 5) Build context for GPT
    context = "\n\n".join(
        [
            f"ידע {i+1} (score={score:.3f}, tags={tags}):\nשאלה: {qq}\nתשובה: {aa}"
            for i, (score, _, qq, aa, _, tags) in enumerate(top)
        ]
    )
    t.mark("build_context_ms")

    system = (
        "את עוזרת דיגיטלית רגועה ולא שיפוטית לאימהות אחרי לידה. "
        "תני תשובה קצרה וברורה בעברית. "
        "חובה מבנה קבוע:\n"
        "1) תשובה קצרה וברורה (2–4 משפטים)\n"
        "2) מה לעשות עכשיו (3 נקודות)\n"
        "3) מתי לפנות לרופא/ה (2–3 נקודות)\n"
        "בלי אבחנות. בלי הפחדות. אם חסר מידע, שאלי שאלה אחת קצרה."
    )

    user = (
        f"שאלת האמא: {req.question}\n\n"
        f"השתמשי רק במידע הבא (אל תוסיפי ידע מבחוץ):\n{context}"
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

    final_answer = resp.choices[0].message.content

    # 7) Cache set
    cache_set(ck, final_answer)
    t.mark("cache_set_ms")

    print("TIMING /ask_final (gpt):", t.snapshot(), "top_score:", round(top_score, 4))

    return {
        "answer": final_answer,
        "cached": False,
        "used_gpt": True,
        "top_score": round(top_score, 4),
        "top_matches": [
            {"score": round(score, 4), "id": rid, "tags": tags}
            for score, rid, _, _, _, tags in top
        ],
        "timing": t.snapshot(),
    }
