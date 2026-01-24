
import os, json, sqlite3, time, hashlib, re
import numpy as np
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

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
        "https://harmonious-scone-ad9f51.netlify.app",  # הכתובת החדשה שנטליפיי יצר
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


def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text.replace("\n", " ").strip()]
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def fetch_all_rows():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        f"SELECT id, question, answer, source, tags, {EMB_COL} "
        f"FROM {TABLE} WHERE {EMB_COL} IS NOT NULL AND {EMB_COL} <> ''"
    )
    rows = cur.fetchall()
    con.close()
    return rows


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
    s = re.sub(r"[^0-9a-zA-Zא-ת\s]", "", s)  # מוריד פיסוק/סימנים
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
# -----------------------------------


@app.get("/health")
def health():
    return {
        "ok": True,
        "table": TABLE,
        "pilot_mode": PILOT_MODE,
        "top_score_threshold": TOP_SCORE_THRESHOLD,
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
    }


# Debug endpoint: מחזיר התאמות
@app.post("/ask")
def ask(req: AskReq):
    qv = get_embedding(req.question)
    rows = fetch_all_rows()

    scored = []
    for rid, q, a, source, tags, emb_json in rows:
        ev = np.array(json.loads(emb_json), dtype=np.float32)
        scored.append((cosine(qv, ev), rid, q, a, source, tags))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:req.k]

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
        ]
    }


# MVP endpoint: תשובה אחת לאמא + cache + (אופציונלי) בלי GPT לפי score כשלא בפיילוט
@app.post("/ask_final")
def ask_final(req: AskReq):
    qv = get_embedding(req.question)
    rows = fetch_all_rows()

    scored = []
    for rid, q, a, source, tags, emb_json in rows:
        ev = np.array(json.loads(emb_json), dtype=np.float32)
        scored.append((cosine(qv, ev), rid, q, a, source, tags))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:req.k]

    if not top:
        return {
            "answer": "לא מצאתי מידע רלוונטי במאגר כרגע. תוכלי לנסח מחדש או להוסיף פרט אחד קטן?",
            "cached": False,
            "used_gpt": False,
            "top_matches": [],
        }

    top_score = float(top[0][0])
    top_ids = [rid for _, rid, *_ in top]
    ck = make_cache_key(req.question, top_ids, req.k)

    # 1) Cache hit
    cached = cache_get(ck)
    if cached:
        return {
            "answer": cached,
            "cached": True,
            "used_gpt": False,
            "top_matches": [
                {"score": round(score, 4), "id": rid, "tags": tags}
                for score, rid, _, _, _, tags in top
            ],
        }

    # 2) אם לא בפיילוט, אפשר להחזיר תשובת DB בלי GPT לפי score
    if (not PILOT_MODE) and (top_score >= TOP_SCORE_THRESHOLD):
        _, rid, qq, aa, source, tags = top[0]
        return {
            "answer": aa,
            "cached": False,
            "used_gpt": False,
            "top_matches": [
                {"score": round(score, 4), "id": rid, "tags": tags}
                for score, rid, _, _, _, tags in top
            ],
        }

    # 3) אחרת: קוראים ל-GPT
    context = "\n\n".join(
        [
            f"ידע {i+1} (score={score:.3f}, tags={tags}):\nשאלה: {qq}\nתשובה: {aa}"
            for i, (score, _, qq, aa, _, tags) in enumerate(top)
        ]
    )

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

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )

    final_answer = resp.choices[0].message.content

    # שומרים לקאש כדי שלא נשלם פעם נוספת על אותה תוצאה
    cache_set(ck, final_answer)

    return {
        "answer": final_answer,
        "cached": False,
        "used_gpt": True,
        "top_score": round(top_score, 4),
        "top_matches": [
            {"score": round(score, 4), "id": rid, "tags": tags}
            for score, rid, _, _, _, tags in top
        ],
    }
