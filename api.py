# api.py
import os
import time
import sqlite3
from typing import Optional, List, Tuple, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

APP_VERSION = "render-test-15-state-lite-followup-locked"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

# =========================
# Routers
# =========================
from forum_api import router as forum_router
from content_api import router as content_router
from tracker_api import router as tracker_router
from circles_api import router as circles_router

app.include_router(forum_router)
app.include_router(content_router)
app.include_router(tracker_router)
app.include_router(circles_router)

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
    ],
    allow_origin_regex=r"^https://.*\.netlify\.app$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models
# =========================
class AskReq(BaseModel):
    question: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


# =========================
# Chat engine import
# =========================
from chat_engine import (
    build_augmented_question,
    build_gpt_answer,
    build_followup_ack,  # NEW
    topic_fallback,
)

# =========================
# DB
# =========================
DB_PATH = "/data/rag.db"


def db_connect():
    con = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def ensure_schema():
    con = db_connect()
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_state (
          conversation_id TEXT PRIMARY KEY,
          emotional_streak INTEGER NOT NULL DEFAULT 0,
          last_step TEXT,
          updated_at INTEGER NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conv_state_updated
        ON conversation_state(updated_at);
        """
    )

    con.commit()
    con.close()


ensure_schema()

# =========================
# Conversation messages
# =========================
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


def get_recent_messages(conversation_id: str, limit: int = 6) -> List[Tuple[str, str]]:
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
        (conversation_id, limit),
    )
    rows = cur.fetchall()
    con.close()
    return [(r["role"], r["content"]) for r in reversed(rows)]


# =========================
# Conversation state
# =========================
ALLOWED_STEPS: List[str] = [
    "לשתות מים + 3 נשימות איטיות",
    "לשבת/לשכב 2 דקות בלי משימות",
    "לשלוח הודעה קצרה לאדם קרוב: 'קשה לי, אפשר לדבר רגע?'",
    "לבקש עזרה ספציפית אחת: 'תוכלי/תוכל להחזיק את התינוק 15 דקות?'",
    "מקלחת קצרה",
    "לצאת לאור יום/מרפסת ל-3 דקות",
]


def get_state(conversation_id: str) -> Dict[str, Any]:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT emotional_streak, last_step
        FROM conversation_state
        WHERE conversation_id = ?
        """,
        (conversation_id,),
    )
    row = cur.fetchone()
    con.close()

    if not row:
        return {"emotional_streak": 0, "last_step": None}

    return {
        "emotional_streak": int(row["emotional_streak"] or 0),
        "last_step": row["last_step"],
    }


def save_state(conversation_id: str, emotional_streak: int, last_step: Optional[str]):
    ts = int(time.time())
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO conversation_state(conversation_id, emotional_streak, last_step, updated_at)
        VALUES(?,?,?,?)
        ON CONFLICT(conversation_id) DO UPDATE SET
          emotional_streak=excluded.emotional_streak,
          last_step=excluded.last_step,
          updated_at=excluded.updated_at
        """,
        (conversation_id, emotional_streak, last_step, ts),
    )
    con.commit()
    con.close()


def pick_step(conversation_id: str, emotional_streak: int, last_step: Optional[str]) -> str:
    idx = abs(hash(f"{conversation_id}:{emotional_streak}")) % len(ALLOWED_STEPS)
    step = ALLOWED_STEPS[idx]
    if last_step and step == last_step:
        step = ALLOWED_STEPS[(idx + 1) % len(ALLOWED_STEPS)]
    return step


# =========================
# Intent Router
# =========================
def detect_intent(question: str) -> str:
    q = (question or "").strip().lower()

    if len(q.split()) <= 2:
        return "unclear"

    # physical קודם כדי לא להתנגש עם "מרגישה ..."
    if any(w in q for w in ["כואב", "חום", "דימום", "תפרים", "כאבים"]):
        return "physical"

    emotional_phrases = [
        "מוצפת",
        "קשה לי",
        "בוכה",
        "מפחדת",
        "לא עומדת",
        "אומללה",
        "רע לי",
        "עצובה",
        "בדידות",
        "בודדה",
        "עייפה",
        "אין לי כוחות",
        "נגמרו לי הכוחות",
        "מתוסכלת",
        "חסרת אונים",
        "מרגישה לבד",
        "לא מצליחה להתמודד",
        "שבורה",
        "דיכאון",
        "חרדה",
        "מיואשת",
        "מותשת",
        "מותשת נפשית",
        "עומס נפשי",
        "לא מתפקדת",
        "לא מתפקד",
    ]

    if any(w in q for w in emotional_phrases):
        return "emotional"

    # "מרגישה" רק עם רגש ברור
    if "מרגישה" in q and any(x in q for x in ["רע", "לבד", "עצובה", "אומללה"]):
        return "emotional"

    # כלל "רגש/נפש/עומס"
    if any(x in q for x in ["נפש", "רגש", "עומס", "מתוסכל", "מותש", "מותשת"]):
        return "emotional"

    return "general"


# =========================
# Health
# =========================
@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION}


# =========================
# Chat endpoint
# =========================
@app.post("/ask_final")
def ask_final(req: AskReq):
    if not (req.question or "").strip():
        raise HTTPException(status_code=400, detail="question is required")

    history: List[Tuple[str, str]] = []
    state = {"emotional_streak": 0, "last_step": None}

    if req.conversation_id:
        history = get_recent_messages(req.conversation_id)
        state = get_state(req.conversation_id)

    if req.conversation_id:
        add_message(req.conversation_id, "user", req.question)

    intent = detect_intent(req.question)

    if req.conversation_id:
        if intent == "emotional":
            state["emotional_streak"] += 1
        else:
            state["emotional_streak"] = 0

    forced_step: Optional[str] = None

    # 1) unclear → fallback קצר
    if intent == "unclear":
        answer = topic_fallback(req.question)
        used_gpt = False
        mode = "fallback"

    # 2) intent ברור
    else:
        augmented_q = build_augmented_question(req.question, history)
        mode = "full"

        if intent == "emotional":
            has_assistant = any(r == "assistant" for r, _ in history)
            if has_assistant:
                streak = state["emotional_streak"]

                if streak >= 7:
                    mode = "followup_escalation"
                elif streak >= 4:
                    mode = "followup_checkin"
                else:
                    mode = "followup"

                forced_step = pick_step(
                    req.conversation_id or "",
                    streak,
                    state.get("last_step"),
                )

        # ✅ FOLLOWUP נעול בשרת
        if mode in ("followup", "followup_checkin", "followup_escalation"):
            ack = build_followup_ack(history)

            if mode == "followup":
                closed_q = "מה יותר קשה לך עכשיו — בדידות, עייפות, או לחץ סביב התינוק?"
            elif mode == "followup_checkin":
                closed_q = "הצלחת לנסות משהו קטן מאז? — כן, לא, או לא בטוחה?"
            else:
                closed_q = "איך התפקוד שלך כרגע היום — מצליחה, חלקית, או כמעט לא מצליחה?"

            step_line = forced_step or "לשבת/לשכב 2 דקות בלי משימות"
            answer = f"{ack}\n{step_line}\n{closed_q}"
            used_gpt = True  # השתמשנו ב-GPT למשפט אחד בלבד

        # FULL רגיל
        else:
            answer = build_gpt_answer(
                question=augmented_q,
                history=history,
                context="",
                mode=mode,
                forced_step=None,
            )
            used_gpt = True

    if req.conversation_id:
        add_message(req.conversation_id, "assistant", answer)
        save_state(
            req.conversation_id,
            int(state["emotional_streak"] or 0),
            forced_step if intent == "emotional" else None,
        )

    return {
        "answer": answer,
        "intent": intent,
        "used_gpt": used_gpt,
        "mode": mode,
        "emotional_streak": state["emotional_streak"],
    }
