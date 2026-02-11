# api.py
import os
import time
import sqlite3
from typing import Optional, List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

APP_VERSION = "render-test-11-followup-context"

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
    topic_fallback,
)

# =========================
# DB helpers
# =========================
DB_PATH = "/data/rag.db"


def db_connect():
    con = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


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
# Intent Router
# =========================
def detect_intent(question: str) -> str:
    q = (question or "").strip().lower()

    if len(q.split()) <= 2:
        return "unclear"

    if any(w in q for w in ["מוצפת", "קשה לי", "בוכה", "מפחדת", "לא עומדת"]):
        return "emotional"

    if any(w in q for w in ["כואב", "חום", "דימום", "תפרים"]):
        return "physical"

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
    if req.conversation_id:
        history = get_recent_messages(req.conversation_id)

    if req.conversation_id:
        add_message(req.conversation_id, "user", req.question)

    intent = detect_intent(req.question)

    # 1) חוסר מידע → fallback קצר, בלי GPT
    if intent == "unclear":
        answer = topic_fallback(req.question)
        used_gpt = False

    # 2) יש intent ברור → GPT
    else:
        augmented_q = build_augmented_question(req.question, history)

        mode = "full"
        if intent == "emotional":
            # followup רק אם כבר יש הודעת assistant בהיסטוריה
            if any(r == "assistant" and (c or "").strip() for r, c in history):
                mode = "followup"

        answer = build_gpt_answer(
            question=augmented_q,
            history=history,
            context="",
            mode=mode,
        )
        used_gpt = True

    if req.conversation_id:
        add_message(req.conversation_id, "assistant", answer)

    return {
        "answer": answer,
        "intent": intent,
        "used_gpt": used_gpt,
    }
