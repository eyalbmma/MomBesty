# api.py
import os
import time
from typing import Optional, List, Tuple

from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# =========================
# App config
# =========================
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
    k: int = 3
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


# =========================
# Chat engine import
# =========================
from chat_engine import (
    build_augmented_question,
    topic_fallback,
    build_gpt_answer,
    message_is_just_age_answer,
)

# =========================
# DB helpers (נשארים כאן)
# =========================
import sqlite3

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
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    history: List[Tuple[str, str]] = []
    if req.conversation_id:
        history = get_recent_messages(req.conversation_id)

    if req.conversation_id:
        add_message(req.conversation_id, "user", req.question)

    # 🔹 שלב זהה למה שהיה – רק מופרד
    augmented_q = build_augmented_question(req.question, history)

    # ⚠️ כרגע אין Intent / RAG כאן בכוונה
    # אם אין הקשר מספק – fallback
    if not augmented_q or len(augmented_q) < 3:
        answer = topic_fallback(req.question)
    else:
        # בינתיים context ריק – נתקן בשלב הבא
        answer = build_gpt_answer(
            question=req.question,
            history=history,
            context="",
        )

    if req.conversation_id:
        add_message(req.conversation_id, "assistant", answer)

    return {
        "answer": answer,
        "used_gpt": True,
        "cached": False,
    }
