# chat_engine.py
import hashlib
import re
from typing import List, Tuple, Optional

from openai import OpenAI

# =========================
# Chat config
# =========================
CHAT_MODEL = "gpt-4o-mini"

HISTORY_LIMIT_TO_GPT = 4
MAX_TOKENS = 320
TEMPERATURE = 0.3

# bump version so you can track prompt changes in logs/caching later
PROMPT_VER = "v12-full-guardrail-no-journal-meditation-music"

client = OpenAI()

# =========================
# Text helpers
# =========================
def norm_q(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^0-9a-zA-Zא-ת\s]", "", s)
    return s


def clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


# =========================
# Follow-up + age helpers
# =========================
_AGE_RE = re.compile(r"^(?:בן|בת)?\s*\d{1,2}\s*(?:חודשים|חודש|שנים|שנה)?$")


def message_is_just_age_answer(text: str) -> bool:
    return bool(_AGE_RE.match(norm_q(text)))


def assistant_asked_age(text: str) -> bool:
    t = norm_q(text)
    return any(x in t for x in ["בן כמה", "בת כמה", "גיל התינוק", "בן/בת כמה"])


def build_augmented_question(question: str, history: List[Tuple[str, str]]) -> str:
    """
    anti-loop: אם המשתמש/ת ענו רק גיל אחרי שהעוזרת שאלה גיל,
    מחברים את הגיל לשאלה הקודמת לצורך דיוק.
    """
    if not message_is_just_age_answer(question):
        return question

    last_assistant = next((c for r, c in reversed(history) if r == "assistant"), "")
    if not assistant_asked_age(last_assistant):
        return question

    prev_user = next(
        (c for r, c in reversed(history) if r == "user" and not message_is_just_age_answer(c)),
        "",
    )
    if not prev_user:
        return question

    return f"שאלה קודמת: {clip(prev_user, 260)}\nגיל: {question}"


# =========================
# Topic fallback
# =========================
def topic_fallback(question: str) -> str:
    qn = norm_q(question)
    if any(x in qn for x in ["הנקה", "שד", "פטמה"]):
        return "כדי לדייק: בן/בת כמה התינוק, והאם יש כאב/סדקים או קושי בהיצמדות?"
    if any(x in qn for x in ["שינה", "בכי", "אוכל"]):
        return "כדי לדייק: בן/בת כמה התינוק, ומה בדיוק קורה עכשיו?"
    if any(x in qn for x in ["חום", "דימום", "כאב", "תפרים"]):
        return "כדי לדייק: כמה זמן אחרי לידה את, והאם יש החמרה?"
    return "כדי לענות מדויק, תוכלי להוסיף עוד משפט של הקשר?"


# =========================
# Followup ACK only (one short sentence)
# =========================
def build_followup_ack(history: List[Tuple[str, str]]) -> str:
    """
    מחזיר משפט פתיחה קצר אחד בלבד (ללא עצות/שאלות),
    כדי שהשרת ינעל את הצעד + השאלה ולא ייווצרו "קיצורים" של GPT.
    """
    system = (
        "את עוזרת לאימהות אחרי לידה.\n"
        "כתבי משפט אחד בלבד, קצר בעברית, שמכיר בזה שקשה.\n"
        "אסור להוסיף עצות, אסור להציע פעולות, אסור לשאול שאלה.\n"
        "אסור להשתמש במספור.\n"
        "רק משפט אחד."
    )

    history_text = "\n".join(
        [f"{'אמא' if r == 'user' else 'עוזרת'}: {clip(c, 160)}" for r, c in history[-4:]]
    )

    user = f"הקשר קצר:\n{history_text or 'אין'}\n"

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=40,
    )

    text = (resp.choices[0].message.content or "").strip()
    return text if text else "שומעת אותך—זה באמת קשה."


# =========================
# (Optional) cache key — not used in api.py right now
# =========================
def make_cache_key_conversational(
    question: str,
    top_ids: List[int],
    k: int,
    conversation_id: Optional[str],
    history: List[Tuple[str, str]],
) -> str:
    hist = "|".join([f"{r}:{norm_q(c)[:60]}" for r, c in history[-HISTORY_LIMIT_TO_GPT:]])
    base = (
        f"{PROMPT_VER}|{CHAT_MODEL}|k={k}"
        f"|conv={conversation_id or ''}"
        f"|q={norm_q(question)}"
        f"|ids={','.join(map(str, top_ids))}"
        f"|hist={hist}"
    )
    return hashlib.sha256(base.encode()).hexdigest()


# =========================
# GPT answer builder (FULL only in practice)
# =========================
def build_gpt_answer(
    question: str,
    history: List[Tuple[str, str]],
    context: str,
    mode: str = "full",  # נשאר לתאימות
    forced_step: Optional[str] = None,  # נשאר לתאימות
) -> str:
    history_text = "\n".join(
        [
            f"{'אמא' if r == 'user' else 'עוזרת'}: {clip(c, 220)}"
            for r, c in history[-HISTORY_LIMIT_TO_GPT:]
        ]
    )

    # ✅ NEW: Guardrail for FULL
    system = (
        "את עוזרת לאימהות אחרי לידה. כתבי בעברית פשוטה, רגישה ולא שיפוטית.\n"
        "מבנה חובה:\n"
        "1) תיקוף רגשי קצר\n"
        "2) מה אפשר לעשות עכשיו (2–3 נקודות)\n"
        "3) מתי לפנות לבדיקה\n"
        "4) שאלה אחת רק אם חסר פרט קריטי\n\n"
        "כלל קשיח:\n"
        "אל תציעי יומן רגשות, מדיטציה, מוזיקה או תרגולים כלליים.\n"
        "תציעי רק צעדים פרקטיים קטנים ומיידיים ותמיכה רגשית ישירה.\n\n"
        "אל תאבחני ואל תתני טיפול רפואי."
    )

    user = (
        f"שאלה: {question}\n"
        f"הקשר קודם:\n{history_text or 'אין'}\n\n"
        f"מידע מהמאגר:\n{context}\n"
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

    return resp.choices[0].message.content or ""
