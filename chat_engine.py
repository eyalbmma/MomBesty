# chat_engine_state_lite.py
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

PROMPT_VER = "v10-state-lite-steps"

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
# GPT answer builder
# =========================
def build_gpt_answer(
    question: str,
    history: List[Tuple[str, str]],
    context: str,
    mode: str = "full",  # "full" | "followup" | "followup_checkin" | "followup_escalation"
    forced_step: Optional[str] = None,
) -> str:
    history_text = "\n".join(
        [
            f"{'אמא' if r == 'user' else 'עוזרת'}: {clip(c, 220)}"
            for r, c in history[-HISTORY_LIMIT_TO_GPT:]
        ]
    )

    if mode in ("followup", "followup_checkin", "followup_escalation"):
        step_line = forced_step or "לשבת/לשכב 2 דקות בלי משימות"

        if mode == "followup":
            closed_q = "מה יותר קשה לך עכשיו — בדידות, עייפות, או לחץ סביב התינוק?"
        elif mode == "followup_checkin":
            closed_q = "הצלחת לנסות משהו קטן מאז? — כן, לא, או לא בטוחה?"
        else:
            closed_q = "איך התפקוד שלך כרגע היום — מצליחה, חלקית, או כמעט לא מצליחה?"

        system = (
            "את עוזרת לאימהות אחרי לידה.\n"
            "כבר ניתן תיקוף רגשי קודם בשיחה.\n"
            "אל תחזרי על פתיח רגשי כללי.\n\n"
            "מטרה: תשובת המשך קצרה, ממוקדת ויציבה.\n"
            "מבנה חובה:\n"
            "1) משפט אחד קצר שמכיר בזה שזה עדיין קשה\n"
            "2) צעד קטן אחד בלבד לביצוע עכשיו\n"
            "3) שאלה סגורה אחת בלבד\n\n"
            "כללים קשיחים:\n"
            f"- הצעד חייב להיות בדיוק: {step_line}\n"
            f"- השאלה חייבת להיות בדיוק: '{closed_q}'\n"
            "אל תוסיפי צעד נוסף, אל תחליפי ניסוח, ואל תוסיפי עצות.\n\n"
            "אסור להציע מוזיקה, יומן, מדיטציה או עצות כלליות אחרות.\n"
            "אל תאבחני ואל תתני טיפול רפואי.\n"
            "אם יש סימנים שמצריכים עזרה: נסחי בעדינות המלצה לפנות לאחות טיפת חלב/רופא/ה/גורם מקצועי."
        )

    else:
        system = (
            "את עוזרת לאימהות אחרי לידה. כתבי בעברית פשוטה, רגישה ולא שיפוטית.\n"
            "מבנה חובה:\n"
            "1) תיקוף רגשי קצר\n"
            "2) מה אפשר לעשות עכשיו (2–3 נקודות)\n"
            "3) מתי לפנות לבדיקה\n"
            "4) שאלה אחת רק אם חסר פרט קריטי\n"
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
