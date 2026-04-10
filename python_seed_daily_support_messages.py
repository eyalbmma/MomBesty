import sqlite3
import time
from typing import List, Dict, Any

DB_PATH = "/data/rag.db"  # שנה אם צריך

def now_ts() -> int:
    return int(time.time())

def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def create_table(con: sqlite3.Connection):
    con.execute("""
    CREATE TABLE IF NOT EXISTS daily_support_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        day_index INTEGER NOT NULL UNIQUE,         -- 1..60
        stage TEXT NOT NULL,                       -- early|adjustment|growth|return|stability
        text TEXT NOT NULL,                        -- הודעה
        interaction_hint TEXT,                     -- אופציונלי: משפט הזמנה רכה
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at INTEGER NOT NULL
    );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_daily_support_active_day ON daily_support_messages(is_active, day_index);")
    con.commit()

def seed_messages(con: sqlite3.Connection, rows: List[Dict[str, Any]]):
    ts = now_ts()
    for r in rows:
        con.execute("""
        INSERT INTO daily_support_messages (day_index, stage, text, interaction_hint, is_active, created_at)
        VALUES (?, ?, ?, ?, 1, ?)
        ON CONFLICT(day_index) DO UPDATE SET
            stage=excluded.stage,
            text=excluded.text,
            interaction_hint=excluded.interaction_hint,
            is_active=1;
        """, (r["day_index"], r["stage"], r["text"], r.get("interaction_hint"), ts))
    con.commit()

def get_message_for_day(con: sqlite3.Connection, day_index: int):
    cur = con.execute("""
        SELECT day_index, stage, text, interaction_hint
        FROM daily_support_messages
        WHERE day_index=? AND is_active=1
        LIMIT 1;
    """, (day_index,))
    return cur.fetchone()

def build_expo_push_payload(expo_push_token: str, msg_row) -> Dict[str, Any]:
    """
    זה payload שאתה שולח ל-Expo Push API.
    שים לב ל-data: זה מה שמאפשר ניווט אוטומטי לצ'אט AI בלחיצה.
    """
    body = msg_row["text"]
    hint = msg_row["interaction_hint"]
    final_body = body if not hint else f"{body}\n{hint}"

    return {
        "to": expo_push_token,
        "title": "באה להיות איתך רגע",
        "body": final_body,
        "sound": "default",
        "data": {
            "screen": "AskChat",                 # 👈 שם המסך בניווט שלך
            "source": "daily_support",
            "day_index": msg_row["day_index"],
            "prefill": body                      # 👈 אפשר למלא בצ'אט AI טקסט פתיחה
        }
    }

def messages_60() -> List[Dict[str, Any]]:
    # עיקרון: לא כל הודעה שאלה. לפעמים רק "פתח רך" שאפשר להתעלם ממנו.
    return [
        # ימים 1–14 | early
        {"day_index": 1, "stage": "early", "text": "היום מותר פשוט להיות.", "interaction_hint": "אם בא לך—מה עובר עלייך עכשיו?"},
        {"day_index": 2, "stage": "early", "text": "הגוף שלך עושה עבודה ענקית.", "interaction_hint": "יש תחושה אחת שבולטת היום?"},
        {"day_index": 3, "stage": "early", "text": "רגשות מתחלפים עכשיו זה טבעי.", "interaction_hint": "אפשר לשתף במילה אחת."},
        {"day_index": 4, "stage": "early", "text": "נשימה אחת קטנה גם נחשבת.", "interaction_hint": "רוצה לנסות לנשום רגע יחד?"},
        {"day_index": 5, "stage": "early", "text": "את לא אמורה לדעת הכול.", "interaction_hint": "משהו אחד שלא ברור לך היום?"},
        {"day_index": 6, "stage": "early", "text": "גם בלבול הוא חלק מההתחלה.", "interaction_hint": "אם תרצי—אפשר לפרוק פה."},
        {"day_index": 7, "stage": "early", "text": "מגיעה לך מילה טובה מעצמך.", "interaction_hint": "מה היית אומרת לחברה במצבך?"},
        {"day_index": 8, "stage": "early", "text": "התקופה הזו רגישה, וזה מובן.", "interaction_hint": "יש משהו שצריך יותר עדינות היום?"},
        {"day_index": 9, "stage": "early", "text": "מותר לנוח בלי להסביר לאף אחד.", "interaction_hint": None},
        {"day_index": 10, "stage": "early", "text": "את לא לבד במה שאת מרגישה.", "interaction_hint": "אם בא לך—כתבי שורה אחת."},
        {"day_index": 11, "stage": "early", "text": "דבר קטן יכול להחזיק יום שלם.", "interaction_hint": "מה היה הדבר הקטן שלך היום?"},
        {"day_index": 12, "stage": "early", "text": "גם היום הזה נחשב.", "interaction_hint": None},
        {"day_index": 13, "stage": "early", "text": "מותר לבכות גם בלי סיבה ברורה.", "interaction_hint": "יש רגש שרוצה מקום?"},
        {"day_index": 14, "stage": "early", "text": "החזקת הרבה היום—וזה מספיק.", "interaction_hint": "רוצה שנשים לב לזה יחד?"},

        # ימים 15–30 | adjustment
        {"day_index": 15, "stage": "adjustment", "text": "איך אפשר להקל עלייך היום?", "interaction_hint": "אפילו שינוי קטן נחשב."},
        {"day_index": 16, "stage": "adjustment", "text": "עייפות לא אומרת כישלון.", "interaction_hint": None},
        {"day_index": 17, "stage": "adjustment", "text": "מותר להוריד ציפיות בתקופה הזו.", "interaction_hint": "מה היית מורידה היום ב-10%?"},
        {"day_index": 18, "stage": "adjustment", "text": "זה בסדר אם לא הספקת הכול.", "interaction_hint": None},
        {"day_index": 19, "stage": "adjustment", "text": "שמת לב למשהו טוב קטן?", "interaction_hint": "אפשר גם משהו ממש קטן."},
        {"day_index": 20, "stage": "adjustment", "text": "את עושה התאמות כל הזמן.", "interaction_hint": "מה השתנה אצלך מאז השבוע הראשון?"},
        {"day_index": 21, "stage": "adjustment", "text": "לבקש עזרה זה כוח, לא חולשה.", "interaction_hint": "יש מישהו שאפשר להיעזר בו היום?"},
        {"day_index": 22, "stage": "adjustment", "text": "גם 'לא מושלם' יכול להיות מספיק טוב.", "interaction_hint": None},
        {"day_index": 23, "stage": "adjustment", "text": "הקשבה לגוף היא חכמה.", "interaction_hint": "מה הגוף מבקש היום?"},
        {"day_index": 24, "stage": "adjustment", "text": "את לומדת את זה תוך כדי.", "interaction_hint": None},
        {"day_index": 25, "stage": "adjustment", "text": "רגע שקט קטן יכול לעזור.", "interaction_hint": "אפשר למצוא דקה אחת היום?"},
        {"day_index": 26, "stage": "adjustment", "text": "גם ימים כאלה עוברים.", "interaction_hint": None},
        {"day_index": 27, "stage": "adjustment", "text": "מותר לך להיות לא מושלמת.", "interaction_hint": "מה היית משחררת מעצמך היום?"},
        {"day_index": 28, "stage": "adjustment", "text": "את מחזיקה רצף, גם בעייפות.", "interaction_hint": None},
        {"day_index": 29, "stage": "adjustment", "text": "מה עזר לך להמשיך היום?", "interaction_hint": "אפשר לענות במילה אחת."},
        {"day_index": 30, "stage": "adjustment", "text": "את עושה כמיטב יכולתך עכשיו.", "interaction_hint": "וזה באמת הרבה."},

        # ימים 31–40 | growth
        {"day_index": 31, "stage": "growth", "text": "היום זה יום של החזקה.", "interaction_hint": "רוצה לבחור דבר אחד להחזיק איתו?"},
        {"day_index": 32, "stage": "growth", "text": "אפשר לרכך את היום, לא לפתור אותו.", "interaction_hint": None},
        {"day_index": 33, "stage": "growth", "text": "חוסר שקט הוא שלב שעובר.", "interaction_hint": "איך הוא פוגש אותך היום?"},
        {"day_index": 34, "stage": "growth", "text": "גם רק להיות שם—זה המון.", "interaction_hint": None},
        {"day_index": 35, "stage": "growth", "text": "נשימה עמוקה אחת יכולה לעזור.", "interaction_hint": "אם בא לך—עכשיו אחת."},
        {"day_index": 36, "stage": "growth", "text": "מותר שיהיה קשה היום.", "interaction_hint": None},
        {"day_index": 37, "stage": "growth", "text": "קושי לא אומר שמשהו לא בסדר.", "interaction_hint": "רוצה לשתף מה מקשה?"},
        {"day_index": 38, "stage": "growth", "text": "גם לך מגיע רגע קטן לעצמך.", "interaction_hint": "איזה רגע היית רוצה?"},
        {"day_index": 39, "stage": "growth", "text": "שינויים מביאים בלבול זמני.", "interaction_hint": None},
        {"day_index": 40, "stage": "growth", "text": "את מתמודדת באומץ.", "interaction_hint": "אם תרצי—אפשר לדבר על זה כאן."},

        # ימים 41–50 | return
        {"day_index": 41, "stage": "return", "text": "חשבת על עצמך רגע היום?", "interaction_hint": "גם מחשבה קטנה נחשבת."},
        {"day_index": 42, "stage": "return", "text": "מותר לרצות להרגיש טוב יותר.", "interaction_hint": None},
        {"day_index": 43, "stage": "return", "text": "הגוף חוזר בקצב שלו.", "interaction_hint": "מה היית רוצה שיידעו עליו?"},
        {"day_index": 44, "stage": "return", "text": "מה עושה לך קצת טוב לאחרונה?", "interaction_hint": "אפשר משהו ממש פשוט."},
        {"day_index": 45, "stage": "return", "text": "את בונה איזון חדש.", "interaction_hint": None},
        {"day_index": 46, "stage": "return", "text": "מגיע מקום גם לצורך שלך.", "interaction_hint": "איזה צורך מבקש תשומת לב?"},
        {"day_index": 47, "stage": "return", "text": "את לא צריכה למהר לשום מקום.", "interaction_hint": None},
        {"day_index": 48, "stage": "return", "text": "יש לך זכות לדאוג לעצמך.", "interaction_hint": "מה יעזור לך היום ב-5%?"},
        {"day_index": 49, "stage": "return", "text": "גבול אחד קטן יכול להגן עלייך.", "interaction_hint": "יש גבול שבא לך לשמור?"},
        {"day_index": 50, "stage": "return", "text": "את מתקרבת לעצמך מחדש.", "interaction_hint": "לאט זה גם דרך."},

        # ימים 51–60 | stability
        {"day_index": 51, "stage": "stability", "text": "מה עוזר לך להחזיק רצף?", "interaction_hint": "אפשר לבחור דבר אחד."},
        {"day_index": 52, "stage": "stability", "text": "יש בך חוסן שקט.", "interaction_hint": None},
        {"day_index": 53, "stage": "stability", "text": "בחרת משהו אחד חשוב היום—זה מספיק.", "interaction_hint": None},
        {"day_index": 54, "stage": "stability", "text": "את לא עומדת במקום.", "interaction_hint": "שמת לב להתקדמות קטנה?"},
        {"day_index": 55, "stage": "stability", "text": "מותר לצמוח בהדרגה.", "interaction_hint": None},
        {"day_index": 56, "stage": "stability", "text": "התקדמות קטנה היא עדיין התקדמות.", "interaction_hint": "מה השתפר טיפה?"},
        {"day_index": 57, "stage": "stability", "text": "את סומכת יותר על עצמך.", "interaction_hint": "איפה את מרגישה את זה?"},
        {"day_index": 58, "stage": "stability", "text": "כל צעד שלך נחשב.", "interaction_hint": None},
        {"day_index": 59, "stage": "stability", "text": "אפשר לקחת מחר דבר אחד עדין.", "interaction_hint": "מה היית רוצה לקחת איתך?"},
        {"day_index": 60, "stage": "stability", "text": "את עושה משהו משמעותי מאוד.", "interaction_hint": "אם בא לך—בואי נכתוב על זה יחד."},
    ]

def main():
    con = connect()
    create_table(con)
    seed_messages(con, messages_60())

    

    con.close()

if __name__ == "__main__":
    main()
