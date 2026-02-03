import sqlite3
from typing import List, Dict

from daily_support_utils import compute_day_index, now_ts


def get_opted_in_users(con: sqlite3.Connection) -> List[Dict]:
    cur = con.execute("""
        SELECT
            user_id,
            postpartum_start_ts,
            created_at
        FROM postpartum_profiles
        WHERE opt_in = 1
    """)
    return [dict(row) for row in cur.fetchall()]


def already_sent_today(con: sqlite3.Connection, user_id: str, day_index: int) -> bool:
    cur = con.execute("""
        SELECT 1
        FROM daily_support_delivery_log
        WHERE user_id = ? AND day_index = ?
        LIMIT 1
    """, (user_id, day_index))
    return cur.fetchone() is not None


def get_message_for_day(con: sqlite3.Connection, day_index: int):
    cur = con.execute("""
        SELECT id, day_index, text, interaction_hint
        FROM daily_support_messages
        WHERE day_index = ? AND is_active = 1
        LIMIT 1
    """, (day_index,))
    row = cur.fetchone()
    return dict(row) if row else None


def build_daily_support_plan(con: sqlite3.Connection) -> List[Dict]:
    """
    מחזיר רשימה של:
      { user_id, day_index, message }
    רק למי שצריכה לקבל היום הודעה.
    """
    results: List[Dict] = []
    users = get_opted_in_users(con)
    now = now_ts()

    for u in users:
        day_index = compute_day_index(
            postpartum_start_ts=u["postpartum_start_ts"],
            fallback_start_ts=u["created_at"],
            now=now,
        )

        msg = get_message_for_day(con, day_index)
        if not msg:
            continue

        if already_sent_today(con, u["user_id"], day_index):
            continue

        results.append({
            "user_id": u["user_id"],
            "day_index": day_index,
            "message": msg,
        })

    return results
