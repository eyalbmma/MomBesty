import sqlite3

DB_PATH = "data/rag.db"

def main():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS tracker_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        baby_id TEXT NOT NULL DEFAULT 'default',
        type TEXT NOT NULL,              -- feed/diaper/sleep
        occurred_at INTEGER NOT NULL,     -- זמן האירוע

        method TEXT,                     -- feed
        amount_ml INTEGER,               -- feed
        diaper_kind TEXT,                -- diaper (pee/poo/both)
        duration_min INTEGER,            -- sleep
        sleep_started_at INTEGER,        -- שעת התחלת שינה (epoch seconds)
        sleep_ended_at INTEGER,          -- שעת סיום שינה (epoch seconds)
        is_active_sleep INTEGER NOT NULL DEFAULT 0, -- 1=שינה פעילה
        pump_started_at INTEGER,         -- שעת התחלת שאיבה (epoch seconds)
        pump_ended_at INTEGER,           -- שעת סיום שאיבה (epoch seconds)
        is_active_pump INTEGER NOT NULL DEFAULT 0, -- 1=שאיבה פעילה

        note TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        created_at INTEGER NOT NULL
    )
    """)

    # ניסיון עדין להוסיף עמודות אם הטבלה כבר קיימת (SQLite לא תומך IF NOT EXISTS ב-ALTER)
    # נתעלם משגיאה אם העמודה כבר קיימת.
    alter_statements = [
        "ALTER TABLE tracker_entries ADD COLUMN sleep_started_at INTEGER",
        "ALTER TABLE tracker_entries ADD COLUMN sleep_ended_at INTEGER",
        "ALTER TABLE tracker_entries ADD COLUMN is_active_sleep INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE tracker_entries ADD COLUMN pump_started_at INTEGER",
        "ALTER TABLE tracker_entries ADD COLUMN pump_ended_at INTEGER",
        "ALTER TABLE tracker_entries ADD COLUMN is_active_pump INTEGER NOT NULL DEFAULT 0",
    ]
    for stmt in alter_statements:
        try:
            cur.execute(stmt)
        except Exception:
            pass

    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracker_user ON tracker_entries(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracker_occurred ON tracker_entries(occurred_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracker_type ON tracker_entries(type)")

    con.commit()
    con.close()
    print("✅ Tracker tables created/verified in rag.db")

if __name__ == "__main__":
    main()
