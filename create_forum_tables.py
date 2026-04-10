import sqlite3
import time

DB_PATH = "data/rag.db"

def db_connect():
    return sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)

def create_forum_tables():
    con = db_connect()
    cur = con.cursor()

    # Posts (פוסטים)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS forum_posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        content TEXT NOT NULL,
        is_anonymous INTEGER NOT NULL DEFAULT 1,
        empathy_count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'active', -- active/hidden/deleted
        created_at INTEGER NOT NULL
    );
    """)

    # Comments (תגובות)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS forum_comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER NOT NULL,
        user_id TEXT NOT NULL,
        content TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active', -- active/hidden/deleted
        created_at INTEGER NOT NULL,
        FOREIGN KEY(post_id) REFERENCES forum_posts(id)
    );
    """)

    # Reports (דיווחים)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS forum_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_type TEXT NOT NULL, -- 'post' / 'comment'
        target_id INTEGER NOT NULL,
        reporter_user_id TEXT NOT NULL,
        reason TEXT,
        created_at INTEGER NOT NULL
    );
    """)

    # Rate limiting פשוט (כדי למנוע ספאם)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS forum_rate_limits (
        user_id TEXT NOT NULL,
        action TEXT NOT NULL, -- 'post' / 'comment' / 'report'
        last_ts INTEGER NOT NULL,
        PRIMARY KEY(user_id, action)
    );
    """)

    # Indexes לביצועים בפיד
    cur.execute("CREATE INDEX IF NOT EXISTS idx_forum_posts_created ON forum_posts(created_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_forum_posts_status_created ON forum_posts(status, created_at);")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_forum_comments_post_created ON forum_comments(post_id, created_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_forum_comments_status_post_created ON forum_comments(status, post_id, created_at);")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_forum_reports_created ON forum_reports(created_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_forum_reports_target ON forum_reports(target_type, target_id);")

    con.commit()
    con.close()

def main():
    print("Creating forum tables in:", DB_PATH)
    create_forum_tables()
    print("Done ✅")

if __name__ == "__main__":
    main()
