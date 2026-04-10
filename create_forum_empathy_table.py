import sqlite3

DB_PATH = "data/rag.db"

def main():
    con = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS forum_empathy (
        post_id INTEGER NOT NULL,
        user_id TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        PRIMARY KEY (post_id, user_id),
        FOREIGN KEY(post_id) REFERENCES forum_posts(id)
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_forum_empathy_post ON forum_empathy(post_id);")

    con.commit()
    con.close()
    print("Done ✅ forum_empathy table created")

if __name__ == "__main__":
    main()
