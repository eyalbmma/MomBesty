import sqlite3

DB_PATH = "/data/rag.db"

def main():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS content_topics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        order_index INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS content_articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        summary TEXT,
        content TEXT NOT NULL,
        source TEXT,
        order_index INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL,
        FOREIGN KEY(topic_id) REFERENCES content_topics(id)
    )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_topic ON content_articles(topic_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_created ON content_articles(created_at)")

    con.commit()
    con.close()
    print("✅ Content tables created/verified in rag.db")

if __name__ == "__main__":
    main()
