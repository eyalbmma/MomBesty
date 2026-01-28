import os
import json
import sqlite3
import time
from openai import OpenAI

print("RUNNING embed_db.py")

DB_PATH = "rag.db"
TABLE = "rag_clean"
ID_COL = "id"
EMB_COL = "embedding"
EMBED_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(text: str):
    text = (text or "").replace("\n", " ").strip()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def main():
    con = sqlite3.connect(DB_PATH, timeout=30)
    cur = con.cursor()

    # ודא שיש עמודת embedding
    try:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN {EMB_COL} TEXT;")
        con.commit()
        print("✅ Added embedding column")
    except sqlite3.OperationalError:
        pass

    # שלוף שורות בלי embedding
    cur.execute(
        f"""
        SELECT {ID_COL}, question, answer
        FROM {TABLE}
        WHERE {EMB_COL} IS NULL OR {EMB_COL} = ''
        """
    )
    rows = cur.fetchall()
    total = len(rows)
    print(f"Found {total} rows to embed...")

    for i, (row_id, q, a) in enumerate(rows, 1):
        combined = f"Q: {q}\nA: {a}"
        try:
            vec = embed(combined)
        except Exception as e:
            print(f"❌ failed on id={row_id}: {e}")
            time.sleep(2)
            continue

        cur.execute(
            f"UPDATE {TABLE} SET {EMB_COL} = ? WHERE {ID_COL} = ?",
            (json.dumps(vec), row_id)
        )

        if i % 50 == 0:
            con.commit()
            print(f"…embedded {i}/{total}")

    con.commit()
    con.close()
    print("✅ Done")

if __name__ == "__main__":
    main()
