import os
import json
import sqlite3
from openai import OpenAI
print("RUNNING embed_db.py")

DB_PATH = "rag.db"
TABLE = "rag1"          # לפי מה שרואים אצלך
ID_COL = "id"
TEXT_COLS = ["question", "answer"]
EMB_COL = "embedding"

EMBED_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(text: str) -> list[float]:
    text = text.replace("\n", " ").strip()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def main():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # ודא שיש עמודת embedding (אם כבר יצרת, זה לא יזיק לנסות ולתפוס שגיאה)
    try:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN {EMB_COL} TEXT;")
        con.commit()
        print("✅ Added embedding column")
    except sqlite3.OperationalError:
        pass  # כבר קיים

    # שלוף שורות בלי embedding
    cur.execute(
        f"SELECT {ID_COL}, {', '.join(TEXT_COLS)} FROM {TABLE} "
        f"WHERE {EMB_COL} IS NULL OR {EMB_COL} = ''"
    )
    rows = cur.fetchall()
    print(f"Found {len(rows)} rows to embed...")

    for i, (row_id, q, a) in enumerate(rows, 1):
        combined = f"Q: {q}\nA: {a}"
        vec = embed(combined)
        cur.execute(
            f"UPDATE {TABLE} SET {EMB_COL} = ? WHERE {ID_COL} = ?",
            (json.dumps(vec), row_id)
        )
        if i % 20 == 0:
            con.commit()
            print(f"…embedded {i}/{len(rows)}")

    con.commit()
    con.close()
    print("✅ Done")

if __name__ == "__main__":
    main()
