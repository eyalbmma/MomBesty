import json
import sqlite3
from pathlib import Path

DB_PATH = r"C:\Project\babys\rag.db"
JSON_PATH = r"C:\Project\babys\input_368_390.json"

print("Using DB:", DB_PATH)
print("Using JSON:", JSON_PATH)


def to_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_str(value):
    if value is None:
        return ""
    return str(value).strip()


def main():
    db_file = Path(DB_PATH)
    json_file = Path(JSON_PATH)

    if not db_file.exists():
        raise FileNotFoundError(f"DB not found: {db_file}")

    if not json_file.exists():
        raise FileNotFoundError(f"JSON not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError("JSON must contain a list of rows")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    inserted = 0
    skipped = 0

    for row in rows:
        original_id = row.get("original_id") or row.get("id")
        original_id = to_int(original_id, None)

        if original_id is None:
            print("❌ Missing original_id/id in row, skipping")
            skipped += 1
            continue

        question_curated = to_str(row.get("question"))
        answer_curated = to_str(row.get("answer"))
        stage = to_str(row.get("stage"))
        category = to_str(row.get("category"))
        age_bucket = to_str(row.get("age_bucket"))
        topic_group = to_str(row.get("topic_group"))
        notes = to_str(row.get("notes"))
        audit_status = to_str(row.get("audit_status")) or "REWRITE"
        safety_level = to_str(row.get("safety_level")) or "normal"
        priority_score = to_int(row.get("priority_score"), 3)
        is_active = to_int(row.get("is_active"), 1)

        embedding_text = to_str(row.get("embedding_text"))
        if not embedding_text:
            embedding_text = f"{question_curated} {answer_curated}".strip()

        # בדיקת כפילות
        cur.execute(
            "SELECT 1 FROM rag_curated WHERE original_id = ?",
            (original_id,)
        )
        if cur.fetchone():
            print(f"⚠️ Already exists: {original_id}")
            skipped += 1
            continue

        cur.execute(
            """
            INSERT INTO rag_curated (
                original_id,
                question_original,
                answer_original,
                question_curated,
                answer_curated,
                audit_status,
                stage,
                source,
                tags,
                notes,
                category,
                age_bucket,
                embedding_text,
                is_active,
                topic_group,
                priority_score,
                safety_level
            )
            SELECT
                rc.id,
                rc.question,
                rc.answer,
                ?,
                ?,
                ?,
                ?,
                rc.source,
                rc.tags,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            FROM rag_clean rc
            WHERE rc.id = ?
            """,
            (
                question_curated,
                answer_curated,
                audit_status,
                stage,
                notes,
                category,
                age_bucket,
                embedding_text,
                is_active,
                topic_group,
                priority_score,
                safety_level,
                original_id,
            ),
        )

        if cur.rowcount > 0:
            inserted += 1
            print(f"✅ Inserted: {original_id}")
        else:
            print(f"❌ Not found in rag_clean: {original_id}")
            skipped += 1

    conn.commit()
    conn.close()

    print(f"\n✅ Done. Inserted {inserted} rows into rag_curated.")
    print(f"⚠️ Skipped {skipped} rows.")


if __name__ == "__main__":
    main()