import json, re, hashlib

INP = "llli_dump.jsonl"
OUT = "llli_dump_clean.jsonl"

# ביטויים שמצביעים על "רעש" (תפריטים/עמודי פורום כלליים)
BAD_TEXT_PATTERNS = [
    r"\btop of page\b",
    r"\bbottom of page\b",
    r"\bskip to content\b",
]

BAD_TITLE_PATTERNS = [
    r"פורום הנקה\s*\|",         # דפי אינדקס/כללי
]

def norm_text(t: str) -> str:
    t = t.replace("\u200f", " ").replace("\u200e", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def is_bad_record(title: str, text: str) -> bool:
    if not text or len(text) < 400:  # תרים/תוריד סף לפי מה שראית
        return True
    for p in BAD_TEXT_PATTERNS:
        if re.search(p, text, flags=re.I):
            return True
    for p in BAD_TITLE_PATTERNS:
        if re.search(p, title, flags=re.I):
            return True
    return False

seen_hashes = set()
kept = 0
dropped = 0
dupes = 0

with open(INP, encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fout:
    for line in fin:
        r = json.loads(line)
        title = (r.get("title") or "").strip()
        text = norm_text(r.get("text") or "")

        if is_bad_record(title, text):
            dropped += 1
            continue

        # dedup לפי טקסט מנורמל
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            dupes += 1
            continue
        seen_hashes.add(h)

        r["title"] = title
        r["text"] = text
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        kept += 1

print("kept:", kept, "dropped:", dropped, "dupes:", dupes, "->", OUT)
