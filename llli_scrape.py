import argparse
import json
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import trafilatura
from tqdm import tqdm


UA = "Mozilla/5.0 (compatible; MomBestyCrawler/1.0)"
HEADERS = {"User-Agent": UA}


def norm_url(u: str) -> str:
    u = u.strip()
    u = u.split("#")[0]
    return u


def is_same_domain(url: str, domain: str) -> bool:
    return urlparse(url).netloc.lower().endswith(domain)


def fetch(url: str, timeout: int = 25) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.text
    except Exception:
        return None


def extract_title_and_text(html: str, url: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    # trafilatura מצוין למאמרים/תוכן
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=True,   # בפורום זה יכול לעזור (אם התגובות הן "comments")
        include_tables=False
    ) or ""

    text = text.strip()

    # fallback אם trafilatura לא מצא
    if not text:
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = "\n".join(
            ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()
        )

    return title.strip(), text


def collect_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        u = norm_url(urljoin(base_url, href))
        if u.startswith("http"):
            out.append(u)
    return out


# -------------------------
# Forum logic (pagination)
# -------------------------
def forum_page_url(base: str, page_num: int) -> str:
    # לפי התמונה: /breastfeeding-forum/p-15
    return f"{base}/p-{page_num}"


def is_forum_list_page(url: str) -> bool:
    return "/breastfeeding-forum/p-" in url


def is_forum_thread_url(url: str) -> bool:
    # נחשב "דיון" כל URL תחת breastfeeding-forum שאינו רשימת עמודים
    return ("/breastfeeding-forum/" in url) and ("/p-" not in url)


def scrape_forum_threads(forum_base: str, pages: int, delay: float) -> list[str]:
    thread_urls = set()

    for p in range(1, pages + 1):
        url = forum_page_url(forum_base, p)
        html = fetch(url)
        time.sleep(delay)
        if not html:
            continue

        links = collect_links(html, url)
        for u in links:
            if is_forum_thread_url(u) and is_same_domain(u, "lllisrael.org.il"):
                thread_urls.add(u)

    # מסננים דברים לא רלוונטיים (login, search וכו')
    cleaned = []
    for u in thread_urls:
        if any(x in u.lower() for x in ["login", "register", "wp-admin", "tag/", "category/"]):
            continue
        cleaned.append(u)

    return sorted(set(cleaned))


# -------------------------
# Info site logic (articles)
# -------------------------
def is_info_url(url: str) -> bool:
    return "/breastfeeding-info" in url


def is_probable_article(url: str) -> bool:
    # כלל אצבע: כתבה היא דף deeper, לא העמוד הראשי בלבד
    # לדוגמה: /breastfeeding-info/.../something
    if not is_info_url(url):
        return False
    # לא לשמור את העמוד הראשי/קטגוריות כלליות בלבד (אלא אם תרצה)
    if url.rstrip("/") in [
        "https://www.lllisrael.org.il/breastfeeding-info",
        "https://www.lllisrael.org.il/breastfeeding-info/"
    ]:
        return False
    # מסננים עמודים כלליים לא תוכן
    if any(x in url.lower() for x in ["youtube", "instagram", "facebook"]):
        return False
    return True


def scrape_info_articles(info_base: str, max_queue: int, delay: float) -> list[str]:
    """
    BFS קטן: מתחילים מעמוד info_base, אוספים לינקים פנימיים,
    ושומרים רק מה שנראה כמו "כתבות".
    """
    seen = set([info_base])
    q = [info_base]
    articles = set()

    while q and len(seen) < max_queue:
        url = q.pop(0)
        html = fetch(url)
        time.sleep(delay)
        if not html:
            continue

        for u in collect_links(html, url):
            if not is_same_domain(u, "lllisrael.org.il"):
                continue
            if not is_info_url(u):
                continue
            u = norm_url(u)
            if u in seen:
                continue
            seen.add(u)
            q.append(u)

            if is_probable_article(u):
                articles.add(u)

    return sorted(articles)


def write_jsonl(records: list[dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forum_pages", type=int, default=15, help="כמה עמודי פורום (p-1..p-N)")
    ap.add_argument("--info_max_queue", type=int, default=600, help="כמה דפים לסרוק ב-breastfeeding-info (BFS)")
    ap.add_argument("--delay", type=float, default=1.5, help="דיליי בין בקשות (שניות)")
    ap.add_argument("--out", type=str, default="llli_dump.jsonl", help="קובץ פלט JSONL")
    args = ap.parse_args()

    forum_base = "https://www.lllisrael.org.il/breastfeeding-forum"
    info_base = "https://www.lllisrael.org.il/breastfeeding-info"

    # 1) forum: collect threads
    forum_threads = scrape_forum_threads(forum_base, args.forum_pages, args.delay)
    print(f"Forum threads found: {len(forum_threads)}")

    # 2) info: collect articles
    info_articles = scrape_info_articles(info_base, args.info_max_queue, args.delay)
    print(f"Info articles found: {len(info_articles)}")

    # 3) fetch content
    all_urls = [("forum", u) for u in forum_threads] + [("info", u) for u in info_articles]
    records = []

    for kind, url in tqdm(all_urls, desc="Downloading"):
        html = fetch(url)
        time.sleep(args.delay)
        if not html:
            continue
        title, text = extract_title_and_text(html, url)

        # ניקוי קטן: להוריד טקסט קצר מדי
        if len(text) < 200:
            continue

        records.append({
            "source_site": kind,
            "url": url,
            "title": title,
            "text": text,
            "fetched_at": int(time.time())
        })

    write_jsonl(records, args.out)
    print("Saved:", args.out, "records:", len(records))


if __name__ == "__main__":
    main()
