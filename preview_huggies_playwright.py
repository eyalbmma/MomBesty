# preview_huggies_playwright.py
import asyncio
import re
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

URLS = [
    "https://www.huggies.co.il/parenting/new-born/first-days/how-to-change-a-diaper",
    "https://www.huggies.co.il/parenting/new-born/first-days/returning-home-with-baby",
]

MIN_ANSWER_CHARS = 120
LIMIT = 5

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def normalize_question(h: str) -> str:
    h = clean_text(h)
    if not h:
        return ""
    return h if h.endswith("?") else h + "?"

def extract_qna_from_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.body
    if not main:
        return []

    out = []
    for h in main.find_all(["h2", "h3"]):
        heading = clean_text(h.get_text(" ", strip=True))
        if not heading:
            continue

        parts = []
        for sib in h.find_next_siblings():
            if sib.name in ["h2", "h3"]:
                break
            if sib.name in ["p", "ul", "ol"]:
                txt = clean_text(sib.get_text(" ", strip=True))
                if txt:
                    parts.append(txt)

        if not parts:
            continue

        answer = clean_text(" ".join(parts))
        if len(answer) < MIN_ANSWER_CHARS:
            continue

        out.append((normalize_question(heading), answer))

    return out

async def main():
    printed = 0
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(locale="he-IL")
        page = await context.new_page()

        for url in URLS:
            print("\n" + "="*90)
            print("URL:", url)

            try:
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if resp is None:
                    print("No response (blocked or navigation failed).")
                    continue
                if resp.status >= 400:
                    print("HTTP status:", resp.status)
                    continue

                await page.wait_for_timeout(1500)  # נותן רגע ל-JS/תוכן
                html = await page.content()

                qnas = extract_qna_from_html(html)
                if not qnas:
                    print("No Q&A extracted (or filtered out).")
                    continue

                for q, a in qnas:
                    print("\n--- RECORD ---")
                    print("Q:", q)
                    print("A:", a[:450] + ("..." if len(a) > 450 else ""))
                    printed += 1
                    if printed >= LIMIT:
                        print("\n✅ Preview limit reached.")
                        await browser.close()
                        return

            except Exception as e:
                print("ERROR:", repr(e))

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
