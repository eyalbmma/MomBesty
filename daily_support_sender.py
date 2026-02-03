# daily_support_sender.py
import time
from typing import Any, Dict, List, Optional

from daily_support_plan import build_daily_support_plan
from forum_api import db_connect, _get_user_push_tokens
from push_utils import send_expo_push

PUSH_TITLE = "באה להיות איתך רגע"


def _now_ts() -> int:
    return int(time.time())


def _clip(s: str, n: int = 120) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def _compose_body(msg: Dict[str, Any]) -> str:
    text = (msg.get("text") or "").strip()
    hint = (msg.get("interaction_hint") or "").strip()
    if hint:
        return f"{text}\n{hint}"
    return text


def _get_tokens(con, user_id: str) -> List[str]:
    """
    תומך בשתי חתימות אפשריות של _get_user_push_tokens:
    A) _get_user_push_tokens(con, user_id)
    B) _get_user_push_tokens(user_id)
    """
    try:
        tokens = _get_user_push_tokens(con, user_id)  # type: ignore[arg-type]
    except TypeError:
        tokens = _get_user_push_tokens(user_id)  # type: ignore[call-arg]
    return [t for t in (tokens or []) if isinstance(t, str) and t.strip()]


def _send_to_tokens(tokens: List[str], title: str, body: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    תומך בשתי חתימות אפשריות של send_expo_push:
    A) send_expo_push(tokens=[...], title=..., body=..., data=...)
    B) send_expo_push(expo_token="...", title=..., body=..., data=...)
    """
    # ניסיון 1: שליחה באצווה (tokens=...)
    try:
        r = send_expo_push(tokens=tokens, title=title, body=body, data=data)  # type: ignore[call-arg]
        return r if isinstance(r, dict) else {"ok": bool(r)}
    except TypeError:
        pass

    # ניסיון 2: שליחה לכל טוקן בנפרד (expo_token=...)
    ok_any = False
    errors: List[str] = []
    results: List[Dict[str, Any]] = []

    for tok in tokens:
        try:
            rr = send_expo_push(expo_token=tok, title=title, body=body, data=data)  # type: ignore[call-arg]
            rr_dict = rr if isinstance(rr, dict) else {"ok": bool(rr)}
            results.append(rr_dict)
            ok_any = ok_any or bool(rr_dict.get("ok"))
        except Exception as e:
            errors.append(str(e))

    return {"ok": ok_any, "results": results[:5], "errors": errors[:3]}


def run_daily_support(
    con,
    dry_run: bool = True,
    limit: int = 200,
) -> Dict[str, Any]:
    """
    מריץ את מנגנון ה-daily support.
    - dry_run=True: לא שולח ולא רושם לוג, רק מדווח
    - dry_run=False: שולח + רושם daily_support_delivery_log
    """
    plan = build_daily_support_plan(con)
    print(f"[DAILY_SUPPORT] candidates={len(plan)} dry_run={dry_run}")

    sent = 0
    skipped_no_tokens = 0
    failed = 0
    preview: List[Dict[str, Any]] = []

    for item in plan[: max(0, int(limit))]:
        user_id = item.get("user_id")
        day_index = int(item.get("day_index") or 0)
        msg = item.get("message") or {}

        if not user_id or day_index <= 0 or not msg.get("text"):
            preview.append({"status": "skip:bad_item", "item": str(item)[:200]})
            continue

        tokens = _get_tokens(con, user_id)
        if not tokens:
            skipped_no_tokens += 1
            preview.append(
                {
                    "user_id": user_id,
                    "day_index": day_index,
                    "message_id": msg.get("id"),
                    "status": "skip:no_tokens",
                    "text": _clip(msg.get("text") or ""),
                }
            )
            continue

        body = _compose_body(msg)

        data = {
            "screen": "AskChat",
            "prefill": msg.get("text") or "",
            "startNewConversation": True,
            "source": "daily_support",
            "dayIndex": day_index,
        }

        if dry_run:
            preview.append(
                {
                    "user_id": user_id,
                    "day_index": day_index,
                    "message_id": msg.get("id"),
                    "tokens": len(tokens),
                    "status": "dry_run",
                    "text": _clip(msg.get("text") or ""),
                }
            )
            continue

        resp = _send_to_tokens(tokens=tokens, title=PUSH_TITLE, body=body, data=data)

        if resp.get("ok"):
            # מונע כפילויות/קריסות אם כבר נשלח היום
            con.execute(
                """
                INSERT OR IGNORE INTO daily_support_delivery_log
                (user_id, day_index, message_id, sent_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, day_index, int(msg.get("id") or 0), _now_ts()),
            )
            con.commit()
            sent += 1
            preview.append(
                {
                    "user_id": user_id,
                    "day_index": day_index,
                    "message_id": msg.get("id"),
                    "tokens": len(tokens),
                    "status": "sent",
                    "text": _clip(msg.get("text") or ""),
                }
            )
            print(f"[DAILY_SUPPORT] sent user={user_id} day={day_index} tokens={len(tokens)}")
        else:
            failed += 1
            preview.append(
                {
                    "user_id": user_id,
                    "day_index": day_index,
                    "message_id": msg.get("id"),
                    "tokens": len(tokens),
                    "status": "fail",
                    "errors": resp.get("errors") or resp.get("error") or None,
                    "text": _clip(msg.get("text") or ""),
                }
            )
            print(f"[DAILY_SUPPORT] failed user={user_id} resp={resp}")

    return {
        "ok": True,
        "dry_run": dry_run,
        "candidates": len(plan),
        "sent": sent,
        "skipped_no_tokens": skipped_no_tokens,
        "failed": failed,
        "preview": preview[:50],  # תשובה קצרה
    }


def send_daily_support_messages():
    """
    תאימות אחורה: אם קראת בעבר send_daily_support_messages()
    זה יריץ שליחה אמיתית.
    """
    con = db_connect()
    try:
        return run_daily_support(con, dry_run=False)
    finally:
        con.close()


if __name__ == "__main__":
    # להרצה ידנית:
    # dry_run=True כדי לראות candidates בלי לשלוח
    con = db_connect()
    try:
        out = run_daily_support(con, dry_run=True)
        print("[DAILY_SUPPORT] summary:", out)
    finally:
        con.close()
