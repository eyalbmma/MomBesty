import requests
from typing import List, Dict

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
MAX_PUSH_BODY_CHARS = 140


def _is_expo_token(token: str) -> bool:
    t = (token or "").strip()
    return t.startswith("ExponentPushToken[") or t.startswith("ExpoPushToken[")


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def send_expo_push(tokens: List[str], title: str, body: str, data: Dict):
    """
    שליחה דרך Expo Push Service (best-effort).
    משותף לפורום ולליווי היומי.
    """
    tokens = [t for t in (tokens or []) if _is_expo_token(t)]
    if not tokens:
        return {"ok": True, "sent": 0}

    payload = [
        {
            "to": t,
            "title": title,
            "body": _clip(body, MAX_PUSH_BODY_CHARS),
            "data": data or {},
            "sound": "default",
        }
        for t in tokens
    ]

    try:
        resp = requests.post(EXPO_PUSH_URL, json=payload, timeout=8)
        return {"ok": resp.ok, "sent": len(payload), "status": resp.status_code}
    except Exception as e:
        return {"ok": False, "sent": 0, "error": str(e)}
