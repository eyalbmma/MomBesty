from push_utils import send_expo_push
from daily_support_plan import build_daily_support_plan
from forum_api import db_connect, _get_user_push_tokens


PUSH_TITLE = "באה להיות איתך רגע"

def send_daily_support_messages():
    con = db_connect()
    plan = build_daily_support_plan(con)

    print(f"[DAILY_SUPPORT] candidates={len(plan)}")

    for item in plan:
        user_id = item["user_id"]
        day_index = item["day_index"]
        msg = item["message"]

        tokens = _get_user_push_tokens(user_id)
        if not tokens:
            print(f"[DAILY_SUPPORT] skip user={user_id} no tokens")
            continue

        body = msg["text"]
        if msg.get("interaction_hint"):
            body = f'{msg["text"]}\n{msg["interaction_hint"]}'

        data = {
            "screen": "AskChat",
            "prefill": msg["text"],
            "startNewConversation": True,
            "source": "daily_support",
            "dayIndex": day_index,
        }

        resp = send_expo_push(
            tokens=tokens,
            title=PUSH_TITLE,
            body=body,
            data=data,
        )

        if resp.get("ok"):
            con.execute(
                """
                INSERT INTO daily_support_delivery_log
                (user_id, day_index, message_id, sent_at)
                VALUES (?, ?, ?, strftime('%s','now'))
                """,
                (user_id, day_index, msg["id"]),
            )
            con.commit()
            print(f"[DAILY_SUPPORT] sent user={user_id} day={day_index}")
        else:
            print(f"[DAILY_SUPPORT] failed user={user_id} resp={resp}")

    con.close()
if __name__ == "__main__":
    send_daily_support_messages()