import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src import bank, config, format_message, state, telegram_client


def _parse_local_date(iso_ts: str, tz: ZoneInfo):
    """Parse an ISO timestamp (assumed UTC if naive) and return its date in tz."""
    try:
        t = datetime.fromisoformat(iso_ts)
    except (ValueError, TypeError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(tz).date()


def already_sent_today(cfg) -> bool:
    """True if a question was sent (pending or in history) on today's local-TZ date."""
    tz = ZoneInfo(cfg.tz)
    today = datetime.now(tz).date()
    s = state.load(cfg.state_path)

    pending = s.get("pending")
    if pending and _parse_local_date(pending.get("sent_at", ""), tz) == today:
        return True

    history = s.get("history") or []
    if history and _parse_local_date(history[-1].get("sent_at", ""), tz) == today:
        return True

    return False


def main() -> int:
    cfg = config.load()
    if already_sent_today(cfg):
        print("Already sent a question today — skipping")
        return 0
    q = bank.next_question(cfg.questions_path, cfg.state_path)
    text = format_message.format_question(q)
    msg_id = telegram_client.send_message(
        cfg.telegram_bot_token, cfg.telegram_chat_id, text
    )
    bank.mark_pending(cfg.state_path, question_id=q["id"], telegram_message_id=msg_id)
    print(f"Sent question {q['id']} (message_id={msg_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
