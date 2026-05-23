"""Helper to discover your Telegram chat_id after starting the bot.

Steps:
  1. Create a bot with @BotFather, get TELEGRAM_BOT_TOKEN, paste into .env.
  2. Open Telegram, search for your bot, send it any message (e.g., "hi").
  3. Run: python -m scripts.get_chat_id
  4. Paste the printed chat_id into .env as TELEGRAM_CHAT_ID.
"""
import os
import sys
import requests
from dotenv import load_dotenv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Set TELEGRAM_BOT_TOKEN in .env first.", file=sys.stderr)
        return 1
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
    r.raise_for_status()
    updates = r.json().get("result", [])
    if not updates:
        print("No updates yet — send a message to your bot in Telegram, then re-run.")
        return 1
    seen = set()
    for u in updates:
        msg = u.get("message") or {}
        chat = msg.get("chat") or {}
        cid = chat.get("id")
        if cid and cid not in seen:
            seen.add(cid)
            name = chat.get("username") or chat.get("first_name") or "(unknown)"
            print(f"chat_id={cid}  user={name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
