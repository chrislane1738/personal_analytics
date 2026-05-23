import sys

from src import bank, config, format_message, telegram_client


def main() -> int:
    cfg = config.load()
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
