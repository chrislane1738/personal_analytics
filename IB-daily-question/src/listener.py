import sys
import time
import traceback

from src import bank, config, format_message, grade, state, telegram_client, transcribe


def handle_update(cfg, update: dict) -> None:
    msg = update.get("message") or {}
    voice = msg.get("voice")

    if voice is None:
        if "text" in msg:
            telegram_client.send_message(
                cfg.telegram_bot_token, cfg.telegram_chat_id,
                "Please answer with a voice message — record like you're talking to an interviewer.",
            )
        return

    pending = bank.current_pending(cfg.questions_path, cfg.state_path)
    if pending is None:
        telegram_client.send_message(
            cfg.telegram_bot_token, cfg.telegram_chat_id,
            "No active question — next one drops at 10 AM.",
        )
        return

    try:
        audio = telegram_client.download_voice(cfg.telegram_bot_token, voice["file_id"])
    except Exception:
        telegram_client.send_message(
            cfg.telegram_bot_token, cfg.telegram_chat_id,
            "Couldn't download your voice message — please resend.",
        )
        return

    try:
        transcript = transcribe.whisper(audio, api_key=cfg.openai_api_key)
    except Exception:
        telegram_client.send_message(
            cfg.telegram_bot_token, cfg.telegram_chat_id,
            "Couldn't transcribe your voice — please resend the voice message.",
        )
        return

    try:
        result = grade.grade(
            question=pending["question"],
            rubric=pending["rubric"],
            transcript=transcript,
            api_key=cfg.anthropic_api_key,
        )
    except Exception:
        telegram_client.send_message(
            cfg.telegram_bot_token, cfg.telegram_chat_id,
            "Grader hiccup — please resend the voice in a minute.",
        )
        return

    telegram_client.send_message(
        cfg.telegram_bot_token, cfg.telegram_chat_id,
        format_message.format_result(result),
    )
    bank.record_response(cfg.state_path, transcript=transcript, grade=result)


def run_once(cfg) -> None:
    """One poll cycle. Used by run() and tests."""
    s = state.load(cfg.state_path)
    offset = s.get("telegram_offset", 0)
    updates = telegram_client.get_updates(cfg.telegram_bot_token, offset=offset, timeout=30)
    for update in updates:
        try:
            handle_update(cfg, update)
        except Exception:
            traceback.print_exc()
        finally:
            with state.mutate(cfg.state_path) as st:
                st["telegram_offset"] = max(st.get("telegram_offset", 0), update["update_id"] + 1)


def run() -> int:
    cfg = config.load()
    print("listener started", flush=True)
    while True:
        try:
            run_once(cfg)
        except Exception:
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    sys.exit(run() or 0)
