"""End-to-end manual test using real APIs.

Sends a known question to your Telegram, waits for a real voice reply,
runs the real Whisper + Haiku pipeline once, prints everything, then exits.

Usage:
    python -m scripts.smoke_test
"""
import sys
import time

from src import bank, config, format_message, grade, state, telegram_client, transcribe


def main() -> int:
    cfg = config.load()
    print(f"Loaded config. Questions: {cfg.questions_path}, State: {cfg.state_path}")

    # 1. Pick + send a question (does not mutate state.pending — uses a throwaway send)
    q = bank.next_question(cfg.questions_path, cfg.state_path)
    print(f"\nPicked question: [{q['category']}] {q['question']}")
    text = format_message.format_question(q)
    msg_id = telegram_client.send_message(cfg.telegram_bot_token, cfg.telegram_chat_id, text)
    print(f"Sent (message_id={msg_id}). Now record a voice reply in Telegram...")

    # Mark pending so the listener pipeline below has context
    bank.mark_pending(cfg.state_path, question_id=q["id"], telegram_message_id=msg_id)

    # 2. Long-poll once for the voice reply (max 5 min wait)
    s = state.load(cfg.state_path)
    offset = s.get("telegram_offset", 0)
    deadline = time.time() + 300
    voice_update = None
    while time.time() < deadline and voice_update is None:
        updates = telegram_client.get_updates(cfg.telegram_bot_token, offset=offset, timeout=30)
        for u in updates:
            offset = max(offset, u["update_id"] + 1)
            msg = u.get("message") or {}
            if msg.get("voice"):
                voice_update = u
                break

    if voice_update is None:
        print("Timed out waiting for voice message.")
        return 1

    with state.mutate(cfg.state_path) as st:
        st["telegram_offset"] = offset

    # 3. Transcribe
    voice = voice_update["message"]["voice"]
    print(f"\nGot voice ({voice.get('duration')}s). Downloading...")
    audio = telegram_client.download_voice(cfg.telegram_bot_token, voice["file_id"])
    print(f"Downloaded {len(audio)} bytes. Transcribing...")
    transcript = transcribe.whisper(audio, api_key=cfg.openai_api_key)
    print(f"\nTranscript:\n{transcript}\n")

    # 4. Grade
    print("Grading with Haiku...")
    result = grade.grade(
        question=q["question"], rubric=q["rubric"],
        transcript=transcript, api_key=cfg.anthropic_api_key,
    )
    print(f"\nGrade result:\n{result}\n")

    # 5. Send result back
    telegram_client.send_message(
        cfg.telegram_bot_token, cfg.telegram_chat_id,
        format_message.format_result(result),
    )
    bank.record_response(cfg.state_path, transcript=transcript, grade=result)
    print("✓ Sent grade to Telegram. Smoke test complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
