import json
from unittest.mock import patch, Mock
from pathlib import Path

from src import listener, state, bank


def _seed_with_pending(tmp_path: Path):
    qs = [{"id": "q1", "category": "DCF", "difficulty": "fundamental",
           "question": "Walk through a DCF.",
           "rubric": {"must_hit": ["cash flows"]}}]
    qp = tmp_path / "questions.json"
    qp.write_text(json.dumps(qs))
    sp = tmp_path / "state.json"
    bank.mark_pending(sp, question_id="q1", telegram_message_id=99)
    return qp, sp


VOICE_UPDATE = {
    "update_id": 500,
    "message": {"message_id": 1, "voice": {"file_id": "VOICE_ID", "duration": 30}},
}

TEXT_UPDATE = {
    "update_id": 501,
    "message": {"message_id": 2, "text": "hello"},
}


@patch("src.listener.telegram_client.send_message")
@patch("src.listener.grade.grade", return_value={
    "score": 80, "letter": "B", "nailed": ["x"], "missed": [], "feedback": "ok"})
@patch("src.listener.transcribe.whisper", return_value="my answer")
@patch("src.listener.telegram_client.download_voice", return_value=b"oggbytes")
@patch("src.listener.config.load")
def test_handle_voice_full_pipeline(mock_cfg, mock_dl, mock_whisper,
                                    mock_grade, mock_send, tmp_path):
    qp, sp = _seed_with_pending(tmp_path)
    mock_cfg.return_value.telegram_bot_token = "TOK"
    mock_cfg.return_value.telegram_chat_id = "CHAT"
    mock_cfg.return_value.openai_api_key = "sk-1"
    mock_cfg.return_value.anthropic_api_key = "sk-ant-1"
    mock_cfg.return_value.questions_path = qp
    mock_cfg.return_value.state_path = sp

    listener.handle_update(mock_cfg.return_value, VOICE_UPDATE)

    mock_dl.assert_called_once_with("TOK", "VOICE_ID")
    mock_whisper.assert_called_once()
    mock_grade.assert_called_once()
    mock_send.assert_called_once()
    sent_text = mock_send.call_args[0][2]
    assert "80/100" in sent_text

    s = state.load(sp)
    assert s["pending"] is None
    assert s["history"][-1]["transcript"] == "my answer"


@patch("src.listener.telegram_client.send_message")
@patch("src.listener.config.load")
def test_handle_text_reply_prompts_for_voice(mock_cfg, mock_send, tmp_path):
    qp, sp = _seed_with_pending(tmp_path)
    mock_cfg.return_value.telegram_bot_token = "TOK"
    mock_cfg.return_value.telegram_chat_id = "CHAT"
    mock_cfg.return_value.questions_path = qp
    mock_cfg.return_value.state_path = sp

    listener.handle_update(mock_cfg.return_value, TEXT_UPDATE)

    sent_text = mock_send.call_args[0][2]
    assert "voice message" in sent_text.lower()


@patch("src.listener.telegram_client.send_message")
@patch("src.listener.config.load")
def test_handle_voice_with_no_pending_question(mock_cfg, mock_send, tmp_path):
    # No pending — empty state
    qp = tmp_path / "questions.json"
    qp.write_text("[]")
    sp = tmp_path / "state.json"
    mock_cfg.return_value.telegram_bot_token = "TOK"
    mock_cfg.return_value.telegram_chat_id = "CHAT"
    mock_cfg.return_value.questions_path = qp
    mock_cfg.return_value.state_path = sp

    listener.handle_update(mock_cfg.return_value, VOICE_UPDATE)

    sent_text = mock_send.call_args[0][2]
    assert "no active question" in sent_text.lower()


@patch("src.listener.telegram_client.send_message")
@patch("src.listener.transcribe.whisper", side_effect=Exception("API down"))
@patch("src.listener.telegram_client.download_voice", return_value=b"oggbytes")
@patch("src.listener.config.load")
def test_handle_voice_transcription_failure_does_not_mark_answered(
    mock_cfg, mock_dl, mock_whisper, mock_send, tmp_path
):
    qp, sp = _seed_with_pending(tmp_path)
    mock_cfg.return_value.telegram_bot_token = "TOK"
    mock_cfg.return_value.telegram_chat_id = "CHAT"
    mock_cfg.return_value.openai_api_key = "sk-1"
    mock_cfg.return_value.anthropic_api_key = "sk-ant-1"
    mock_cfg.return_value.questions_path = qp
    mock_cfg.return_value.state_path = sp

    listener.handle_update(mock_cfg.return_value, VOICE_UPDATE)

    sent_text = mock_send.call_args[0][2]
    assert "transcribe" in sent_text.lower() or "resend" in sent_text.lower()
    s = state.load(sp)
    assert s["pending"] is not None  # still pending


@patch("src.listener.telegram_client.send_message")
@patch("src.listener.telegram_client.get_updates")
@patch("src.listener.config.load")
def test_run_once_advances_offset_past_processed_update(
    mock_cfg, mock_get, mock_send, tmp_path
):
    """offset should advance to max(update_id)+1 even when nothing matches."""
    qp = tmp_path / "questions.json"
    qp.write_text("[]")
    sp = tmp_path / "state.json"
    mock_cfg.return_value.telegram_bot_token = "TOK"
    mock_cfg.return_value.telegram_chat_id = "CHAT"
    mock_cfg.return_value.questions_path = qp
    mock_cfg.return_value.state_path = sp

    mock_get.return_value = [{"update_id": 100, "message": {"text": "ignored"}}]

    listener.run_once(mock_cfg.return_value)

    s = state.load(sp)
    assert s["telegram_offset"] == 101


def _seed_with_history(tmp_path):
    """Seed: q1 already answered, no pending."""
    qs = [{"id": "q1", "category": "DCF", "difficulty": "fundamental",
           "question": "Walk through a DCF.",
           "rubric": {"must_hit": ["cash flows"]}}]
    qp = tmp_path / "questions.json"
    qp.write_text(json.dumps(qs))
    sp = tmp_path / "state.json"
    grade_payload = {"score": 72, "letter": "C-", "nailed": [], "missed": [], "feedback": "ok"}
    bank.mark_pending(sp, question_id="q1", telegram_message_id=99)
    bank.record_response(sp, transcript="t", grade=grade_payload)
    return qp, sp


@patch("src.listener.telegram_client.send_message", return_value=200)
@patch("src.listener.config.load")
def test_regrade_with_no_pending_marks_history_question_pending(
    mock_cfg, mock_send, tmp_path
):
    qp, sp = _seed_with_history(tmp_path)
    mock_cfg.return_value.telegram_bot_token = "TOK"
    mock_cfg.return_value.telegram_chat_id = "CHAT"
    mock_cfg.return_value.questions_path = qp
    mock_cfg.return_value.state_path = sp

    listener.handle_update(mock_cfg.return_value, {
        "update_id": 1, "message": {"text": "regrade"},
    })

    sent_text = mock_send.call_args[0][2]
    assert "Regrading" in sent_text
    assert "72/100" in sent_text  # prior grade surfaced
    assert "Walk through a DCF." in sent_text

    s = state.load(sp)
    assert s["pending"]["question_id"] == "q1"
    assert s["pending"]["telegram_message_id"] == 200


@patch("src.listener.telegram_client.send_message")
@patch("src.listener.config.load")
def test_regrade_with_pending_re_presents_without_state_change(
    mock_cfg, mock_send, tmp_path
):
    qp, sp = _seed_with_pending(tmp_path)  # creates pending q1
    mock_cfg.return_value.telegram_bot_token = "TOK"
    mock_cfg.return_value.telegram_chat_id = "CHAT"
    mock_cfg.return_value.questions_path = qp
    mock_cfg.return_value.state_path = sp

    pending_before = state.load(sp)["pending"]

    listener.handle_update(mock_cfg.return_value, {
        "update_id": 2, "message": {"text": "/regrade"},
    })

    sent_text = mock_send.call_args[0][2]
    assert "Re-presenting" in sent_text

    pending_after = state.load(sp)["pending"]
    assert pending_before == pending_after  # unchanged


@patch("src.listener.telegram_client.send_message")
@patch("src.listener.config.load")
def test_regrade_with_empty_state_replies_no_question(
    mock_cfg, mock_send, tmp_path
):
    qp = tmp_path / "questions.json"; qp.write_text("[]")
    sp = tmp_path / "state.json"
    mock_cfg.return_value.telegram_bot_token = "TOK"
    mock_cfg.return_value.telegram_chat_id = "CHAT"
    mock_cfg.return_value.questions_path = qp
    mock_cfg.return_value.state_path = sp

    listener.handle_update(mock_cfg.return_value, {
        "update_id": 3, "message": {"text": "Regrade"},  # case-insensitive
    })

    sent_text = mock_send.call_args[0][2]
    assert "No previous question" in sent_text


@patch("src.listener.telegram_client.send_message")
@patch("src.listener.config.load")
def test_non_regrade_text_still_prompts_for_voice(mock_cfg, mock_send, tmp_path):
    qp, sp = _seed_with_pending(tmp_path)
    mock_cfg.return_value.telegram_bot_token = "TOK"
    mock_cfg.return_value.telegram_chat_id = "CHAT"
    mock_cfg.return_value.questions_path = qp
    mock_cfg.return_value.state_path = sp

    listener.handle_update(mock_cfg.return_value, {
        "update_id": 4, "message": {"text": "hello there"},
    })

    sent_text = mock_send.call_args[0][2]
    assert "voice message" in sent_text.lower()
    assert "regrade" in sent_text.lower()  # the helpful hint
