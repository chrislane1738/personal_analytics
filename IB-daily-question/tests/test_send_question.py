import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from pathlib import Path

from src import bank, send_question, state


def _seed(tmp_path: Path):
    qs = [{"id": "q1", "category": "DCF", "difficulty": "fundamental",
           "question": "Walk through a DCF.", "rubric": {"must_hit": []}}]
    qp = tmp_path / "questions.json"
    qp.write_text(json.dumps(qs))
    sp = tmp_path / "state.json"
    return qp, sp


def _mock_cfg(mock_cfg, qp, sp):
    mock_cfg.return_value.telegram_bot_token = "TOK"
    mock_cfg.return_value.telegram_chat_id = "CHAT"
    mock_cfg.return_value.questions_path = qp
    mock_cfg.return_value.state_path = sp
    mock_cfg.return_value.tz = "America/Los_Angeles"


@patch("src.send_question.telegram_client.send_message", return_value=12345)
@patch("src.send_question.config.load")
def test_send_question_picks_sends_and_marks_pending(mock_cfg, mock_send, tmp_path):
    qp, sp = _seed(tmp_path)
    _mock_cfg(mock_cfg, qp, sp)

    send_question.main()

    mock_send.assert_called_once()
    args, _ = mock_send.call_args
    assert args[0] == "TOK" and args[1] == "CHAT"
    assert "Walk through a DCF." in args[2]

    s = state.load(sp)
    assert s["pending"]["question_id"] == "q1"
    assert s["pending"]["telegram_message_id"] == 12345


@patch("src.send_question.telegram_client.send_message")
@patch("src.send_question.config.load")
def test_send_noop_when_pending_was_set_today(mock_cfg, mock_send, tmp_path):
    qp, sp = _seed(tmp_path)
    _mock_cfg(mock_cfg, qp, sp)
    bank.mark_pending(sp, question_id="q1", telegram_message_id=1)  # sent_at = now UTC

    rc = send_question.main()

    assert rc == 0
    mock_send.assert_not_called()


@patch("src.send_question.telegram_client.send_message")
@patch("src.send_question.config.load")
def test_send_noop_when_latest_history_is_today(mock_cfg, mock_send, tmp_path):
    qp, sp = _seed(tmp_path)
    _mock_cfg(mock_cfg, qp, sp)
    bank.mark_pending(sp, question_id="q1", telegram_message_id=1)
    bank.record_response(sp, transcript="t",
                         grade={"score": 80, "letter": "B", "nailed": [], "missed": [], "feedback": "ok"})

    rc = send_question.main()

    assert rc == 0
    mock_send.assert_not_called()


@patch("src.send_question.telegram_client.send_message", return_value=42)
@patch("src.send_question.config.load")
def test_send_fires_when_latest_history_is_yesterday(mock_cfg, mock_send, tmp_path):
    qp, sp = _seed(tmp_path)
    _mock_cfg(mock_cfg, qp, sp)
    # Hand-craft a history entry with a sent_at well in the past
    yesterday = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    state.save(sp, {
        "telegram_offset": 0, "pending": None,
        "history": [{
            "question_id": "q1", "sent_at": yesterday, "responded_at": yesterday,
            "transcript": "t", "grade": {"score": 75, "letter": "C", "nailed": [], "missed": [], "feedback": "ok"},
        }],
    })

    rc = send_question.main()

    assert rc == 0
    mock_send.assert_called_once()
