import json
from unittest.mock import patch
from pathlib import Path

from src import send_question, state


def _seed(tmp_path: Path):
    qs = [{"id": "q1", "category": "DCF", "difficulty": "fundamental",
           "question": "Walk through a DCF.", "rubric": {"must_hit": []}}]
    qp = tmp_path / "questions.json"
    qp.write_text(json.dumps(qs))
    sp = tmp_path / "state.json"
    return qp, sp


@patch("src.send_question.telegram_client.send_message", return_value=12345)
@patch("src.send_question.config.load")
def test_send_question_picks_sends_and_marks_pending(mock_cfg, mock_send, tmp_path):
    qp, sp = _seed(tmp_path)
    mock_cfg.return_value.telegram_bot_token = "TOK"
    mock_cfg.return_value.telegram_chat_id = "CHAT"
    mock_cfg.return_value.questions_path = qp
    mock_cfg.return_value.state_path = sp

    send_question.main()

    mock_send.assert_called_once()
    args, _ = mock_send.call_args
    assert args[0] == "TOK" and args[1] == "CHAT"
    assert "Walk through a DCF." in args[2]

    s = state.load(sp)
    assert s["pending"]["question_id"] == "q1"
    assert s["pending"]["telegram_message_id"] == 12345
