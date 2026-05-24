import json
from unittest.mock import patch, call
from pathlib import Path

from src import bank, listener, send_question, state


QUESTIONS = [
    {"id": "wacc", "category": "DCF", "difficulty": "fundamental",
     "question": "How do you calc WACC?",
     "rubric": {
         "must_hit": ["WACC formula", "after-tax cost of debt", "CAPM"],
         "good_to_hit": ["lever/unlever beta"],
         "common_pitfalls": ["book values"],
         "scoring_weights": {"must_hit": 0.65, "good_to_hit": 0.20, "clarity_structure": 0.15},
     }},
]


def _cfg(tmp_path: Path):
    qp = tmp_path / "questions.json"; qp.write_text(json.dumps(QUESTIONS))
    sp = tmp_path / "state.json"
    cfg = type("C", (), {})()
    cfg.telegram_bot_token = "TOK"
    cfg.telegram_chat_id   = "CHAT"
    cfg.openai_api_key     = "sk-1"
    cfg.anthropic_api_key  = "sk-ant-1"
    cfg.questions_path     = qp
    cfg.state_path         = sp
    cfg.tz                 = "America/Los_Angeles"
    return cfg


@patch("src.listener.grade.grade", return_value={
    "score": 82, "letter": "B+",
    "nailed": ["Got WACC formula", "Mentioned CAPM"],
    "missed": ["No tax shield explanation"],
    "feedback": "Strong skeleton. Tighten the cost-of-debt section."})
@patch("src.listener.transcribe.whisper", return_value="WACC is the weighted average...")
@patch("src.listener.telegram_client.download_voice", return_value=b"oggbytes")
@patch("src.telegram_client.send_message", side_effect=[4421, 1])
@patch("src.send_question.config.load")
@patch("src.listener.config.load")
def test_full_round_trip(mock_lcfg, mock_scfg, mock_send, mock_dl,
                         mock_whisper, mock_grade, tmp_path):
    cfg = _cfg(tmp_path)
    mock_lcfg.return_value = cfg
    mock_scfg.return_value = cfg

    # 1. Send the morning question
    send_question.main()
    q_send_text = mock_send.call_args_list[0][0][2]
    assert "How do you calc WACC?" in q_send_text

    s = state.load(cfg.state_path)
    assert s["pending"]["question_id"] == "wacc"

    # 2. User replies with voice
    listener.handle_update(cfg, {
        "update_id": 1,
        "message": {"voice": {"file_id": "F1", "duration": 45}},
    })

    # 3. Grade reply was sent
    sent_text = mock_send.call_args_list[1][0][2]
    assert "82/100" in sent_text
    assert "B+" in sent_text
    assert "Got WACC formula" in sent_text
    assert "No tax shield" in sent_text

    # 4. State persisted correctly
    s = state.load(cfg.state_path)
    assert s["pending"] is None
    assert s["history"][-1]["transcript"] == "WACC is the weighted average..."
    assert s["history"][-1]["grade"]["score"] == 82
