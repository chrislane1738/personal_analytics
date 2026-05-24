import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src import bank, state


def make_questions(tmp_path: Path) -> Path:
    qs = [
        {"id": "q1", "category": "DCF", "difficulty": "fundamental",
         "question": "Walk me through a DCF.", "rubric": {"must_hit": []}},
        {"id": "q2", "category": "DCF", "difficulty": "fundamental",
         "question": "How do you calc WACC?", "rubric": {"must_hit": []}},
        {"id": "q3", "category": "M&A", "difficulty": "fundamental",
         "question": "Walk me through a merger model.", "rubric": {"must_hit": []}},
        {"id": "q4", "category": "LBO", "difficulty": "fundamental",
         "question": "Walk me through an LBO.", "rubric": {"must_hit": []}},
    ]
    p = tmp_path / "questions.json"
    p.write_text(json.dumps(qs))
    return p


def test_next_question_picks_from_bank(tmp_path):
    qp = make_questions(tmp_path)
    sp = tmp_path / "state.json"
    q = bank.next_question(qp, sp)
    assert q["id"] in {"q1", "q2", "q3", "q4"}


def test_next_question_excludes_last_30_days(tmp_path):
    qp = make_questions(tmp_path)
    sp = tmp_path / "state.json"
    recent = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    state.save(sp, {
        "telegram_offset": 0, "pending": None,
        "history": [
            {"question_id": "q1", "sent_at": recent, "responded_at": recent,
             "transcript": "", "grade": {}},
            {"question_id": "q2", "sent_at": recent, "responded_at": recent,
             "transcript": "", "grade": {}},
        ],
    })
    picks = {bank.next_question(qp, sp)["id"] for _ in range(50)}
    assert picks.issubset({"q3", "q4"})


def test_next_question_resets_when_all_recent(tmp_path):
    qp = make_questions(tmp_path)
    sp = tmp_path / "state.json"
    now = datetime.now(timezone.utc)
    history = [
        {"question_id": f"q{i}", "sent_at": (now - timedelta(days=i)).isoformat(),
         "responded_at": (now - timedelta(days=i)).isoformat(),
         "transcript": "", "grade": {}}
        for i in range(1, 5)
    ]
    state.save(sp, {"telegram_offset": 0, "pending": None, "history": history})
    q = bank.next_question(qp, sp)
    assert q["id"] == "q4"  # oldest responded_at


def test_mark_pending_sets_pending(tmp_path):
    qp = make_questions(tmp_path)
    sp = tmp_path / "state.json"
    bank.mark_pending(sp, question_id="q1", telegram_message_id=99)
    s = state.load(sp)
    assert s["pending"]["question_id"] == "q1"
    assert s["pending"]["telegram_message_id"] == 99
    assert "sent_at" in s["pending"]


def test_current_pending_returns_full_question(tmp_path):
    qp = make_questions(tmp_path)
    sp = tmp_path / "state.json"
    bank.mark_pending(sp, question_id="q1", telegram_message_id=99)
    p = bank.current_pending(qp, sp)
    assert p["question_id"] == "q1"
    assert p["question"] == "Walk me through a DCF."
    assert "rubric" in p


def test_current_pending_none_when_empty(tmp_path):
    qp = make_questions(tmp_path)
    sp = tmp_path / "state.json"
    assert bank.current_pending(qp, sp) is None


def test_record_response_clears_pending_and_appends_history(tmp_path):
    qp = make_questions(tmp_path)
    sp = tmp_path / "state.json"
    bank.mark_pending(sp, question_id="q1", telegram_message_id=99)
    grade = {"score": 80, "letter": "B", "nailed": [], "missed": [], "feedback": "ok"}
    bank.record_response(sp, transcript="hello", grade=grade)
    s = state.load(sp)
    assert s["pending"] is None
    assert len(s["history"]) == 1
    assert s["history"][0]["question_id"] == "q1"
    assert s["history"][0]["transcript"] == "hello"
    assert s["history"][0]["grade"] == grade


def test_previous_answered_none_when_history_empty(tmp_path):
    qp = make_questions(tmp_path)
    sp = tmp_path / "state.json"
    assert bank.previous_answered(qp, sp) is None


def test_previous_answered_returns_last_with_prior_grade(tmp_path):
    qp = make_questions(tmp_path)
    sp = tmp_path / "state.json"
    grade1 = {"score": 70, "letter": "C-", "nailed": [], "missed": [], "feedback": "x"}
    grade2 = {"score": 88, "letter": "B+", "nailed": [], "missed": [], "feedback": "y"}
    bank.mark_pending(sp, question_id="q1", telegram_message_id=1)
    bank.record_response(sp, transcript="t1", grade=grade1)
    bank.mark_pending(sp, question_id="q2", telegram_message_id=2)
    bank.record_response(sp, transcript="t2", grade=grade2)

    prev = bank.previous_answered(qp, sp)
    assert prev["question_id"] == "q2"
    assert prev["question"] == "How do you calc WACC?"
    assert prev["prior_grade"] == grade2
    assert "rubric" in prev
