import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src import state

RECENT_DAYS = 30


def _load_questions(questions_path: Path) -> list[dict]:
    return json.loads(questions_path.read_text())


def _recent_ids(history: list[dict]) -> set[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)
    ids = set()
    for h in history:
        try:
            t = datetime.fromisoformat(h["responded_at"])
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if t >= cutoff:
                ids.add(h["question_id"])
        except (KeyError, ValueError):
            continue
    return ids


def _category_weight(question: dict, last_categories: list[str]) -> float:
    if question["category"] in last_categories[-2:]:
        return 0.25
    if question["category"] in last_categories[-3:]:
        return 0.5
    return 1.0


def next_question(questions_path: Path, state_path: Path) -> dict:
    questions = _load_questions(questions_path)
    s = state.load(state_path)
    recent = _recent_ids(s["history"])

    candidates = [q for q in questions if q["id"] not in recent]
    if not candidates:
        # All asked recently — pick the one with the oldest responded_at
        by_oldest = sorted(
            s["history"], key=lambda h: h.get("responded_at", "")
        )
        oldest_id = by_oldest[0]["question_id"]
        return next(q for q in questions if q["id"] == oldest_id)

    last_cats = [h.get("category") or
                 next((q["category"] for q in questions if q["id"] == h["question_id"]), "")
                 for h in s["history"][-3:]]
    weights = [_category_weight(q, last_cats) for q in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]


def mark_pending(state_path: Path, question_id: str, telegram_message_id: int) -> None:
    with state.mutate(state_path) as s:
        s["pending"] = {
            "question_id": question_id,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "telegram_message_id": telegram_message_id,
        }


def current_pending(questions_path: Path, state_path: Path) -> Optional[dict]:
    s = state.load(state_path)
    if not s.get("pending"):
        return None
    pending = s["pending"]
    questions = _load_questions(questions_path)
    q = next((q for q in questions if q["id"] == pending["question_id"]), None)
    if q is None:
        return None
    return {
        "question_id": q["id"],
        "question": q["question"],
        "category": q["category"],
        "rubric": q["rubric"],
        "sent_at": pending["sent_at"],
    }


def previous_answered(questions_path: Path, state_path: Path) -> Optional[dict]:
    """Return the most recent answered question with its rubric and the prior grade.
    None if history is empty or the question no longer exists in the bank."""
    s = state.load(state_path)
    history = s.get("history") or []
    if not history:
        return None
    last = history[-1]
    questions = _load_questions(questions_path)
    q = next((q for q in questions if q["id"] == last["question_id"]), None)
    if q is None:
        return None
    return {
        "question_id": q["id"],
        "question": q["question"],
        "category": q["category"],
        "rubric": q["rubric"],
        "prior_grade": last.get("grade"),
    }


def record_response(state_path: Path, transcript: str, grade: dict) -> None:
    with state.mutate(state_path) as s:
        if not s.get("pending"):
            return
        entry = dict(s["pending"])
        entry["responded_at"] = datetime.now(timezone.utc).isoformat()
        entry["transcript"] = transcript
        entry["grade"] = grade
        s["history"].append(entry)
        s["pending"] = None
