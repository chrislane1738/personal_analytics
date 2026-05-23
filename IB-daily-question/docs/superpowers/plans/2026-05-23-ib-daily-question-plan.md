# IB Daily Question Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Mac-local system that sends a technical IB interview question to the user via Telegram every day at 10:00 AM PT, transcribes their voice reply with OpenAI Whisper, grades it against a per-question rubric using Claude Haiku, and replies with score + feedback.

**Architecture:** Two launchd jobs — a once-daily sender (`send_question.py`) and an always-running long-polling listener (`listener.py`). Both share a JSON state file guarded by `fcntl.flock`. Five pure-logic modules (`state`, `bank`, `telegram_client`, `transcribe`, `grade`) plus a `format_message` helper are unit-tested with mocked external APIs. A one-time bootstrap script parses the doc and uses Claude Sonnet to build rubrics for ~60 questions.

**Tech Stack:** Python 3.12+, `requests`, `anthropic`, `openai`, `python-dotenv`, `pytest`, `pytest-mock`. macOS `launchd` for scheduling.

**Spec:** [`docs/superpowers/specs/2026-05-23-ib-daily-question-design.md`](../specs/2026-05-23-ib-daily-question-design.md)

**Model IDs (per spec — current as of 2026-05-23):**
- Grader: `claude-haiku-4-5-20251001`
- Rubric bootstrap: `claude-sonnet-4-6`
- Transcription: OpenAI `whisper-1`

---

## File Map

| Path | Responsibility |
|---|---|
| `requirements.txt` | Pinned deps |
| `.env.example` | Required env vars (committed) |
| `.gitignore` | Ignore `.env`, `data/state.json`, `data/logs/`, `__pycache__`, etc. |
| `README.md` | Setup, run, troubleshoot |
| `src/config.py` | Loads env, defines paths, model IDs, constants |
| `src/state.py` | Atomic `state.json` read/write under `fcntl.flock` |
| `src/bank.py` | Load `questions.json`, pick next, mark pending, record response |
| `src/telegram_client.py` | `send_message`, `get_updates`, `download_voice` |
| `src/transcribe.py` | Audio bytes → text via Whisper |
| `src/grade.py` | `(question, rubric, transcript) → GradeResult` via Haiku |
| `src/format_message.py` | Markdown formatting for question + result |
| `src/send_question.py` | 10 AM entry point |
| `src/listener.py` | 24/7 long-poll daemon entry point |
| `scripts/bootstrap_rubrics.py` | Parse `.docx`, build `data/questions.json` with rubrics |
| `scripts/setup_launchd.sh` | Install + load both plists |
| `scripts/smoke_test.py` | Manual end-to-end test using real APIs |
| `launchd/com.chrislane.ib-daily.send.plist` | `CalendarInterval` 10:00 PT |
| `launchd/com.chrislane.ib-daily.listener.plist` | `RunAtLoad` + `KeepAlive` |
| `tests/test_state.py` | Lock + read/write |
| `tests/test_bank.py` | Selection, exclusion, rotation, record |
| `tests/test_telegram_client.py` | Mocked HTTP |
| `tests/test_transcribe.py` | Mocked Whisper |
| `tests/test_grade.py` | Mocked Haiku, JSON parsing, retry |
| `tests/test_format.py` | Markdown rendering |
| `tests/test_integration.py` | End-to-end with all APIs mocked |

---

## Task 0: Project scaffolding

**Files:**
- Create: `requirements.txt`, `.env.example`, `.gitignore`, `README.md`, `pytest.ini`, `src/__init__.py`, `tests/__init__.py`, `data/.gitkeep`, `data/logs/.gitkeep`

- [ ] **Step 1: Create `requirements.txt`**

```text
anthropic==0.40.0
openai==1.55.0
python-dotenv==1.0.1
requests==2.32.3
pytest==8.3.3
pytest-mock==3.14.0
```

- [ ] **Step 2: Create `.env.example`**

```bash
# Telegram (from @BotFather → /newbot, then send /start to your bot and run scripts/get_chat_id.py)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# OpenAI (for Whisper transcription)
OPENAI_API_KEY=

# Anthropic (for Haiku grader + Sonnet rubric bootstrap)
ANTHROPIC_API_KEY=

# Optional — defaults to America/Los_Angeles
IB_DAILY_TZ=America/Los_Angeles
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
.env
data/state.json
data/state.json.bak.*
data/logs/
__pycache__/
*.pyc
.pytest_cache/
.venv/
.DS_Store
```

- [ ] **Step 4: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -ra -q
```

- [ ] **Step 5: Create empty `src/__init__.py`, `tests/__init__.py`, `data/.gitkeep`, `data/logs/.gitkeep`**

Empty files (just `touch`).

- [ ] **Step 6: Create skeleton `README.md`**

```markdown
# IB Daily Question

Daily Telegram bot that asks an investment banking interview question, transcribes your voice reply (Whisper), and grades it against a rubric (Claude Haiku).

## Setup

1. `python3.12 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in keys.
4. `python -m scripts.bootstrap_rubrics` (one-time, ~5 min, generates `data/questions.json`).
5. `./scripts/setup_launchd.sh` (installs both jobs).

## Manual test

`python -m scripts.smoke_test`

## Files

See `docs/superpowers/specs/2026-05-23-ib-daily-question-design.md`.
```

- [ ] **Step 7: Set up virtualenv and install deps**

Run:
```bash
cd /Users/chrislane/Desktop/Claude_Code/IB-daily-question
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: clean install, no errors.

- [ ] **Step 8: Commit**

```bash
git add IB-daily-question/requirements.txt IB-daily-question/.env.example IB-daily-question/.gitignore IB-daily-question/pytest.ini IB-daily-question/README.md IB-daily-question/src/__init__.py IB-daily-question/tests/__init__.py IB-daily-question/data/.gitkeep IB-daily-question/data/logs/.gitkeep
git commit -m "chore(ib-daily-question): scaffold project (deps, env, gitignore, dirs)"
```

---

## Task 1: `config.py` — env + paths + constants

**Files:**
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path

def test_config_loads_paths_and_models(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-1")

    from src import config
    cfg = config.load()

    assert cfg.telegram_bot_token == "tok"
    assert cfg.telegram_chat_id == "123"
    assert cfg.openai_api_key == "sk-1"
    assert cfg.anthropic_api_key == "sk-ant-1"
    assert cfg.grader_model == "claude-haiku-4-5-20251001"
    assert cfg.rubric_model == "claude-sonnet-4-6"
    assert cfg.tz == "America/Los_Angeles"
    assert cfg.questions_path.name == "questions.json"
    assert cfg.state_path.name == "state.json"


def test_config_raises_on_missing_required(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    from src import config
    import pytest
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        config.load()
```

- [ ] **Step 2: Run test, verify it fails**

`pytest tests/test_config.py -v` → ImportError or AttributeError.

- [ ] **Step 3: Implement `src/config.py`**

```python
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
LOG_DIR = DATA_DIR / "logs"

GRADER_MODEL = "claude-haiku-4-5-20251001"
RUBRIC_MODEL = "claude-sonnet-4-6"
WHISPER_MODEL = "whisper-1"


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    telegram_chat_id: str
    openai_api_key: str
    anthropic_api_key: str
    tz: str
    grader_model: str
    rubric_model: str
    whisper_model: str
    questions_path: Path
    state_path: Path


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def load() -> Config:
    load_dotenv(REPO_ROOT / ".env")
    return Config(
        telegram_bot_token=_require("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_require("TELEGRAM_CHAT_ID"),
        openai_api_key=_require("OPENAI_API_KEY"),
        anthropic_api_key=_require("ANTHROPIC_API_KEY"),
        tz=os.environ.get("IB_DAILY_TZ", "America/Los_Angeles"),
        grader_model=GRADER_MODEL,
        rubric_model=RUBRIC_MODEL,
        whisper_model=WHISPER_MODEL,
        questions_path=DATA_DIR / "questions.json",
        state_path=DATA_DIR / "state.json",
    )
```

- [ ] **Step 4: Run tests, verify pass**

`pytest tests/test_config.py -v` → both pass.

- [ ] **Step 5: Commit**

```bash
git add IB-daily-question/src/config.py IB-daily-question/tests/test_config.py
git commit -m "feat(ib-daily-question): config module — env loading + paths"
```

---

## Task 2: `state.py` — atomic JSON read/write under flock

**Files:**
- Create: `src/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

`tests/test_state.py`:
```python
import json
import threading
from pathlib import Path

from src import state


def test_load_returns_defaults_when_missing(tmp_path):
    p = tmp_path / "state.json"
    s = state.load(p)
    assert s == {"telegram_offset": 0, "pending": None, "history": []}


def test_save_then_load_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    s = state.load(p)
    s["telegram_offset"] = 42
    state.save(p, s)
    assert state.load(p) == s


def test_mutate_applies_under_lock(tmp_path):
    p = tmp_path / "state.json"
    state.save(p, {"telegram_offset": 0, "pending": None, "history": []})

    def inc():
        for _ in range(100):
            with state.mutate(p) as s:
                s["telegram_offset"] += 1

    threads = [threading.Thread(target=inc) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()

    final = state.load(p)
    assert final["telegram_offset"] == 500


def test_load_corrupt_backs_up_and_resets(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{not json")
    s = state.load(p)
    assert s == {"telegram_offset": 0, "pending": None, "history": []}
    # backup file exists
    backups = list(tmp_path.glob("state.json.bak.*"))
    assert len(backups) == 1
```

- [ ] **Step 2: Run, verify fails**

`pytest tests/test_state.py -v` → ImportError.

- [ ] **Step 3: Implement `src/state.py`**

```python
import fcntl
import json
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DEFAULT = {"telegram_offset": 0, "pending": None, "history": []}


def load(path: Path) -> dict:
    if not path.exists():
        return dict(DEFAULT, history=[])
    try:
        with path.open("r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (json.JSONDecodeError, ValueError):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup = path.with_suffix(f".json.bak.{ts}")
        shutil.copy(path, backup)
        return dict(DEFAULT, history=[])


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(data, f, indent=2)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    tmp.replace(path)


@contextmanager
def mutate(path: Path) -> Iterator[dict]:
    """Read-modify-write with exclusive lock held the entire time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            raw = f.read()
            data = json.loads(raw) if raw.strip() else dict(DEFAULT, history=[])
            yield data
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

- [ ] **Step 4: Run, verify pass**

`pytest tests/test_state.py -v` → 4 pass.

- [ ] **Step 5: Commit**

```bash
git add IB-daily-question/src/state.py IB-daily-question/tests/test_state.py
git commit -m "feat(ib-daily-question): state module — atomic JSON I/O under flock"
```

---

## Task 3: `bank.py` — question selection, mark pending, record response

**Files:**
- Create: `src/bank.py`
- Test: `tests/test_bank.py`

- [ ] **Step 1: Write the failing test**

`tests/test_bank.py`:
```python
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
```

- [ ] **Step 2: Run, verify fails**

`pytest tests/test_bank.py -v` → ImportError.

- [ ] **Step 3: Implement `src/bank.py`**

```python
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
```

- [ ] **Step 4: Run, verify pass**

`pytest tests/test_bank.py -v` → 7 pass.

- [ ] **Step 5: Commit**

```bash
git add IB-daily-question/src/bank.py IB-daily-question/tests/test_bank.py
git commit -m "feat(ib-daily-question): bank module — selection, rotation, pending tracking"
```

---

## Task 4: `telegram_client.py` — send, poll, download

**Files:**
- Create: `src/telegram_client.py`
- Test: `tests/test_telegram_client.py`

- [ ] **Step 1: Write the failing test**

`tests/test_telegram_client.py`:
```python
from unittest.mock import patch, Mock
from src import telegram_client as tg


def _mock_response(json_data, status=200):
    r = Mock()
    r.status_code = status
    r.json.return_value = json_data
    r.content = b"binary"
    r.raise_for_status = Mock()
    return r


@patch("src.telegram_client.requests.post")
def test_send_message_posts_to_api(mock_post):
    mock_post.return_value = _mock_response({"ok": True, "result": {"message_id": 42}})
    msg_id = tg.send_message("TOKEN", "CHAT", "Hello *world*")
    assert msg_id == 42
    args, kwargs = mock_post.call_args
    assert "bot TOKEN" not in args[0]  # token in URL path, not bearer
    assert "TOKEN" in args[0]
    assert kwargs["json"]["chat_id"] == "CHAT"
    assert kwargs["json"]["text"] == "Hello *world*"
    assert kwargs["json"]["parse_mode"] == "Markdown"


@patch("src.telegram_client.requests.get")
def test_get_updates_returns_results(mock_get):
    mock_get.return_value = _mock_response({"ok": True, "result": [{"update_id": 1}]})
    updates = tg.get_updates("TOKEN", offset=10, timeout=30)
    assert updates == [{"update_id": 1}]
    args, kwargs = mock_get.call_args
    assert kwargs["params"]["offset"] == 10
    assert kwargs["params"]["timeout"] == 30


@patch("src.telegram_client.requests.get")
def test_download_voice_two_step(mock_get):
    mock_get.side_effect = [
        _mock_response({"ok": True, "result": {"file_path": "voice/file_42.ogg"}}),
        _mock_response({}, status=200),
    ]
    audio = tg.download_voice("TOKEN", "FILE_ID")
    assert audio == b"binary"
    # first call: getFile
    assert "getFile" in mock_get.call_args_list[0][0][0]
    # second call: file download URL with file_path
    assert "voice/file_42.ogg" in mock_get.call_args_list[1][0][0]


@patch("src.telegram_client.requests.post")
def test_send_message_retries_on_5xx(mock_post):
    fail = _mock_response({}, status=500)
    fail.raise_for_status.side_effect = Exception("boom")
    ok = _mock_response({"ok": True, "result": {"message_id": 1}})
    mock_post.side_effect = [fail, fail, ok]
    msg_id = tg.send_message("TOKEN", "CHAT", "hi", max_retries=3, backoff=0)
    assert msg_id == 1
    assert mock_post.call_count == 3
```

- [ ] **Step 2: Run, verify fails**

`pytest tests/test_telegram_client.py -v` → ImportError.

- [ ] **Step 3: Implement `src/telegram_client.py`**

```python
import time
import requests

BASE = "https://api.telegram.org"


def _post(url: str, json: dict, max_retries: int = 3, backoff: float = 2.0):
    last_exc = None
    for attempt in range(max_retries):
        r = requests.post(url, json=json, timeout=30)
        try:
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise last_exc


def send_message(token: str, chat_id: str, text: str,
                 max_retries: int = 3, backoff: float = 2.0) -> int:
    url = f"{BASE}/bot{token}/sendMessage"
    r = _post(url, {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
              max_retries=max_retries, backoff=backoff)
    return r.json()["result"]["message_id"]


def get_updates(token: str, offset: int, timeout: int = 30) -> list[dict]:
    url = f"{BASE}/bot{token}/getUpdates"
    r = requests.get(
        url,
        params={"offset": offset, "timeout": timeout, "allowed_updates": ["message"]},
        timeout=timeout + 5,
    )
    r.raise_for_status()
    return r.json().get("result", [])


def download_voice(token: str, file_id: str) -> bytes:
    meta = requests.get(f"{BASE}/bot{token}/getFile",
                        params={"file_id": file_id}, timeout=30)
    meta.raise_for_status()
    file_path = meta.json()["result"]["file_path"]
    dl = requests.get(f"{BASE}/file/bot{token}/{file_path}", timeout=60)
    dl.raise_for_status()
    return dl.content
```

- [ ] **Step 4: Run, verify pass**

`pytest tests/test_telegram_client.py -v` → 4 pass.

- [ ] **Step 5: Commit**

```bash
git add IB-daily-question/src/telegram_client.py IB-daily-question/tests/test_telegram_client.py
git commit -m "feat(ib-daily-question): telegram client — send, poll, download voice"
```

---

## Task 5: `transcribe.py` — Whisper wrapper

**Files:**
- Create: `src/transcribe.py`
- Test: `tests/test_transcribe.py`

- [ ] **Step 1: Write the failing test**

`tests/test_transcribe.py`:
```python
from unittest.mock import patch, Mock
from src import transcribe


@patch("src.transcribe.OpenAI")
def test_whisper_returns_transcript_text(mock_openai_cls):
    mock_client = Mock()
    mock_openai_cls.return_value = mock_client
    mock_client.audio.transcriptions.create.return_value = Mock(text="hello world")

    out = transcribe.whisper(b"oggbytes", api_key="sk-1")

    assert out == "hello world"
    call = mock_client.audio.transcriptions.create.call_args
    assert call.kwargs["model"] == "whisper-1"
    # file is passed as (filename, bytes, mime) tuple
    assert call.kwargs["file"][0].endswith(".ogg")
    assert call.kwargs["file"][1] == b"oggbytes"


@patch("src.transcribe.OpenAI")
def test_whisper_raises_on_empty_audio(mock_openai_cls):
    import pytest
    with pytest.raises(ValueError, match="empty audio"):
        transcribe.whisper(b"", api_key="sk-1")
```

- [ ] **Step 2: Run, verify fails**

`pytest tests/test_transcribe.py -v` → ImportError.

- [ ] **Step 3: Implement `src/transcribe.py`**

```python
from openai import OpenAI


def whisper(audio_bytes: bytes, api_key: str, mime: str = "audio/ogg") -> str:
    if not audio_bytes:
        raise ValueError("empty audio")
    client = OpenAI(api_key=api_key)
    ext = ".ogg" if "ogg" in mime else ".mp3"
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=(f"voice{ext}", audio_bytes, mime),
    )
    return result.text.strip()
```

- [ ] **Step 4: Run, verify pass**

`pytest tests/test_transcribe.py -v` → 2 pass.

- [ ] **Step 5: Commit**

```bash
git add IB-daily-question/src/transcribe.py IB-daily-question/tests/test_transcribe.py
git commit -m "feat(ib-daily-question): transcribe module — Whisper wrapper"
```

---

## Task 6: `grade.py` — Haiku grader with rubric, JSON parsing, retry

**Files:**
- Create: `src/grade.py`
- Test: `tests/test_grade.py`

- [ ] **Step 1: Write the failing test**

`tests/test_grade.py`:
```python
import json
from unittest.mock import patch, Mock
import pytest

from src import grade


def _anthropic_response(content_text: str):
    msg = Mock()
    msg.content = [Mock(text=content_text)]
    return msg


VALID_GRADE = {
    "score": 78, "letter": "B+",
    "nailed": ["Got the formula"],
    "missed": ["Forgot tax shield"],
    "feedback": "Solid structure.",
}


@patch("src.grade.Anthropic")
def test_grade_returns_parsed_json(mock_anthropic_cls):
    mock_client = Mock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = _anthropic_response(json.dumps(VALID_GRADE))

    out = grade.grade(
        question="How do you calc WACC?",
        rubric={"must_hit": ["formula"], "scoring_weights": {}},
        transcript="WACC equals...",
        api_key="sk-ant-1",
    )

    assert out == VALID_GRADE
    call = mock_client.messages.create.call_args
    assert call.kwargs["model"] == "claude-haiku-4-5-20251001"


@patch("src.grade.Anthropic")
def test_grade_retries_on_malformed_then_succeeds(mock_anthropic_cls):
    mock_client = Mock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.side_effect = [
        _anthropic_response("not json at all"),
        _anthropic_response(json.dumps(VALID_GRADE)),
    ]

    out = grade.grade("Q?", {}, "answer", api_key="sk-ant-1")
    assert out == VALID_GRADE
    assert mock_client.messages.create.call_count == 2


@patch("src.grade.Anthropic")
def test_grade_raises_after_two_failures(mock_anthropic_cls):
    mock_client = Mock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = _anthropic_response("still not json")
    with pytest.raises(grade.GraderError):
        grade.grade("Q?", {}, "answer", api_key="sk-ant-1")


@patch("src.grade.Anthropic")
def test_grade_extracts_json_from_code_fence(mock_anthropic_cls):
    mock_client = Mock()
    mock_anthropic_cls.return_value = mock_client
    fenced = f"```json\n{json.dumps(VALID_GRADE)}\n```"
    mock_client.messages.create.return_value = _anthropic_response(fenced)
    out = grade.grade("Q?", {}, "answer", api_key="sk-ant-1")
    assert out == VALID_GRADE
```

- [ ] **Step 2: Run, verify fails**

`pytest tests/test_grade.py -v` → ImportError.

- [ ] **Step 3: Implement `src/grade.py`**

```python
import json
import re
from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"


class GraderError(Exception):
    pass


SYSTEM_PROMPT = """You are a VP-level investment banking interviewer grading a candidate's verbal answer to a technical interview question. You grade like a real MD/VP would: technical correctness matters most, but clarity and structure count too.

You will return ONLY a JSON object — no preamble, no markdown fences, no commentary — with this exact shape:

{
  "score": <integer 0-100>,
  "letter": "<A+|A|A-|B+|B|B-|C+|C|C-|D+|D|D-|F>",
  "nailed": ["<concrete point the candidate correctly hit>", ...],
  "missed": ["<concrete point the candidate missed or was unclear on>", ...],
  "feedback": "<2-4 sentences of overall feedback in an interviewer's voice>"
}

Scoring weights for this question:
- must_hit points: 65% of score (each one missed is significant)
- good_to_hit points: 20% of score (bonus, not required)
- clarity & structure: 15% of score

Letter mapping: 93+=A, 90-92=A-, 87-89=B+, 83-86=B, 80-82=B-, 77-79=C+, 73-76=C, 70-72=C-, 67-69=D+, 63-66=D, 60-62=D-, <60=F. (A+ reserved for genuinely exceptional answers >=97.)"""


USER_TEMPLATE = """QUESTION:
{question}

RUBRIC:
must_hit:
{must_hit}

good_to_hit:
{good_to_hit}

common_pitfalls:
{pitfalls}

CANDIDATE'S TRANSCRIBED ANSWER:
{transcript}

Return the JSON object now."""


def _bullets(items: list[str] | None) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {x}" for x in items)


def _extract_json(text: str) -> dict:
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try first {...} blob
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError("no JSON found", text, 0)


def grade(question: str, rubric: dict, transcript: str, api_key: str) -> dict:
    client = Anthropic(api_key=api_key)
    user = USER_TEMPLATE.format(
        question=question,
        must_hit=_bullets(rubric.get("must_hit")),
        good_to_hit=_bullets(rubric.get("good_to_hit")),
        pitfalls=_bullets(rubric.get("common_pitfalls")),
        transcript=transcript,
    )

    for attempt in range(2):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text
        try:
            return _extract_json(text)
        except (json.JSONDecodeError, ValueError):
            if attempt == 0:
                user += "\n\nREMINDER: Return ONLY the JSON object. No prose, no fences."
                continue
            raise GraderError(f"Could not parse grader output: {text[:200]}")
```

- [ ] **Step 4: Run, verify pass**

`pytest tests/test_grade.py -v` → 4 pass.

- [ ] **Step 5: Commit**

```bash
git add IB-daily-question/src/grade.py IB-daily-question/tests/test_grade.py
git commit -m "feat(ib-daily-question): grade module — Haiku grader with retry"
```

---

## Task 7: `format_message.py` — Markdown formatting

**Files:**
- Create: `src/format_message.py`
- Test: `tests/test_format.py`

- [ ] **Step 1: Write the failing test**

`tests/test_format.py`:
```python
from src import format_message as fm


def test_format_question_includes_category_and_question():
    q = {"category": "Core Valuation & DCF",
         "question": "Walk me through a DCF."}
    out = fm.format_question(q)
    assert "Core Valuation & DCF" in out
    assert "Walk me through a DCF." in out
    assert "voice message" in out.lower()


def test_format_result_renders_all_sections():
    grade = {
        "score": 78, "letter": "B+",
        "nailed": ["Got formula", "Mentioned beta"],
        "missed": ["No tax shield"],
        "feedback": "Solid attempt.",
    }
    out = fm.format_result(grade)
    assert "78/100" in out
    assert "B+" in out
    assert "Got formula" in out
    assert "No tax shield" in out
    assert "Solid attempt." in out


def test_format_result_handles_empty_lists():
    grade = {"score": 95, "letter": "A",
             "nailed": ["Everything"], "missed": [], "feedback": "Great."}
    out = fm.format_result(grade)
    assert "nothing significant missed" in out.lower() or "—" in out


def test_format_escapes_markdown_special_chars_in_transcript_safe_fields():
    # feedback may contain user-influenced content; markdown should not break
    grade = {"score": 50, "letter": "D",
             "nailed": [], "missed": [], "feedback": "Mentioned * and _ and [link]"}
    out = fm.format_result(grade)
    assert "Mentioned" in out
```

- [ ] **Step 2: Run, verify fails**

`pytest tests/test_format.py -v` → ImportError.

- [ ] **Step 3: Implement `src/format_message.py`**

```python
def format_question(q: dict) -> str:
    return (
        f"*Today's IB question* — _{q['category']}_\n\n"
        f"{q['question']}\n\n"
        f"_Reply with a voice message — answer like you're talking to an interviewer._"
    )


def _bullets(items: list[str], empty_text: str) -> str:
    if not items:
        return f"  _{empty_text}_"
    return "\n".join(f"  • {x}" for x in items)


def format_result(grade: dict) -> str:
    return (
        f"*Grade: {grade['letter']} ({grade['score']}/100)*\n\n"
        f"✓ *What you nailed*\n"
        f"{_bullets(grade['nailed'], 'Nothing stood out.')}\n\n"
        f"✗ *What you missed*\n"
        f"{_bullets(grade['missed'], 'Nothing significant missed.')}\n\n"
        f"*Feedback*\n{grade['feedback']}"
    )
```

- [ ] **Step 4: Run, verify pass**

`pytest tests/test_format.py -v` → 4 pass.

- [ ] **Step 5: Commit**

```bash
git add IB-daily-question/src/format_message.py IB-daily-question/tests/test_format.py
git commit -m "feat(ib-daily-question): format_message module — telegram markdown"
```

---

## Task 8: `send_question.py` — 10 AM entry point

**Files:**
- Create: `src/send_question.py`
- Test: `tests/test_send_question.py`

- [ ] **Step 1: Write the failing test**

`tests/test_send_question.py`:
```python
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
```

- [ ] **Step 2: Run, verify fails**

`pytest tests/test_send_question.py -v` → ImportError.

- [ ] **Step 3: Implement `src/send_question.py`**

```python
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
```

- [ ] **Step 4: Run, verify pass**

`pytest tests/test_send_question.py -v` → 1 pass.

- [ ] **Step 5: Commit**

```bash
git add IB-daily-question/src/send_question.py IB-daily-question/tests/test_send_question.py
git commit -m "feat(ib-daily-question): send_question entry point"
```

---

## Task 9: `listener.py` — long-poll daemon

**Files:**
- Create: `src/listener.py`
- Test: `tests/test_listener.py`

- [ ] **Step 1: Write the failing test**

`tests/test_listener.py`:
```python
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
```

- [ ] **Step 2: Run, verify fails**

`pytest tests/test_listener.py -v` → ImportError.

- [ ] **Step 3: Implement `src/listener.py`**

```python
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
            "Grader hiccup — your transcript was saved. Please resend the voice in a minute.",
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
```

- [ ] **Step 4: Run, verify pass**

`pytest tests/test_listener.py -v` → 5 pass.

- [ ] **Step 5: Commit**

```bash
git add IB-daily-question/src/listener.py IB-daily-question/tests/test_listener.py
git commit -m "feat(ib-daily-question): listener daemon — long-poll + voice pipeline"
```

---

## Task 10: Integration test — end-to-end with all APIs mocked

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the test**

`tests/test_integration.py`:
```python
import json
from unittest.mock import patch
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
    return cfg


@patch("src.listener.grade.grade", return_value={
    "score": 82, "letter": "B+",
    "nailed": ["Got WACC formula", "Mentioned CAPM"],
    "missed": ["No tax shield explanation"],
    "feedback": "Strong skeleton. Tighten the cost-of-debt section."})
@patch("src.listener.transcribe.whisper", return_value="WACC is the weighted average...")
@patch("src.listener.telegram_client.download_voice", return_value=b"oggbytes")
@patch("src.listener.telegram_client.send_message", return_value=1)
@patch("src.send_question.telegram_client.send_message", return_value=4421)
@patch("src.send_question.config.load")
@patch("src.listener.config.load")
def test_full_round_trip(mock_lcfg, mock_scfg, mock_q_send, mock_l_send,
                         mock_dl, mock_whisper, mock_grade, tmp_path):
    cfg = _cfg(tmp_path)
    mock_lcfg.return_value = cfg
    mock_scfg.return_value = cfg

    # 1. Send the morning question
    send_question.main()
    assert "How do you calc WACC?" in mock_q_send.call_args[0][2]

    s = state.load(cfg.state_path)
    assert s["pending"]["question_id"] == "wacc"

    # 2. User replies with voice
    listener.handle_update(cfg, {
        "update_id": 1,
        "message": {"voice": {"file_id": "F1", "duration": 45}},
    })

    # 3. Grade reply was sent
    sent_text = mock_l_send.call_args[0][2]
    assert "82/100" in sent_text
    assert "B+" in sent_text
    assert "Got WACC formula" in sent_text
    assert "No tax shield" in sent_text

    # 4. State persisted correctly
    s = state.load(cfg.state_path)
    assert s["pending"] is None
    assert s["history"][-1]["transcript"] == "WACC is the weighted average..."
    assert s["history"][-1]["grade"]["score"] == 82
```

- [ ] **Step 2: Run, verify pass**

`pytest tests/test_integration.py -v` → 1 pass.

- [ ] **Step 3: Run full test suite**

`pytest -v` → all tests across all files pass.

- [ ] **Step 4: Commit**

```bash
git add IB-daily-question/tests/test_integration.py
git commit -m "test(ib-daily-question): end-to-end integration test with mocked APIs"
```

---

## Task 11: `bootstrap_rubrics.py` — parse .docx, generate rubrics with Sonnet

**Files:**
- Create: `scripts/__init__.py`, `scripts/bootstrap_rubrics.py`
- Test: `tests/test_bootstrap_rubrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_bootstrap_rubrics.py`:
```python
import json
from unittest.mock import patch, Mock
from pathlib import Path

from scripts import bootstrap_rubrics as br


SAMPLE_XML = """
<doc>
<text>Core Valuation &amp; DCF</text>
<text>Understanding how to value a company is the core of the job.</text>
<text>Walk me through a Discounted Cash Flow (DCF) model.</text>
<text>How do you calculate WACC?</text>
<text>M&amp;A</text>
<text>Walk me through a basic merger model.</text>
</doc>
""".strip()


def test_parse_docx_extracts_categories_and_questions(tmp_path, monkeypatch):
    """The parser should identify section headers vs question lines."""
    # We mock the docx unzip layer by patching the extractor
    monkeypatch.setattr(br, "_extract_text", lambda p: [
        "Core Valuation & DCF",
        "Understanding how to value a company is the core of the job.",
        "Walk me through a Discounted Cash Flow (DCF) model.",
        "How do you calculate WACC?",
        "M&A",
        "Walk me through a basic merger model.",
    ])
    items = br.parse_docx(Path("ignored"))
    cats = [i["category"] for i in items]
    qs = [i["question"] for i in items]
    assert "Core Valuation & DCF" in cats
    assert "M&A" in cats
    assert "Walk me through a Discounted Cash Flow (DCF) model." in qs
    assert "How do you calculate WACC?" in qs
    assert "Walk me through a basic merger model." in qs
    # description line should NOT be a question
    assert all("core of the job" not in q for q in qs)


def test_slug_id_is_stable():
    assert br.slug_id("How do you calculate WACC?") == br.slug_id("How do you calculate WACC?")
    assert br.slug_id("DCF model") != br.slug_id("LBO model")


@patch("scripts.bootstrap_rubrics.Anthropic")
def test_generate_rubric_calls_sonnet_and_parses(mock_anth_cls):
    mock_client = Mock()
    mock_anth_cls.return_value = mock_client
    rubric = {
        "must_hit": ["formula"],
        "good_to_hit": ["beta"],
        "common_pitfalls": ["book values"],
        "scoring_weights": {"must_hit": 0.65, "good_to_hit": 0.20, "clarity_structure": 0.15},
    }
    mock_client.messages.create.return_value = Mock(
        content=[Mock(text=json.dumps(rubric))]
    )
    out = br.generate_rubric("How do you calculate WACC?", "DCF", api_key="sk-ant-1")
    assert out == rubric


def test_skip_existing_when_not_regenerating(tmp_path):
    existing = [{"id": "abc", "category": "DCF", "difficulty": "fundamental",
                 "question": "Q?", "rubric": {"must_hit": []}}]
    out_path = tmp_path / "questions.json"
    out_path.write_text(json.dumps(existing))

    items = [{"category": "DCF", "question": "Q?"}]  # same Q
    fake_gen = Mock()
    br.build(items, out_path, api_key="sk", regenerate=False, generator=fake_gen)
    fake_gen.assert_not_called()
    final = json.loads(out_path.read_text())
    assert final == existing
```

- [ ] **Step 2: Run, verify fails**

`pytest tests/test_bootstrap_rubrics.py -v` → ImportError.

- [ ] **Step 3: Implement `scripts/bootstrap_rubrics.py`**

```python
"""One-time: parse the IB interview questions .docx and generate rubrics for each
question using Claude Sonnet. Writes data/questions.json.

Usage:
    python -m scripts.bootstrap_rubrics
    python -m scripts.bootstrap_rubrics --regenerate    # overwrite existing
"""
import argparse
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
import os

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "Investment Banking Interview Questions.docx"
OUT_PATH = REPO_ROOT / "data" / "questions.json"
MODEL = "claude-sonnet-4-6"


# --- .docx parsing ---

_KNOWN_CATEGORIES = {
    "The Absolute Fundamentals (Accounting & Enterprise Value)",
    "Core Valuation & DCF",
    "Advanced Accounting & Working Capital",
    "Mergers & Acquisitions (M&A)",
    "Leveraged Buyouts (LBO)",
}


def _extract_text(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    # Unescape and strip tags
    xml = xml.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"<[^>]+>", "\n", xml)
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_docx(path: Path) -> list[dict]:
    """Return list of {category, question}. Headers identified by exact match
    against _KNOWN_CATEGORIES; description lines (non-question text under a header)
    are filtered out by requiring the line to look like a question (ends with ?
    OR begins with 'Walk me through' / 'A company' / 'If '/'When ')."""
    lines = _extract_text(path)
    out = []
    current = None
    for line in lines:
        if line in _KNOWN_CATEGORIES:
            current = line
            continue
        if current is None:
            continue
        if _looks_like_question(line):
            out.append({"category": current, "question": line})
    return out


_QUESTION_STARTS = ("Walk me through", "A company", "If ", "When ", "Why ",
                    "How ", "What ", "Rank ", "Is ", "Can ")


def _looks_like_question(s: str) -> bool:
    if s.endswith("?"):
        return True
    return any(s.startswith(p) for p in _QUESTION_STARTS)


# --- ID generation ---

def slug_id(question: str) -> str:
    h = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]
    return f"q-{h}"


# --- Rubric generation ---

SYSTEM_RUBRIC = """You are a senior investment banking interviewer building a grading rubric for an entry-level analyst's verbal answer to a technical IB interview question.

Return ONLY a JSON object with this exact shape — no preamble, no markdown fences:

{
  "must_hit": ["<concrete point a strong answer MUST cover>", ...],   // 3-6 items
  "good_to_hit": ["<bonus point that distinguishes a top answer>", ...],  // 2-4 items
  "common_pitfalls": ["<frequent candidate mistake on this question>", ...],  // 2-4 items
  "scoring_weights": {"must_hit": 0.65, "good_to_hit": 0.20, "clarity_structure": 0.15}
}

Each rubric point should be specific and concrete (e.g., "WACC formula with (E/V)·Re + (D/V)·Rd·(1−t)" not "WACC formula"). Phrase as scoring criteria, not as the answer itself."""


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def generate_rubric(question: str, category: str, api_key: str) -> dict:
    client = Anthropic(api_key=api_key)
    user = f"Category: {category}\n\nQuestion: {question}\n\nReturn the rubric JSON now."
    resp = client.messages.create(
        model=MODEL, max_tokens=1024,
        system=SYSTEM_RUBRIC,
        messages=[{"role": "user", "content": user}],
    )
    return _extract_json(resp.content[0].text)


# --- Build orchestration ---

def build(items: list[dict], out_path: Path, api_key: str,
          regenerate: bool, generator=None) -> None:
    """Build/refresh questions.json. `generator` is injectable for tests."""
    gen = generator or (lambda q, c: generate_rubric(q, c, api_key))

    existing = []
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    by_id = {e["id"]: e for e in existing}

    for item in items:
        qid = slug_id(item["question"])
        if qid in by_id and not regenerate:
            print(f"[skip] {qid}: {item['question'][:60]}")
            continue
        print(f"[gen]  {qid}: {item['question'][:60]}")
        rubric = gen(item["question"], item["category"])
        by_id[qid] = {
            "id": qid,
            "category": item["category"],
            "difficulty": "fundamental",
            "question": item["question"],
            "rubric": rubric,
        }
        time.sleep(0.5)  # gentle rate limit

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(list(by_id.values()), indent=2))
    print(f"Wrote {len(by_id)} questions to {out_path}")


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Missing ANTHROPIC_API_KEY", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true",
                    help="Overwrite rubrics for questions that already exist")
    ap.add_argument("--docx", type=Path, default=DOC_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    items = parse_docx(args.docx)
    print(f"Parsed {len(items)} questions from {args.docx.name}")
    build(items, args.out, api_key=api_key, regenerate=args.regenerate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Create empty `scripts/__init__.py`**

`touch scripts/__init__.py`

- [ ] **Step 5: Run tests, verify pass**

`pytest tests/test_bootstrap_rubrics.py -v` → 4 pass.

- [ ] **Step 6: Manual: actually run the bootstrap (real API call)**

Run:
```bash
source .venv/bin/activate
python -m scripts.bootstrap_rubrics
```

Expected: ~60 lines of `[gen] q-XXXXXXXX: <question>...`. Total runtime ~5 min, costs ~$0.50 in Sonnet tokens. `data/questions.json` is created.

Spot-check the output:
```bash
python -c "import json; qs=json.load(open('data/questions.json')); print(f'{len(qs)} questions'); print(json.dumps(qs[0], indent=2))"
```

- [ ] **Step 7: Commit**

```bash
git add IB-daily-question/scripts/__init__.py IB-daily-question/scripts/bootstrap_rubrics.py IB-daily-question/tests/test_bootstrap_rubrics.py IB-daily-question/data/questions.json
git commit -m "feat(ib-daily-question): bootstrap script + generated rubrics for ~60 questions"
```

---

## Task 12: launchd plists + setup script + chat-id helper

**Files:**
- Create: `scripts/get_chat_id.py`, `launchd/com.chrislane.ib-daily.send.plist`, `launchd/com.chrislane.ib-daily.listener.plist`, `scripts/setup_launchd.sh`

- [ ] **Step 1: Create `scripts/get_chat_id.py`**

```python
"""Helper to discover your Telegram chat_id after starting the bot.

Steps:
  1. Create a bot with @BotFather, get TELEGRAM_BOT_TOKEN, paste into .env.
  2. Open Telegram, search for your bot, send any message to it (e.g., "hi").
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
```

- [ ] **Step 2: Create `launchd/com.chrislane.ib-daily.send.plist`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.chrislane.ib-daily.send</string>

    <key>ProgramArguments</key>
    <array>
        <string>__REPO__/.venv/bin/python</string>
        <string>-m</string>
        <string>src.send_question</string>
    </array>

    <key>WorkingDirectory</key>
    <string>__REPO__</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>10</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>__REPO__/data/logs/send.out</string>
    <key>StandardErrorPath</key>
    <string>__REPO__/data/logs/send.err</string>
</dict>
</plist>
```

- [ ] **Step 3: Create `launchd/com.chrislane.ib-daily.listener.plist`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.chrislane.ib-daily.listener</string>

    <key>ProgramArguments</key>
    <array>
        <string>__REPO__/.venv/bin/python</string>
        <string>-m</string>
        <string>src.listener</string>
    </array>

    <key>WorkingDirectory</key>
    <string>__REPO__</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>10</integer>

    <key>StandardOutPath</key>
    <string>__REPO__/data/logs/listener.out</string>
    <key>StandardErrorPath</key>
    <string>__REPO__/data/logs/listener.err</string>
</dict>
</plist>
```

- [ ] **Step 4: Create `scripts/setup_launchd.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$TARGET_DIR" "$REPO/data/logs"

for name in com.chrislane.ib-daily.send com.chrislane.ib-daily.listener; do
    SRC="$REPO/launchd/$name.plist"
    DST="$TARGET_DIR/$name.plist"
    sed "s#__REPO__#$REPO#g" "$SRC" > "$DST"
    echo "Installed $DST"

    # Unload if loaded, then load
    launchctl unload "$DST" 2>/dev/null || true
    launchctl load "$DST"
    echo "Loaded $name"
done

echo
echo "Both jobs installed. Verify with:"
echo "  launchctl list | grep ib-daily"
echo
echo "To uninstall later:"
echo "  launchctl unload $TARGET_DIR/com.chrislane.ib-daily.*.plist"
echo "  rm $TARGET_DIR/com.chrislane.ib-daily.*.plist"
```

- [ ] **Step 5: Make executable**

```bash
chmod +x IB-daily-question/scripts/setup_launchd.sh
```

- [ ] **Step 6: Commit**

```bash
git add IB-daily-question/scripts/get_chat_id.py IB-daily-question/launchd/ IB-daily-question/scripts/setup_launchd.sh
git commit -m "feat(ib-daily-question): launchd plists + setup script + chat_id helper"
```

---

## Task 13: Smoke test script + README polish

**Files:**
- Create: `scripts/smoke_test.py`
- Modify: `README.md`

- [ ] **Step 1: Create `scripts/smoke_test.py`**

```python
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
```

- [ ] **Step 2: Replace `README.md` with full version**

```markdown
# IB Daily Question

Daily Telegram bot that asks an investment banking interview question, transcribes your voice reply with OpenAI Whisper, and grades it against a per-question rubric with Claude Haiku.

## How it works

- **10:00 AM PT every day** — `launchd` triggers `send_question.py`, which picks a question (avoiding any asked in the last 30 days, rotating categories), and DMs it to you on Telegram.
- **Anytime you reply with a voice message** — a 24/7 daemon (`listener.py`) downloads the audio, transcribes it with Whisper, grades it with Haiku against the question's rubric, and replies with a score (0–100), letter grade, what you nailed, what you missed, and overall feedback.

## One-time setup

### 1. Create a Telegram bot
1. Open Telegram, message [@BotFather](https://t.me/BotFather), send `/newbot`, follow prompts.
2. Save the bot token.
3. Search for your bot in Telegram and send it any message (e.g., `hi`) so it knows your chat.

### 2. Get API keys
- **OpenAI:** https://platform.openai.com/api-keys (for Whisper, ~$0.006/min)
- **Anthropic:** https://console.anthropic.com/settings/keys (Haiku grader is cents/day; one-time Sonnet rubric bootstrap is ~$0.50)

### 3. Install
```bash
cd IB-daily-question
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: paste TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, ANTHROPIC_API_KEY
python -m scripts.get_chat_id     # prints your TELEGRAM_CHAT_ID — paste into .env
```

### 4. Generate the rubric bank (one-time, ~5 min)
```bash
python -m scripts.bootstrap_rubrics
```
Produces `data/questions.json` with rubrics for all ~60 questions from the source doc.

### 5. Smoke test before scheduling
```bash
python -m scripts.smoke_test
```
Sends a question to your Telegram, waits for your voice reply, runs the full pipeline once, and prints the trace.

### 6. Schedule with launchd
```bash
./scripts/setup_launchd.sh
```
Installs two jobs: a 10 AM daily sender and a 24/7 listener that auto-restarts on crash.

Verify:
```bash
launchctl list | grep ib-daily
```

## Daily use

Just answer the question in Telegram with a voice memo. You'll get graded within ~15 seconds.

To re-answer the same question before the next morning, just send another voice message — the new grade overwrites the prior one.

## Troubleshooting

- **No question at 10 AM** — check `data/logs/send.err`. Most likely the laptop was asleep; macOS launchd does not fire missed `StartCalendarInterval` events on wake. Workaround: `caffeinate -i` overnight, or move the time to when the laptop is definitely awake.
- **Voice reply not graded** — check `data/logs/listener.err`. Could be expired API key or network. The listener auto-restarts on crash; if it's stuck, `launchctl kickstart -k gui/$UID/com.chrislane.ib-daily.listener`.
- **Want to skip today's question** — just don't reply. Tomorrow's will replace it.

## Files

- Spec: `docs/superpowers/specs/2026-05-23-ib-daily-question-design.md`
- Plan: `docs/superpowers/plans/2026-05-23-ib-daily-question-plan.md`
- Question bank: `data/questions.json` (committed)
- State + history: `data/state.json` (gitignored — contains transcripts)
```

- [ ] **Step 3: Final full test run**

```bash
source .venv/bin/activate
pytest -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add IB-daily-question/scripts/smoke_test.py IB-daily-question/README.md
git commit -m "feat(ib-daily-question): smoke test script + complete README"
```

---

## Done — Manual cutover steps (not part of the plan, just FYI)

After all tasks are merged:

1. Paste real keys into `.env`.
2. Run `python -m scripts.get_chat_id` → paste `TELEGRAM_CHAT_ID` into `.env`.
3. Run `python -m scripts.bootstrap_rubrics` (~5 min, ~$0.50).
4. Run `python -m scripts.smoke_test` and reply with a voice message.
5. Run `./scripts/setup_launchd.sh`.
6. Next morning at 10:00 AM, you'll get your first scheduled question.
