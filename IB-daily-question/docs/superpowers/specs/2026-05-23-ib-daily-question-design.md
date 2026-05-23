# IB Daily Question — Design Spec

**Date:** 2026-05-23
**Owner:** Chris Lane
**Status:** Approved (brainstorm) — pending implementation plan

## 1. Problem & Goal

Daily investment-banking interview prep. Each morning at 10:00 AM PT, a technical IB question (drawn from a professor-provided question bank) is sent to the user via Telegram. The user answers with a voice message (recorded in Telegram as if speaking to an interviewer). The system transcribes the voice, grades the answer against a per-question rubric, and replies with a grade, what was nailed, and what was missed.

**Success criteria:**
- Question arrives reliably at 10:00 AM PT every day.
- User can reply with a voice message at any time of day; reply gets graded within ~15 seconds.
- Grades feel like a realistic IB interviewer (0–100 + letter grade), with concrete callouts.
- No public URL / no cloud hosting required — runs entirely on the user's Mac.

## 2. Non-Goals

- Multi-user support (single user: Chris).
- Persistent web dashboard or history UI (history is queryable via `state.json` if needed).
- Behavioral / "fit" interview questions — technical only.
- Auto-generated questions beyond the doc-provided bank (may revisit later).

## 3. Sources

- **Question bank source:** `Investment Banking Interview Questions.docx` in the project root. ~60 questions across 5 categories: The Absolute Fundamentals (Accounting & Enterprise Value), Core Valuation & DCF, Advanced Accounting & Working Capital, Mergers & Acquisitions (M&A), and Leveraged Buyouts (LBO).

## 4. User Stories

- **As a user**, at 10:00 AM PT I receive a Telegram message with one technical IB question.
- **As a user**, I record a 30–120 second voice memo answering the question and send it back.
- **As a user**, within ~15 seconds I receive a Telegram reply with:
  - Numeric score (0–100) and letter grade on a 13-step scale (A+, A, A−, B+, B, B−, C+, C, C−, D+, D, D−, F).
  - Bullets of points I correctly hit.
  - Bullets of points I missed or was unclear on.
  - A short paragraph of overall feedback in an interviewer's voice.
- **As a user**, if I want to re-answer the same question before the next morning, I can — the new grade overwrites the prior one.
- **As a user**, I never need to keep a terminal window open or remember to start anything; launchd handles all scheduling and restarts.

## 5. Architecture

### 5.1 File layout

```
IB-daily-question/
├── data/
│   ├── questions.json          # ~60 questions + rubrics (committed)
│   └── state.json              # gitignored: telegram_offset, pending, history
├── src/
│   ├── __init__.py
│   ├── bank.py                 # load questions, pick next, mark pending/answered
│   ├── telegram_client.py      # send_message, get_updates, download_voice
│   ├── transcribe.py           # OpenAI Whisper API wrapper
│   ├── grade.py                # Claude Haiku grader against rubric
│   ├── send_question.py        # entry: launchd-triggered at 10 AM PT
│   └── listener.py             # entry: 24/7 long-poll daemon
├── scripts/
│   ├── bootstrap_rubrics.py    # one-time: docx → questions.json with rubrics
│   ├── setup_launchd.sh        # install + load both .plist files
│   └── smoke_test.py           # end-to-end manual test
├── launchd/
│   ├── com.chrislane.ib-daily.send.plist     # CalendarInterval 10:00 PT
│   └── com.chrislane.ib-daily.listener.plist # RunAtLoad + KeepAlive
├── tests/
├── .env.example
├── README.md
└── requirements.txt
```

### 5.2 Units & responsibilities

| Unit | Responsibility | Inputs / deps |
|---|---|---|
| `bank` | Pick next question, track pending, append to history | `data/questions.json`, `data/state.json` |
| `telegram_client` | Pure Telegram I/O — send message, long-poll `getUpdates`, download voice files | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` |
| `transcribe` | `(audio_bytes, mime) → str` via OpenAI Whisper API | `OPENAI_API_KEY` |
| `grade` | `(question, rubric, transcript) → GradeResult` via Claude Haiku, returning structured JSON | `ANTHROPIC_API_KEY` |
| `send_question` | Thin orchestrator: pick → send → mark pending → exit | All of the above |
| `listener` | Long-poll loop: on voice msg → download → transcribe → grade → reply → persist | All of the above |

Each unit can be unit-tested in isolation with the external APIs mocked.

### 5.3 Process model

Two launchd jobs:

- **`com.chrislane.ib-daily.send`** — `CalendarInterval { Hour: 10, Minute: 0 }`, America/Los_Angeles. Runs `python -m src.send_question` once, exits.
- **`com.chrislane.ib-daily.listener`** — `RunAtLoad: true`, `KeepAlive: true`. Runs `python -m src.listener` as a daemon that long-polls Telegram in a loop. If it crashes, launchd restarts it within seconds.

`StandardOutPath` and `StandardErrorPath` rotate weekly into `data/logs/`.

## 6. Data Flow

### 6.1 Daily send (10:00 AM PT)

```
launchd → send_question.py
  1. bank.next_question() → Question
  2. telegram_client.send_message(format_question(Question))
  3. bank.mark_pending(question_id, sent_at, telegram_message_id)
  4. exit 0
```

### 6.2 Voice reply (any time)

```
listener.py main loop:
  while True:
    updates = telegram_client.get_updates(offset, timeout=30)
    for update in updates:
        bank.persist_offset(update.update_id + 1)
        if update has voice:
            handle_voice(update.voice)
        elif update has text:
            telegram_client.send_message("Please answer with a voice message.")

handle_voice(voice):
    pending = bank.current_pending()
    if pending is None:
        telegram_client.send_message("No active question. Next one drops at 10 AM.")
        return
    audio_bytes = telegram_client.download_voice(voice.file_id)
    transcript  = transcribe.whisper(audio_bytes, mime="audio/ogg")
    grade       = grade.grade(pending.question, pending.rubric, transcript)
    telegram_client.send_message(format_result(grade))
    bank.record_response(pending.question_id, transcript, grade)
```

### 6.3 Grader contract

Claude Haiku is prompted to return strict JSON:

```json
{
  "score": 78,
  "letter": "B+",
  "nailed":  ["Correctly walked through the WACC formula with weights", "..."],
  "missed":  ["Didn't explain why we use after-tax cost of debt", "..."],
  "feedback": "Solid structure. To tighten: lead with the formula, then explain each component..."
}
```

Grader system prompt frames Haiku as a "VP-level IB interviewer" using the rubric's `must_hit` (65% weight), `good_to_hit` (20%), and `clarity_structure` (15%). If Haiku returns malformed JSON, retry once with a stricter reminder; on second failure, send a generic "grader hiccup" message and keep the transcript so the user can resend.

### 6.4 Formatted Telegram reply (Markdown)

```
*Grade: B+ (78/100)*

✓ *What you nailed*
  • Correctly walked through the WACC formula...
  • Mentioned levered vs unlevered beta distinction

✗ *What you missed*
  • Didn't explain why we use after-tax cost of debt
  • Glossed over how to derive market value of equity

*Feedback*
Solid structure. To tighten: lead with the formula, then explain each component...
```

## 7. State & Question Bank Schema

### 7.1 `data/questions.json`

```json
[
  {
    "id": "wacc-001",
    "category": "Core Valuation & DCF",
    "difficulty": "fundamental",
    "question": "How do you calculate the Weighted Average Cost of Capital (WACC)?",
    "rubric": {
      "must_hit": [
        "Formula: WACC = (E/V)·Re + (D/V)·Rd·(1−t)",
        "After-tax cost of debt — the (1−t) tax shield and why",
        "Cost of equity via CAPM: Rf + β·(Rm−Rf)",
        "Use market values of debt and equity, not book"
      ],
      "good_to_hit": [
        "Why WACC is the discount rate for unlevered FCF",
        "Rd estimated from yield on existing debt or comparable yields",
        "How to lever/unlever beta if peer betas are used"
      ],
      "common_pitfalls": [
        "Using book values instead of market values",
        "Forgetting the (1−t) on cost of debt",
        "Confusing Rd with the coupon rate"
      ],
      "scoring_weights": { "must_hit": 0.65, "good_to_hit": 0.20, "clarity_structure": 0.15 }
    }
  }
]
```

All ~60 entries are generated once by `scripts/bootstrap_rubrics.py`:
1. Parses the `.docx` to extract `(category, question)` pairs.
2. For each question, calls Claude Sonnet (not Haiku — quality matters here) to generate the rubric in the above shape.
3. Writes `data/questions.json`. Re-runnable — by default skips entries with an existing `id`, with a `--regenerate` flag to overwrite.

### 7.2 `data/state.json`

```json
{
  "telegram_offset": 84729103,
  "pending": {
    "question_id": "wacc-001",
    "sent_at": "2026-05-23T17:00:00Z",
    "telegram_message_id": 4421
  },
  "history": [
    {
      "question_id": "ev-eqv-001",
      "sent_at": "2026-05-22T17:00:00Z",
      "responded_at": "2026-05-22T17:34:12Z",
      "transcript": "Enterprise value is the value of the entire business...",
      "grade": {
        "score": 82, "letter": "B+",
        "nailed": ["..."], "missed": ["..."], "feedback": "..."
      }
    }
  ]
}
```

### 7.3 Selection algorithm — `bank.next_question()`

1. Exclude any `question_id` answered in the last 30 days (from `history`).
2. Of the remainder, weight by category to avoid 3 of the same category in a row — track last-3 categories from history and downweight matches.
3. If all ~60 have been asked in the last 30 days, reset and pick the question with the oldest `responded_at`.
4. Return one question (random within weighted set).

### 7.4 Concurrency / writes

Two processes touch `state.json`:

- `send_question.py` (once daily) — sets `pending` after sending the morning question.
- `listener.py` (continuous) — advances `telegram_offset` on every poll, clears `pending` and appends to `history` after grading.

The race window is small but real (e.g., user replies at exactly 10:00:00.x). Mitigation: every write does a read → mutate → write of the entire `state.json` under an exclusive file lock (`fcntl.flock`). Simple, robust, no DB needed.

## 8. Error Handling

| Failure | Handling |
|---|---|
| Telegram API down at 10 AM | `send_question.py` retries 3× with exponential backoff (2/4/8s), then logs error to `data/logs/send.err`, exits non-zero (visible in `Console.app`) |
| Whisper API fails | Reply "Couldn't transcribe — please resend the voice message," do NOT mark answered |
| Claude API fails / returns invalid JSON | Retry grade once; on second failure reply "Grader hiccup — your transcript was saved, resend the voice in a minute," do NOT mark answered |
| User sends text while a question is pending | Reply: "Please answer with a voice message — record like you're talking to an interviewer." |
| User sends voice with no pending question | Reply: "No active question — next one drops at 10 AM." |
| Listener crashes | `KeepAlive=true` → launchd restarts; persisted `telegram_offset` prevents missed messages |
| Voice message > 60s | Process normally; Whisper handles long audio. No artificial cap. |
| Network drop mid-grade | Transcript is saved to `state.json` before the grading call, so user can resend the voice to retry; idempotent on `question_id` (latest grade wins) |
| `state.json` corrupted | On parse failure, back it up to `state.json.bak.<ts>` and start fresh; surface a Telegram message: "State reset — please re-send your last voice message if you were mid-answer." |

**Logging:** structured JSON lines to `data/logs/listener.log` and `data/logs/send.log`. launchd handles file rotation via separate weekly paths if needed; otherwise simple append-only.

## 9. Secrets & Config

Loaded from a project-local `.env` (gitignored) via `python-dotenv`:

- `TELEGRAM_BOT_TOKEN` — from @BotFather.
- `TELEGRAM_CHAT_ID` — Chris's personal chat with the bot (resolved on first run via `/start`; bootstrap script can grab it).
- `OPENAI_API_KEY` — for Whisper.
- `ANTHROPIC_API_KEY` — for Claude Haiku (grader) and Sonnet (one-time rubric bootstrap).
- `IB_DAILY_TZ` — defaults to `America/Los_Angeles`.

`.env.example` is committed showing all required keys with placeholder values. No keys in launchd plists — plists invoke a wrapper that sources `.env`.

## 10. Testing

### 10.1 Unit tests (pytest, mocked APIs)

- `tests/test_bank.py`
  - 30-day exclusion
  - Category-rotation weighting (no 3-in-a-row)
  - Cycle reset when all asked
  - `record_response` overwrite semantics
  - File-lock behavior on concurrent writes (use a thread + small sleep)
- `tests/test_telegram_client.py`
  - Send message — request shape, error retry
  - `get_updates` — offset increment, timeout
  - `download_voice` — two-step file_path → bytes flow
- `tests/test_transcribe.py`
  - Whisper call shape, multipart content
  - Surfaces error on non-200
- `tests/test_grade.py`
  - Prompt assembly with rubric
  - Parses well-formed JSON
  - Retries once on malformed, raises on second failure
- `tests/test_format.py`
  - Question and result formatting (Markdown shape, escaping)

### 10.2 Integration test

`tests/test_integration.py` — full flow with all three external APIs mocked:
fake voice bytes → transcribe (stub) → grade (stub) → assert Telegram send called with correctly formatted result and state mutated.

### 10.3 Smoke test (manual, not in CI)

`scripts/smoke_test.py`:
- Forces a known question to be the next-up.
- Sends it via the real Telegram bot.
- Waits for a real voice reply (long-poll).
- Runs the real Whisper + Haiku.
- Sends real grade back.
- Prints the full pipeline trace to stdout.

Run once after setup to validate end-to-end before enabling launchd.

## 11. Out of Scope (Future Work)

- Web-based history dashboard.
- Auto-generated questions beyond the doc bank.
- Spaced-repetition scheduling weighted by past scores ("review the ones you bombed more often").
- Behavioral / "tell me about a time" questions.
- Multi-user / shareable.

## 12. Open Questions

None at spec time. Implementation plan will surface any edge cases that need follow-up.
