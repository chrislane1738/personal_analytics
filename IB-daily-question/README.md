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
