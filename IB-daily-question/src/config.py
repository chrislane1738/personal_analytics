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
