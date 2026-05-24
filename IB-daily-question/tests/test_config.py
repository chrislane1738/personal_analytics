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
    # Bypass real .env file — it would otherwise repopulate the deleted var
    monkeypatch.setattr("src.config.load_dotenv", lambda *a, **kw: None)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    from src import config
    import pytest
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        config.load()
