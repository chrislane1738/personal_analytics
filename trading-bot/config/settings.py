"""Global configuration loaded from .env and defaults."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # FMP
    fmp_api_key: str = ""
    fmp_base_url: str = "https://financialmodelingprep.com"
    fmp_rate_limit: int = 700  # req/min, headroom below 750

    # Databento (futures intraday data)
    databento_api_key: str = ""

    # Database
    db_path: str = "db/trading_bot.db"

    # Defaults
    default_benchmark: str = "SPY"
    default_capital: float = 100_000.0
    default_position_size: float = 0.06  # 6%
    default_max_sector_pct: float = 0.25
    default_max_positions: int = 20
    default_drawdown_limit: float = -0.15
    default_commission_per_share: float = 0.005
    default_slippage_pct: float = 0.0001  # 0.01%
    risk_free_rate: float = 0.04  # 4% default

    # Broker (future)
    schwab_client_id: str = ""
    schwab_client_secret: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
