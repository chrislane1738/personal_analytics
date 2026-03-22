"""Tests for the Typer CLI (main.py) and utils/logging_config.py.

Test coverage:
1.  Top-level --help renders and contains "trading-bot"
2.  Sub-command help text renders for data / backtest / report / paper
3.  data status with empty DB — no crash, informative message
4.  data status with pre-loaded bars — shows symbol table
5.  report list with empty DB — no crash
6.  report list with pre-loaded run — shows run details
7.  report last with empty DB — no crash
8.  report last with pre-loaded run — shows report
9.  report show with unknown run_id — exit code 1, error message
10. report show with known run_id — shows console report
11. paper start / status / stop — all return "not yet implemented"
12. backtest compare / optimize — both return "not yet implemented"
13. backtest run with fixture data — full integration, no FMP calls needed
14. setup_logging does not raise
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from data.storage.database import Database
from data.storage.models import DailyBar, RunRecord
from main import app
from tests.fixtures.sample_bars import generate_spy_bars
from utils.logging_config import setup_logging

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db_path(tmp_path):
    """Return a path string for a temporary DB file (not yet created)."""
    return str(tmp_path / "test_cli.db")


@pytest.fixture
def empty_db(tmp_db_path):
    """Open an empty database with tables created."""
    db = Database(tmp_db_path)
    db.create_tables()
    yield db
    db.close()


@pytest.fixture
def loaded_db(tmp_db_path):
    """Database with 100 SPY bars and one fake run pre-loaded."""
    db = Database(tmp_db_path)
    db.create_tables()

    bars = generate_spy_bars("SPY")
    db.insert_daily_bars(bars)

    run = RunRecord(
        run_id="abc12345",
        mode="backtest",
        strategy_name="SimpleSMAStrategy",
        config="",
        start_date=bars[0].date,
        end_date=bars[-1].date,
        initial_capital=100_000.0,
        final_value=102_500.0,
        total_return=0.025,
        sharpe=0.85,
        max_drawdown=-0.05,
        created_at=datetime.now(tz=timezone.utc),
        full_metrics=json.dumps(
            {
                "total_return": "0.025",
                "sharpe_ratio": "0.85",
                "max_drawdown_pct": "-0.05",
                "total_trades": "3",
                "start_date": str(bars[0].date),
                "end_date": str(bars[-1].date),
            }
        ),
    )
    db.insert_run(run)
    yield db, tmp_db_path
    db.close()


# ---------------------------------------------------------------------------
# Helper: patch get_settings to point at our temp DB
# ---------------------------------------------------------------------------

def _settings_patch(db_path: str):
    """Return a context-manager that overrides the DB path in Settings."""
    from config.settings import Settings

    patched = Settings(db_path=db_path, fmp_api_key="test-key")

    return patch("main.get_settings", return_value=patched)


# ---------------------------------------------------------------------------
# 1–2. Help text
# ---------------------------------------------------------------------------

class TestHelp:
    def test_root_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "trading-bot" in result.output

    def test_data_help(self):
        result = runner.invoke(app, ["data", "--help"])
        assert result.exit_code == 0
        assert "data" in result.output.lower()

    def test_backtest_help(self):
        result = runner.invoke(app, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "backtest" in result.output.lower()

    def test_report_help(self):
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0
        assert "report" in result.output.lower()

    def test_paper_help(self):
        result = runner.invoke(app, ["paper", "--help"])
        assert result.exit_code == 0
        assert "paper" in result.output.lower()

    def test_data_fetch_help(self):
        result = runner.invoke(app, ["data", "fetch", "--help"])
        assert result.exit_code == 0
        assert "symbol" in result.output.lower()

    def test_backtest_run_help(self):
        result = runner.invoke(app, ["backtest", "run", "--help"])
        assert result.exit_code == 0
        assert "strategy" in result.output.lower()


# ---------------------------------------------------------------------------
# 3–4. data status
# ---------------------------------------------------------------------------

class TestDataStatus:
    def test_empty_db_no_crash(self, empty_db, tmp_db_path):
        with _settings_patch(tmp_db_path):
            result = runner.invoke(app, ["data", "status"])
        assert result.exit_code == 0
        assert "No data cached" in result.output

    def test_loaded_db_shows_symbol(self, loaded_db):
        db, db_path = loaded_db
        with _settings_patch(db_path):
            result = runner.invoke(app, ["data", "status"])
        assert result.exit_code == 0
        assert "SPY" in result.output
        assert "100" in result.output  # bar count


# ---------------------------------------------------------------------------
# 5–6. report list
# ---------------------------------------------------------------------------

class TestReportList:
    def test_empty_db_no_crash(self, empty_db, tmp_db_path):
        with _settings_patch(tmp_db_path):
            result = runner.invoke(app, ["report", "list"])
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_loaded_db_shows_run(self, loaded_db):
        db, db_path = loaded_db
        with _settings_patch(db_path):
            result = runner.invoke(app, ["report", "list"])
        assert result.exit_code == 0
        assert "abc12345" in result.output
        assert "SimpleSMAStrategy" in result.output


# ---------------------------------------------------------------------------
# 7–8. report last
# ---------------------------------------------------------------------------

class TestReportLast:
    def test_empty_db_no_crash(self, empty_db, tmp_db_path):
        with _settings_patch(tmp_db_path):
            result = runner.invoke(app, ["report", "last"])
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_loaded_db_shows_report(self, loaded_db):
        db, db_path = loaded_db
        with _settings_patch(db_path):
            result = runner.invoke(app, ["report", "last"])
        assert result.exit_code == 0
        assert "BACKTEST REPORT" in result.output
        assert "SimpleSMAStrategy" in result.output


# ---------------------------------------------------------------------------
# 9–10. report show
# ---------------------------------------------------------------------------

class TestReportShow:
    def test_unknown_run_id_exits_1(self, empty_db, tmp_db_path):
        with _settings_patch(tmp_db_path):
            result = runner.invoke(app, ["report", "show", "nonexistent"])
        assert result.exit_code == 1

    def test_known_run_id_shows_report(self, loaded_db):
        db, db_path = loaded_db
        with _settings_patch(db_path):
            result = runner.invoke(app, ["report", "show", "abc12345"])
        assert result.exit_code == 0
        assert "BACKTEST REPORT" in result.output


# ---------------------------------------------------------------------------
# 11. paper commands — all "not yet implemented"
# ---------------------------------------------------------------------------

class TestPaperCommands:
    def test_paper_start(self):
        result = runner.invoke(app, ["paper", "start", "some.Strategy"])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output.lower()

    def test_paper_status(self):
        result = runner.invoke(app, ["paper", "status"])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output.lower()

    def test_paper_stop(self):
        result = runner.invoke(app, ["paper", "stop"])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output.lower()


# ---------------------------------------------------------------------------
# 12. backtest compare / optimize — "not yet implemented"
# ---------------------------------------------------------------------------

class TestBacktestStubs:
    def test_compare_not_implemented(self, empty_db, tmp_db_path):
        with _settings_patch(tmp_db_path):
            result = runner.invoke(
                app,
                ["backtest", "compare", "strategy.A,strategy.B", "--start", "2024-01-01"],
            )
        assert result.exit_code == 0
        assert "not yet implemented" in result.output.lower()

    def test_optimize_not_implemented(self):
        result = runner.invoke(
            app, ["backtest", "optimize", "strategy.library.X.X"]
        )
        assert result.exit_code == 0
        assert "not yet implemented" in result.output.lower()


# ---------------------------------------------------------------------------
# 13. backtest run — full integration with fixture data, no FMP calls
# ---------------------------------------------------------------------------

class _BuyAndHoldStrategy:
    """Minimal inline strategy: buy every symbol on the first bar."""

    from strategy.base import Strategy
    from core.events import BarEvent, SignalEvent

    class _Impl:
        """Inline buy-and-hold: buy on first bar, never sell."""

        def __init__(self):
            self.indicators = {}
            self._bought: set = set()

        def warm_up_period(self):
            return 0

        def generate_signals(self, bar, portfolio):
            from core.events import SignalEvent

            if bar.symbol not in self._bought and not portfolio.has_position(bar.symbol):
                self._bought.add(bar.symbol)
                return [
                    SignalEvent(
                        symbol=bar.symbol,
                        direction="long",
                        reason="buy and hold",
                        strength=1.0,
                    )
                ]
            return []

        def on_universe_bar(self, bars, portfolio):
            signals = []
            for bar in bars.values():
                signals.extend(self.generate_signals(bar, portfolio))
            return signals

        def on_start(self, context):
            pass

        def on_end(self, portfolio):
            return []

        def on_fill(self, fill):
            pass


class TestBacktestRun:
    def test_backtest_run_with_fixture_data(self, loaded_db):
        """Run a full backtest against pre-loaded SPY bars — no FMP network calls."""
        db, db_path = loaded_db

        bars = generate_spy_bars("SPY")
        start_str = str(bars[0].date)
        end_str = str(bars[-1].date)

        strategy_ref = "tests.fixtures.inline_strategy.BuyAndHoldStrategy"

        # Patch get_settings and also patch StrategyRegistry.resolve to return
        # our inline strategy (avoids needing a real importable module).
        inline_strat = _BuyAndHoldStrategy._Impl()

        with (
            _settings_patch(db_path),
            patch("main.StrategyRegistry.resolve", return_value=inline_strat),
        ):
            result = runner.invoke(
                app,
                [
                    "backtest",
                    "run",
                    strategy_ref,
                    "--start",
                    start_str,
                    "--end",
                    end_str,
                    "--universe",
                    "SPY",
                ],
            )

        assert result.exit_code == 0, f"CLI failed:\n{result.output}\n{result.exception}"
        # The engine always prints the console report
        assert "BACKTEST REPORT" in result.output

    def test_backtest_run_unknown_strategy_exits_1(self, empty_db, tmp_db_path):
        with _settings_patch(tmp_db_path):
            result = runner.invoke(
                app,
                [
                    "backtest",
                    "run",
                    "no.Such.Module.Strategy",
                    "--start",
                    "2024-01-01",
                    "--universe",
                    "SPY",
                ],
            )
        assert result.exit_code == 1

    def test_backtest_run_no_universe_exits_1(self, empty_db, tmp_db_path):
        """If strategy has no universe and none is passed, exit with error."""
        inline_strat = _BuyAndHoldStrategy._Impl()

        with (
            _settings_patch(tmp_db_path),
            patch("main.StrategyRegistry.resolve", return_value=inline_strat),
        ):
            result = runner.invoke(
                app,
                [
                    "backtest",
                    "run",
                    "any.Strategy",
                    "--start",
                    "2024-01-01",
                ],
            )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 14. setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_info_level(self):
        """setup_logging('INFO') should not raise."""
        setup_logging("INFO")

    def test_debug_level(self):
        setup_logging("DEBUG")

    def test_unknown_level_defaults_to_info(self):
        """Unknown level should not raise (falls back to INFO)."""
        setup_logging("NOTAREAL")

    def test_lowercase_accepted(self):
        setup_logging("warning")
