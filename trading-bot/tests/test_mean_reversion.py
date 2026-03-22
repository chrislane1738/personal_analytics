"""End-to-end integration test for the mean reversion YAML strategy.

Runs the full backtest engine with config/strategies/mean_reversion.yaml and
verifies that results are coherent: metrics are populated, equity curves are
recorded, and the run is persisted to the database.
"""

from __future__ import annotations

import pytest
from datetime import date
from pathlib import Path

from data.storage.database import Database
from data.storage.models import DailyBar
from strategy.config_strategy import ConfigStrategy
from core.engine import BacktestEngine
from tests.fixtures.sample_bars import generate_spy_bars


class TestMeanReversionStrategy:
    def _setup_backtest(self, tmp_path):
        """Helper to set up DB with sample data and create engine."""
        db = Database(str(tmp_path / "test.db"))
        db.create_tables()
        bars = generate_spy_bars()
        db.insert_daily_bars(bars)

        config_path = str(
            Path(__file__).parent.parent / "config" / "strategies" / "mean_reversion.yaml"
        )
        strategy = ConfigStrategy(config_path)

        # Get date range from bars
        start_date = min(b.date for b in bars)
        end_date = max(b.date for b in bars)

        engine = BacktestEngine(
            strategy=strategy,
            database=db,
            universe=["SPY"],
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            benchmark_symbol="SPY",
        )
        return engine, db

    def test_backtest_completes(self, tmp_path):
        """Strategy runs to completion without errors."""
        engine, db = self._setup_backtest(tmp_path)
        metrics = engine.run()
        assert metrics is not None
        assert "total_return" in metrics
        db.close()

    def test_metrics_have_expected_keys(self, tmp_path):
        """All expected metric keys are present."""
        engine, db = self._setup_backtest(tmp_path)
        metrics = engine.run()
        # sharpe_ratio and sortino_ratio are the actual keys returned by compute_all_metrics
        for key in ["sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "total_return", "total_trades"]:
            assert key in metrics, f"Missing key: {key}"
        db.close()

    def test_equity_curve_populated(self, tmp_path):
        """Engine records equity values over time."""
        engine, db = self._setup_backtest(tmp_path)
        engine.run()
        assert len(engine.portfolio.equity_curve) > 0
        db.close()

    def test_warm_up_period_correct(self, tmp_path):
        """ConfigStrategy's warm-up is max of indicator warm-ups (BB-20 = 20, SMA-50 = 50)."""
        config_path = str(
            Path(__file__).parent.parent / "config" / "strategies" / "mean_reversion.yaml"
        )
        strategy = ConfigStrategy(config_path)
        assert strategy.warm_up_period() >= 20  # At least BB period

    def test_benchmark_tracked(self, tmp_path):
        """Benchmark equity curve is populated alongside strategy."""
        engine, db = self._setup_backtest(tmp_path)
        engine.run()
        assert len(engine.benchmark.equity_curve) > 0
        db.close()

    def test_run_saved_to_database(self, tmp_path):
        """Backtest run is persisted to the runs table."""
        engine, db = self._setup_backtest(tmp_path)
        engine.run()
        runs = db.list_runs()
        assert len(runs) >= 1
        assert runs[0].strategy_name == "ConfigStrategy"
        db.close()
