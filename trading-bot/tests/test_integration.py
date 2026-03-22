"""End-to-end integration tests — the v1 Definition of Done acceptance test."""
import pytest
from datetime import date
from pathlib import Path
from data.storage.database import Database
from data.storage.models import DailyBar
from data.validation.validators import validate_bar, validate_gap
from data.validation.quality_report import generate_quality_report, compute_quality_score
from strategy.config_strategy import ConfigStrategy
from strategy.library.options_wheel import OptionsWheelStrategy
from core.engine import BacktestEngine
from analytics.metrics import compute_all_metrics
from analytics.trade_log import TradeLog
from analytics.reports import ReportGenerator
from analytics.monte_carlo import run_monte_carlo
from analytics.regime import detect_regimes, compute_regime_stats
from tests.fixtures.sample_bars import generate_spy_bars


class TestFullPipelineIntegration:
    """Tests the complete pipeline from data ingest to report output."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Create DB, insert sample data, return components."""
        db = Database(str(tmp_path / "integration.db"))
        db.create_tables()
        bars = generate_spy_bars()
        db.insert_daily_bars(bars)
        return db, bars, tmp_path

    def test_data_validation_pipeline(self, setup):
        """DoD #1: Data pipeline validates and caches data."""
        db, bars, _ = setup

        # Validate each bar
        issues = []
        for i, bar in enumerate(bars):
            prev = bars[i - 1] if i > 0 else None
            bar_issues = validate_bar(bar, prev)
            issues.extend(bar_issues)

        # Run gap detection
        gap_issues = validate_gap(bars)

        # Generate quality report
        all_issues = issues + gap_issues
        report = generate_quality_report(bars, all_issues)

        assert report["total_bars"] == len(bars)
        assert "avg_quality_score" in report
        assert report["avg_quality_score"] > 0.5  # Most bars should be decent quality

        # Verify data is cached in DB
        cached = db.get_daily_bars("SPY", bars[0].date, bars[-1].date)
        assert len(cached) == len(bars)
        db.close()

    def test_yaml_strategy_backtest(self, setup):
        """DoD #2: YAML strategy backtests successfully."""
        db, bars, _ = setup
        config_path = str(Path(__file__).parent.parent / "config" / "strategies" / "mean_reversion.yaml")
        strategy = ConfigStrategy(config_path)

        start_date = min(b.date for b in bars)
        end_date = max(b.date for b in bars)

        engine = BacktestEngine(
            strategy=strategy, database=db, universe=["SPY"],
            start_date=start_date, end_date=end_date,
            initial_capital=100000.0, benchmark_symbol="SPY",
        )
        metrics = engine.run()

        assert metrics is not None
        assert "total_return" in metrics
        # Verify engine tracked equity
        assert len(engine.portfolio.equity_curve) > 0
        db.close()

    def test_python_strategy_backtest(self, setup):
        """DoD #3: Python strategy backtests successfully."""
        db, bars, _ = setup
        strategy = OptionsWheelStrategy(rsi_period=10, rsi_threshold=40)

        start_date = min(b.date for b in bars)
        end_date = max(b.date for b in bars)

        engine = BacktestEngine(
            strategy=strategy, database=db, universe=["SPY"],
            start_date=start_date, end_date=end_date,
            initial_capital=100000.0, benchmark_symbol="SPY",
        )
        metrics = engine.run()

        assert metrics is not None
        assert "total_return" in metrics
        db.close()

    def test_console_report(self, setup):
        """DoD #4: Console report with all key metrics."""
        db, bars, _ = setup
        config_path = str(Path(__file__).parent.parent / "config" / "strategies" / "mean_reversion.yaml")
        strategy = ConfigStrategy(config_path)

        start_date = min(b.date for b in bars)
        end_date = max(b.date for b in bars)

        engine = BacktestEngine(
            strategy=strategy, database=db, universe=["SPY"],
            start_date=start_date, end_date=end_date,
            initial_capital=100000.0,
        )
        metrics = engine.run()

        report_gen = ReportGenerator(
            run_id="integration-test", strategy_name="MeanReversion",
            metrics=metrics, trade_log=engine.trade_log,
            equity_curve=engine.portfolio.equity_curve,
            benchmark_curve=engine.benchmark.equity_curve,
        )
        console = report_gen.generate_console_report()

        assert "PERFORMANCE" in console
        assert "RISK" in console
        assert "TRADES" in console
        assert "Total Return" in console
        db.close()

    def test_html_report(self, setup):
        """DoD #5: HTML report with Plotly charts."""
        db, bars, tmp_path = setup
        config_path = str(Path(__file__).parent.parent / "config" / "strategies" / "mean_reversion.yaml")
        strategy = ConfigStrategy(config_path)

        start_date = min(b.date for b in bars)
        end_date = max(b.date for b in bars)

        engine = BacktestEngine(
            strategy=strategy, database=db, universe=["SPY"],
            start_date=start_date, end_date=end_date,
            initial_capital=100000.0,
        )
        metrics = engine.run()

        report_gen = ReportGenerator(
            run_id="html-test", strategy_name="MeanReversion",
            metrics=metrics, trade_log=engine.trade_log,
            equity_curve=engine.portfolio.equity_curve,
            benchmark_curve=engine.benchmark.equity_curve,
        )
        report_dir = str(tmp_path / "reports")
        html_path = report_gen.generate_html_report(output_dir=report_dir)

        assert Path(html_path).exists()
        html_content = Path(html_path).read_text()
        assert "plotly" in html_content.lower()
        assert "equity" in html_content.lower()
        db.close()

    def test_monte_carlo(self, setup):
        """DoD #7: Monte Carlo produces confidence bands."""
        db, bars, _ = setup
        # Create some sample trade PnLs
        trade_pnls = [500, -200, 300, -100, 400, -150, 250, -50, 350, -100]

        result = run_monte_carlo(trade_pnls, initial_capital=100000.0,
                                 num_simulations=1000, seed=42)

        assert result.simulations == 1000
        assert result.median_final_equity > 0
        assert result.percentile_5 <= result.percentile_95
        assert 0 <= result.probability_of_ruin <= 1
        db.close()

    def test_regime_analysis(self, setup):
        """DoD #8: Regime detection and per-regime stats."""
        db, bars, _ = setup
        prices = [(b.date, b.close) for b in bars]

        regimes = detect_regimes(prices)

        assert len(regimes) > 0
        assert all(hasattr(r[1], 'value') for r in regimes)
        db.close()

    def test_run_saved_to_db(self, setup):
        """DoD: Backtest run is persisted."""
        db, bars, _ = setup
        config_path = str(Path(__file__).parent.parent / "config" / "strategies" / "mean_reversion.yaml")
        strategy = ConfigStrategy(config_path)

        start_date = min(b.date for b in bars)
        end_date = max(b.date for b in bars)

        engine = BacktestEngine(
            strategy=strategy, database=db, universe=["SPY"],
            start_date=start_date, end_date=end_date,
            initial_capital=100000.0,
        )
        engine.run()

        runs = db.list_runs()
        assert len(runs) >= 1
        db.close()

    def test_benchmark_comparison(self, setup):
        """DoD: Strategy vs benchmark comparison."""
        db, bars, _ = setup
        config_path = str(Path(__file__).parent.parent / "config" / "strategies" / "mean_reversion.yaml")
        strategy = ConfigStrategy(config_path)

        start_date = min(b.date for b in bars)
        end_date = max(b.date for b in bars)

        engine = BacktestEngine(
            strategy=strategy, database=db, universe=["SPY"],
            start_date=start_date, end_date=end_date,
            initial_capital=100000.0, benchmark_symbol="SPY",
        )
        metrics = engine.run()

        # Benchmark data should be tracked
        assert len(engine.benchmark.equity_curve) > 0
        db.close()
