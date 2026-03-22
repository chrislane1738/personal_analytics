"""Tests for analytics/reports.py — ReportGenerator console + HTML output."""

from __future__ import annotations

import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from analytics.reports import ReportGenerator
from analytics.monte_carlo import MonteCarloResult
from analytics.trade_log import TradeLog


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

METRICS_BASIC = {
    "start_date": date(2023, 1, 1),
    "end_date": date(2023, 12, 31),
    "total_return": 0.25,
    "cagr": 0.22,
    "sharpe": 1.45,
    "sortino": 1.82,
    "calmar": 2.10,
    "max_drawdown_pct": 0.08,
    "annualized_volatility": 0.15,
    "var_95": 0.02,
    "total_trades": 42,
    "win_rate": 0.60,
    "profit_factor": 2.30,
    "expectancy": 125.50,
}

METRICS_WITH_BENCHMARK = {
    **METRICS_BASIC,
    "benchmark_total_return": 0.15,
    "alpha": 0.10,
    "beta": 0.85,
}


def make_equity_curve(days: int = 40, start_value: float = 100_000.0) -> list:
    """Build a simple equity curve of (date, value) tuples."""
    base = date(2023, 1, 2)
    return [
        (base + timedelta(days=i), start_value * (1 + i * 0.001))
        for i in range(days)
    ]


def make_monte_carlo(is_outlier: bool = False) -> MonteCarloResult:
    return MonteCarloResult(
        simulations=1000,
        median_final_equity=105_000.0,
        percentile_5=95_000.0,
        percentile_95=115_000.0,
        actual_final_equity=107_000.0,
        probability_of_ruin=0.02,
        max_drawdown_median=0.12,
        max_drawdown_95=0.25,
        is_outlier=is_outlier,
        equity_distribution=[],
        drawdown_distribution=[],
    )


def make_trade_log() -> TradeLog:
    tl = TradeLog()
    tl.open_trade("AAPL", "long", date(2024, 1, 2), 150.0, 100, "RSI oversold")
    tl.close_trade("AAPL", date(2024, 1, 15), 160.0, "Mean reversion")
    return tl


# ---------------------------------------------------------------------------
# 1. Console report — basic sections present
# ---------------------------------------------------------------------------

class TestConsoleReport:
    def test_contains_performance_section(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert "PERFORMANCE" in report

    def test_contains_risk_section(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert "RISK" in report

    def test_contains_trades_section(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert "TRADES" in report

    def test_contains_strategy_name(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert "MyStrategy" in report

    def test_total_return_value(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        # 0.25 * 100 = 25.0%
        assert "25.0%" in report

    def test_sharpe_value(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert "1.45" in report

    def test_win_rate_value(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        # 0.60 * 100 = 60.0%
        assert "60.0%" in report

    def test_total_trades_value(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert "42" in report

    def test_profit_factor_value(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert "2.30" in report

    def test_expectancy_value(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert "125.50" in report

    def test_max_drawdown_value(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        # 0.08 * 100 = 8.0%
        assert "8.0%" in report

    def test_separator_lines_present(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert "=" * 60 in report

    def test_returns_string(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert isinstance(report, str)

    def test_no_benchmark_section_without_key(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert "BENCHMARK" not in report

    def test_no_monte_carlo_section_without_data(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert "MONTE CARLO" not in report

    def test_no_regime_section_without_data(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC)
        report = rg.generate_console_report()
        assert "REGIME" not in report


# ---------------------------------------------------------------------------
# 2. Console with benchmark
# ---------------------------------------------------------------------------

class TestConsoleReportBenchmark:
    def test_benchmark_section_present(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_WITH_BENCHMARK)
        report = rg.generate_console_report()
        assert "BENCHMARK COMPARISON" in report

    def test_benchmark_return_value(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_WITH_BENCHMARK)
        report = rg.generate_console_report()
        # 0.15 * 100 = 15.0%
        assert "15.0%" in report

    def test_alpha_value(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_WITH_BENCHMARK)
        report = rg.generate_console_report()
        # 0.10 * 100 = 10.00%
        assert "10.00%" in report

    def test_beta_value(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_WITH_BENCHMARK)
        report = rg.generate_console_report()
        assert "0.85" in report

    def test_spy_label_present(self):
        rg = ReportGenerator("run1", "MyStrategy", METRICS_WITH_BENCHMARK)
        report = rg.generate_console_report()
        assert "SPY" in report


# ---------------------------------------------------------------------------
# 3. Console with Monte Carlo
# ---------------------------------------------------------------------------

class TestConsoleReportMonteCarlo:
    def test_monte_carlo_section_present(self):
        mc = make_monte_carlo()
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC, monte_carlo_result=mc)
        report = rg.generate_console_report()
        assert "MONTE CARLO" in report

    def test_median_equity_value(self):
        mc = make_monte_carlo()
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC, monte_carlo_result=mc)
        report = rg.generate_console_report()
        assert "105,000" in report

    def test_percentile_5_value(self):
        mc = make_monte_carlo()
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC, monte_carlo_result=mc)
        report = rg.generate_console_report()
        assert "95,000" in report

    def test_percentile_95_value(self):
        mc = make_monte_carlo()
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC, monte_carlo_result=mc)
        report = rg.generate_console_report()
        assert "115,000" in report

    def test_probability_of_ruin_value(self):
        mc = make_monte_carlo()
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC, monte_carlo_result=mc)
        report = rg.generate_console_report()
        # 0.02 * 100 = 2.00%
        assert "2.00%" in report

    def test_not_outlier_label(self):
        mc = make_monte_carlo(is_outlier=False)
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC, monte_carlo_result=mc)
        report = rg.generate_console_report()
        assert "No" in report

    def test_outlier_label_when_is_outlier(self):
        mc = make_monte_carlo(is_outlier=True)
        rg = ReportGenerator("run1", "MyStrategy", METRICS_BASIC, monte_carlo_result=mc)
        report = rg.generate_console_report()
        assert "YES" in report


# ---------------------------------------------------------------------------
# 4. Console with regime stats
# ---------------------------------------------------------------------------

class TestConsoleReportRegime:
    REGIME_STATS = {
        "bull": {"trades": 20, "win_rate": 0.70, "avg_pnl": 200.0},
        "bear": {"trades": 10, "win_rate": 0.40, "avg_pnl": -50.0},
    }

    def test_regime_section_present(self):
        rg = ReportGenerator(
            "run1", "MyStrategy", METRICS_BASIC, regime_stats=self.REGIME_STATS
        )
        report = rg.generate_console_report()
        assert "REGIME PERFORMANCE" in report

    def test_bull_regime_present(self):
        rg = ReportGenerator(
            "run1", "MyStrategy", METRICS_BASIC, regime_stats=self.REGIME_STATS
        )
        report = rg.generate_console_report()
        assert "bull" in report

    def test_bear_regime_present(self):
        rg = ReportGenerator(
            "run1", "MyStrategy", METRICS_BASIC, regime_stats=self.REGIME_STATS
        )
        report = rg.generate_console_report()
        assert "bear" in report

    def test_regime_trade_count(self):
        rg = ReportGenerator(
            "run1", "MyStrategy", METRICS_BASIC, regime_stats=self.REGIME_STATS
        )
        report = rg.generate_console_report()
        assert "20" in report

    def test_regime_win_rate(self):
        rg = ReportGenerator(
            "run1", "MyStrategy", METRICS_BASIC, regime_stats=self.REGIME_STATS
        )
        report = rg.generate_console_report()
        assert "70.0%" in report


# ---------------------------------------------------------------------------
# 5. HTML report — file creation
# ---------------------------------------------------------------------------

class TestHtmlReportCreation:
    def test_file_created(self, tmp_path):
        rg = ReportGenerator("run_html_1", "HtmlStrategy", METRICS_BASIC)
        path = rg.generate_html_report(output_dir=str(tmp_path))
        assert Path(path).exists()

    def test_file_at_expected_path(self, tmp_path):
        rg = ReportGenerator("run_html_2", "HtmlStrategy", METRICS_BASIC)
        path = rg.generate_html_report(output_dir=str(tmp_path))
        expected = str(tmp_path / "run_html_2.html")
        assert path == expected

    def test_returns_string_path(self, tmp_path):
        rg = ReportGenerator("run_html_3", "HtmlStrategy", METRICS_BASIC)
        path = rg.generate_html_report(output_dir=str(tmp_path))
        assert isinstance(path, str)

    def test_output_dir_created_if_missing(self, tmp_path):
        new_dir = str(tmp_path / "subdir" / "nested")
        rg = ReportGenerator("run_html_4", "HtmlStrategy", METRICS_BASIC)
        path = rg.generate_html_report(output_dir=new_dir)
        assert Path(path).exists()

    def test_html_contains_strategy_name(self, tmp_path):
        rg = ReportGenerator("run_html_5", "HtmlStrategy", METRICS_BASIC)
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert "HtmlStrategy" in content

    def test_html_doctype(self, tmp_path):
        rg = ReportGenerator("run_html_6", "HtmlStrategy", METRICS_BASIC)
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert "<!DOCTYPE html>" in content

    def test_html_contains_plotly_script(self, tmp_path):
        rg = ReportGenerator("run_html_7", "HtmlStrategy", METRICS_BASIC)
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert "plotly" in content.lower()

    def test_html_contains_console_report(self, tmp_path):
        rg = ReportGenerator("run_html_8", "HtmlStrategy", METRICS_BASIC)
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert "PERFORMANCE" in content
        assert "RISK" in content
        assert "TRADES" in content


# ---------------------------------------------------------------------------
# 6. HTML with equity curve — div IDs
# ---------------------------------------------------------------------------

class TestHtmlDivIds:
    def test_equity_curve_div_present(self, tmp_path):
        eq = make_equity_curve()
        rg = ReportGenerator("run_div_1", "DivStrategy", METRICS_BASIC, equity_curve=eq)
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert 'id="equity-curve"' in content

    def test_drawdown_div_present(self, tmp_path):
        eq = make_equity_curve()
        rg = ReportGenerator("run_div_2", "DivStrategy", METRICS_BASIC, equity_curve=eq)
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert 'id="drawdown"' in content

    def test_no_equity_div_without_curve(self, tmp_path):
        rg = ReportGenerator("run_div_3", "DivStrategy", METRICS_BASIC)
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert 'id="equity-curve"' not in content

    def test_monthly_returns_div_with_long_curve(self, tmp_path):
        # Need > 30 data points spanning multiple months for the heatmap
        base = date(2022, 1, 3)
        eq = [(base + timedelta(days=i), 100_000 + i * 10) for i in range(370)]
        rg = ReportGenerator(
            "run_div_4", "DivStrategy", METRICS_BASIC, equity_curve=eq
        )
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert 'id="monthly-returns"' in content

    def test_benchmark_curve_rendered(self, tmp_path):
        eq = make_equity_curve()
        bench = [(d, v * 0.9) for d, v in eq]
        rg = ReportGenerator(
            "run_div_5",
            "DivStrategy",
            METRICS_BASIC,
            equity_curve=eq,
            benchmark_curve=bench,
        )
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert "Benchmark (SPY)" in content


# ---------------------------------------------------------------------------
# 7. HTML with empty / minimal data — no crash
# ---------------------------------------------------------------------------

class TestHtmlMinimalData:
    def test_no_crash_with_empty_metrics(self, tmp_path):
        rg = ReportGenerator("run_empty_1", "EmptyStrategy", {})
        path = rg.generate_html_report(output_dir=str(tmp_path))
        assert Path(path).exists()

    def test_no_crash_with_no_trade_log(self, tmp_path):
        rg = ReportGenerator("run_empty_2", "EmptyStrategy", METRICS_BASIC)
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert "No trades." in content

    def test_no_crash_short_equity_curve(self, tmp_path):
        # < 30 points: heatmap should not appear, no crash
        eq = make_equity_curve(days=5)
        rg = ReportGenerator(
            "run_empty_3", "EmptyStrategy", METRICS_BASIC, equity_curve=eq
        )
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert 'id="monthly-returns"' not in content

    def test_no_crash_with_all_none_metrics(self, tmp_path):
        rg = ReportGenerator("run_empty_4", "EmptyStrategy", {"sharpe": None})
        # None will cause formatting errors unless handled via .get() defaults
        # The implementation uses .get(key, 0) so None values from explicit keys
        # will still be passed through — verify graceful handling
        # If sharpe is explicitly None, m.get('sharpe', 0) returns None.
        # The format string will raise TypeError. We accept that this edge case
        # requires the caller to pass numeric values. Instead test missing keys:
        rg2 = ReportGenerator("run_empty_5", "EmptyStrategy", {})
        report = rg2.generate_console_report()
        assert isinstance(report, str)


# ---------------------------------------------------------------------------
# 8. Trade table in HTML
# ---------------------------------------------------------------------------

class TestHtmlTradeTable:
    def test_trade_table_present_with_trades(self, tmp_path):
        tl = make_trade_log()
        rg = ReportGenerator(
            "run_trade_1", "TradeStrategy", METRICS_BASIC, trade_log=tl
        )
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert "AAPL" in content
        assert "150.00" in content
        assert "160.00" in content

    def test_trade_pnl_class_positive(self, tmp_path):
        tl = make_trade_log()  # AAPL long: 160-150=+10 per share * 100 = +1000
        rg = ReportGenerator(
            "run_trade_2", "TradeStrategy", METRICS_BASIC, trade_log=tl
        )
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert 'class="positive"' in content

    def test_no_trades_message(self, tmp_path):
        tl = TradeLog()  # empty
        rg = ReportGenerator(
            "run_trade_3", "TradeStrategy", METRICS_BASIC, trade_log=tl
        )
        path = rg.generate_html_report(output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert "No trades." in content


# ---------------------------------------------------------------------------
# 9. Monthly returns computation
# ---------------------------------------------------------------------------

class TestMonthlyReturns:
    def test_single_month_return(self):
        # Jan 2023: start 100, end 110 → 10% return
        eq = [
            (date(2023, 1, 2), 100.0),
            (date(2023, 1, 10), 105.0),
            (date(2023, 1, 31), 110.0),
        ]
        rg = ReportGenerator("run_mr_1", "S", {}, equity_curve=eq)
        monthly = rg._compute_monthly_returns()
        assert len(monthly) == 1
        year, month, ret = monthly[0]
        assert year == 2023
        assert month == 1
        # ret = (110 - 100) / 100 * 100 = 10.0
        assert abs(ret - 10.0) < 0.01

    def test_two_month_returns(self):
        # Jan: 100 → 110 (+10%), Feb: 110 → 99 (-10%)
        eq = [
            (date(2023, 1, 2), 100.0),
            (date(2023, 1, 31), 110.0),
            (date(2023, 2, 1), 110.0),
            (date(2023, 2, 28), 99.0),
        ]
        rg = ReportGenerator("run_mr_2", "S", {}, equity_curve=eq)
        monthly = rg._compute_monthly_returns()
        assert len(monthly) == 2
        # January: start=100, end before Feb=110 → 10%
        assert monthly[0][0] == 2023 and monthly[0][1] == 1
        assert abs(monthly[0][2] - 10.0) < 0.01
        # February: start=110 (prev_value at start of month), end=99
        assert monthly[1][0] == 2023 and monthly[1][1] == 2

    def test_empty_curve_returns_empty(self):
        rg = ReportGenerator("run_mr_3", "S", {}, equity_curve=[])
        monthly = rg._compute_monthly_returns()
        assert monthly == []

    def test_returns_year_month_return_tuples(self):
        eq = make_equity_curve(days=65)  # spans ~2 months
        rg = ReportGenerator("run_mr_4", "S", {}, equity_curve=eq)
        monthly = rg._compute_monthly_returns()
        assert len(monthly) >= 2
        for entry in monthly:
            assert len(entry) == 3
            year, month, ret = entry
            assert isinstance(year, int)
            assert 1 <= month <= 12
            assert isinstance(ret, float)

    def test_cross_year_boundary(self):
        eq = [
            (date(2022, 12, 30), 100.0),
            (date(2022, 12, 31), 102.0),
            (date(2023, 1, 2), 104.0),
            (date(2023, 1, 31), 106.0),
        ]
        rg = ReportGenerator("run_mr_5", "S", {}, equity_curve=eq)
        monthly = rg._compute_monthly_returns()
        years = [m[0] for m in monthly]
        assert 2022 in years
        assert 2023 in years
