"""Trading Bot CLI — entry point for all backtest and data operations.

Usage:
    trading-bot data fetch AAPL --start 2023-01-01
    trading-bot data status
    trading-bot backtest run strategy.library.momentum.MomentumStrategy --start 2023-01-01
    trading-bot report list
    trading-bot report show <run-id>
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import typer

from analytics.reports import ReportGenerator
from config.settings import get_settings
from strategy.registry import StrategyRegistry
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App tree
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="trading-bot",
    help="Proprietary trading bot backtesting framework",
    add_completion=False,
)
data_app = typer.Typer(help="Data management commands")
backtest_app = typer.Typer(help="Backtesting commands")
report_app = typer.Typer(help="Report commands")
paper_app = typer.Typer(help="Paper trading commands")

app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(report_app, name="report")
app.add_typer(paper_app, name="paper")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FLOAT_METRIC_KEYS = {
    "total_return",
    "cagr",
    "sharpe",
    "sharpe_ratio",
    "sortino",
    "calmar",
    "max_drawdown_pct",
    "annualized_volatility",
    "var_95",
    "win_rate",
    "profit_factor",
    "expectancy",
    "benchmark_total_return",
    "alpha",
    "beta",
    "end_value",
    "start_value",
}

_INT_METRIC_KEYS = {"total_trades"}


def _coerce_metrics(metrics: dict) -> dict:
    """Coerce persisted string metric values back to numeric types.

    The engine serialises all values as strings when calling
    ``json.dumps({k: str(v) ...})``.  The ReportGenerator uses ``%f``
    formatting so values must be float/int, not str.
    """
    result = {}
    for k, v in metrics.items():
        if k in _INT_METRIC_KEYS:
            try:
                result[k] = int(float(v))
            except (TypeError, ValueError):
                result[k] = v
        elif k in _FLOAT_METRIC_KEYS:
            try:
                result[k] = float(v)
            except (TypeError, ValueError):
                result[k] = v
        else:
            # Attempt a best-effort numeric parse for unknown keys
            if isinstance(v, str):
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
            else:
                result[k] = v
    return result


def _parse_date(value: str | None, fallback: date | None = None) -> date | None:
    """Parse a YYYY-MM-DD string into a date, returning *fallback* if None."""
    if value is None:
        return fallback
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        typer.echo(f"Error: invalid date '{value}' — expected YYYY-MM-DD", err=True)
        raise typer.Exit(1) from exc


def _make_database():
    """Create and initialise the default Database."""
    from data.storage.database import Database

    settings = get_settings()
    db = Database(settings.db_path)
    db.create_tables()
    return db


def _make_feed():
    """Create an FMPFeed using configured credentials."""
    from data.feeds.fmp_feed import FMPFeed
    from utils.rate_limiter import RateLimiter

    settings = get_settings()
    return FMPFeed(
        api_key=settings.fmp_api_key,
        base_url=settings.fmp_base_url,
        rate_limiter=RateLimiter(settings.fmp_rate_limit),
    )


# ---------------------------------------------------------------------------
# DATA commands
# ---------------------------------------------------------------------------

@data_app.command("fetch")
def data_fetch(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD (default: today)"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Fetch and cache historical OHLCV data from FMP for a symbol."""
    setup_logging(log_level)

    from data.storage.cache import DataCache

    start_date = _parse_date(start)
    end_date = _parse_date(end, fallback=date.today())

    if start_date is None:
        typer.echo("Error: --start is required", err=True)
        raise typer.Exit(1)

    typer.echo(f"Fetching {symbol.upper()} from {start_date} to {end_date} …")

    try:
        db = _make_database()
        feed = _make_feed()
        cache = DataCache(db, feed)
        bars = cache.get_daily_bars(symbol.upper(), start_date, end_date)
    except Exception as exc:
        typer.echo(f"Error fetching data: {exc}", err=True)
        logger.exception("data fetch failed")
        raise typer.Exit(1) from exc

    if not bars:
        typer.echo("No bars returned — check symbol and date range.")
        raise typer.Exit(0)

    actual_start = bars[0].date
    actual_end = bars[-1].date
    typer.echo(
        f"Cached {len(bars)} bars for {symbol.upper()}  "
        f"({actual_start} → {actual_end})"
    )


@data_app.command("validate")
def data_validate(
    symbol: Optional[str] = typer.Option(None, help="Single symbol to validate"),
    all: bool = typer.Option(False, "--all", help="Validate all cached symbols"),
    report: bool = typer.Option(False, "--report", help="Show full quality report"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Run data quality validation on cached bar data."""
    setup_logging(log_level)

    from data.validation.quality_report import generate_quality_report
    from data.validation.validators import validate_bar, validate_gap

    db = _make_database()

    # Determine which symbols to validate
    if all:
        rows = db.execute(
            "SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol"
        ).fetchall()
        symbols = [r[0] for r in rows]
    elif symbol:
        symbols = [symbol.upper()]
    else:
        typer.echo(
            "Error: provide --symbol TICKER or --all", err=True
        )
        raise typer.Exit(1)

    if not symbols:
        typer.echo("No cached data found.")
        raise typer.Exit(0)

    for sym in symbols:
        min_date, max_date = db.get_cached_date_range(sym)
        if min_date is None:
            typer.echo(f"{sym}: no data cached")
            continue

        bars = db.get_daily_bars(sym, min_date, max_date)
        all_issues = []
        prev = None
        for bar in bars:
            issues = validate_bar(bar, prev_bar=prev)
            all_issues.extend(issues)
            prev = bar

        gap_issues = validate_gap(bars)
        all_issues.extend(gap_issues)

        error_count = sum(1 for i in all_issues if i.severity == "error")
        warn_count = sum(1 for i in all_issues if i.severity == "warning")

        typer.echo(
            f"{sym}: {len(bars)} bars  |  "
            f"errors={error_count}  warnings={warn_count}"
        )

        if report and all_issues:
            quality = generate_quality_report(bars, all_issues)
            typer.echo(json.dumps(quality, indent=2, default=str))


@data_app.command("status")
def data_status(
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Show all cached symbols with their date ranges and bar counts."""
    setup_logging(log_level)

    db = _make_database()

    rows = db.execute(
        """
        SELECT symbol, MIN(date) AS min_date, MAX(date) AS max_date,
               COUNT(*) AS bar_count
        FROM daily_bars
        GROUP BY symbol
        ORDER BY symbol
        """
    ).fetchall()

    if not rows:
        typer.echo("No data cached yet. Run 'trading-bot data fetch' to populate.")
        return

    header = f"{'Symbol':<12} {'From':<12} {'To':<12} {'Bars':>6}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for row in rows:
        typer.echo(
            f"{row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:>6}"
        )


# ---------------------------------------------------------------------------
# BACKTEST commands
# ---------------------------------------------------------------------------

@backtest_app.command("run")
def backtest_run(
    strategy: str = typer.Argument(
        ...,
        help=(
            "Strategy ref: Python dotted path (e.g. strategy.library.momentum.MomentumStrategy)"
            " or path to a .yaml config file."
        ),
    ),
    start: str = typer.Option(..., help="Backtest start date YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, help="Backtest end date (default: today)"),
    capital: float = typer.Option(100_000.0, help="Initial capital in USD"),
    universe: Optional[str] = typer.Option(
        None, help="Comma-separated tickers (e.g. AAPL,MSFT). Overrides strategy universe."
    ),
    benchmark: str = typer.Option("SPY", help="Benchmark symbol"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Run a backtest with the specified strategy."""
    setup_logging(log_level)

    from core.engine import BacktestEngine
    from strategy.config_strategy import ConfigStrategy

    start_date = _parse_date(start)
    end_date = _parse_date(end, fallback=date.today())

    if start_date is None:
        typer.echo("Error: --start is required", err=True)
        raise typer.Exit(1)

    # Resolve strategy
    try:
        if strategy.endswith((".yaml", ".yml")):
            strat_obj = ConfigStrategy(strategy)
            resolved_universe: list[str] = (
                universe.upper().split(",") if universe
                else getattr(strat_obj, "universe", [])
            )
        else:
            strat_obj = StrategyRegistry.resolve(strategy)
            resolved_universe = (
                universe.upper().split(",")
                if universe
                else getattr(strat_obj, "universe", [])
            )
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Error resolving strategy: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not resolved_universe:
        typer.echo(
            "Error: no universe specified. Pass --universe AAPL,MSFT or embed in strategy config.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(
        f"Running backtest: {strategy}  |  "
        f"{start_date} → {end_date}  |  "
        f"capital=${capital:,.0f}  |  "
        f"universe={resolved_universe}"
    )

    db = _make_database()

    try:
        engine = BacktestEngine(
            strategy=strat_obj,
            database=db,
            universe=resolved_universe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=capital,
            benchmark_symbol=benchmark,
        )
        metrics = engine.run()
    except Exception as exc:
        typer.echo(f"Backtest failed: {exc}", err=True)
        logger.exception("backtest run failed")
        raise typer.Exit(1) from exc

    # The engine already prints a console report; emit the run summary too
    typer.echo(
        f"\nTotal return: {metrics.get('total_return', 0) * 100:.2f}%  |  "
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}  |  "
        f"Trades: {metrics.get('total_trades', 0)}"
    )


@backtest_app.command("compare")
def backtest_compare(
    strategies: str = typer.Argument(..., help="Comma-separated strategy references"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, help="End date YYYY-MM-DD"),
    capital: float = typer.Option(100_000.0, help="Initial capital"),
) -> None:
    """Compare multiple strategies. Strategies are comma-separated."""
    typer.echo("Not yet implemented — coming in v2")


@backtest_app.command("optimize")
def backtest_optimize(
    strategy: str = typer.Argument(..., help="Strategy reference"),
    param: list[str] = typer.Option(None, "--param", help="param=start:stop:step"),
    objective: str = typer.Option("sharpe", help="Optimisation objective"),
) -> None:
    """Grid-search parameter optimisation for a strategy."""
    typer.echo("Not yet implemented — see Task 20")


# ---------------------------------------------------------------------------
# REPORT commands
# ---------------------------------------------------------------------------

@report_app.command("list")
def report_list(
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """List all historical backtest runs."""
    setup_logging(log_level)

    db = _make_database()
    runs = db.list_runs()

    if not runs:
        typer.echo("No runs found. Run 'trading-bot backtest run' first.")
        return

    header = f"{'Run ID':<12} {'Strategy':<30} {'Start':<12} {'End':<12} {'Return':>8} {'Sharpe':>7}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for run in runs:
        total_ret = (run.total_return or 0) * 100
        typer.echo(
            f"{run.run_id:<12} {(run.strategy_name or ''):<30} "
            f"{str(run.start_date or ''):<12} {str(run.end_date or ''):<12} "
            f"{total_ret:>7.1f}% {(run.sharpe or 0):>7.2f}"
        )


@report_app.command("last")
def report_last(
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Show the most recent backtest report."""
    setup_logging(log_level)

    db = _make_database()
    runs = db.list_runs()

    if not runs:
        typer.echo("No runs found.")
        raise typer.Exit(0)

    run = runs[0]

    # Reconstruct metrics from the persisted full_metrics JSON
    raw_metrics: dict = {}
    if run.full_metrics:
        try:
            raw_metrics = json.loads(run.full_metrics)
        except json.JSONDecodeError:
            pass
    metrics = _coerce_metrics(raw_metrics)
    metrics.setdefault("start_date", str(run.start_date))
    metrics.setdefault("end_date", str(run.end_date))
    metrics.setdefault("total_return", run.total_return or 0)
    metrics.setdefault("sharpe_ratio", run.sharpe or 0)
    metrics.setdefault("max_drawdown_pct", run.max_drawdown or 0)
    metrics.setdefault("total_trades", 0)

    report = ReportGenerator(
        run_id=run.run_id,
        strategy_name=run.strategy_name or "Unknown",
        metrics=metrics,
    )
    typer.echo(report.generate_console_report())


@report_app.command("show")
def report_show(
    run_id: str = typer.Argument(..., help="Run ID to display"),
    format: str = typer.Option("console", help="Output format: console or html"),
    output_dir: str = typer.Option("reports", help="Output directory for HTML reports"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Generate a report for a specific run ID."""
    setup_logging(log_level)

    db = _make_database()
    run = db.get_run(run_id)

    if run is None:
        typer.echo(f"Error: run '{run_id}' not found.", err=True)
        raise typer.Exit(1)

    raw_metrics2: dict = {}
    if run.full_metrics:
        try:
            raw_metrics2 = json.loads(run.full_metrics)
        except json.JSONDecodeError:
            pass
    metrics2 = _coerce_metrics(raw_metrics2)
    metrics2.setdefault("start_date", str(run.start_date))
    metrics2.setdefault("end_date", str(run.end_date))
    metrics2.setdefault("total_return", run.total_return or 0)
    metrics2.setdefault("sharpe_ratio", run.sharpe or 0)
    metrics2.setdefault("max_drawdown_pct", run.max_drawdown or 0)
    metrics2.setdefault("total_trades", 0)

    report = ReportGenerator(
        run_id=run.run_id,
        strategy_name=run.strategy_name or "Unknown",
        metrics=metrics2,
    )

    if format == "html":
        path = report.generate_html_report(output_dir=output_dir)
        typer.echo(f"HTML report written to: {path}")
    else:
        typer.echo(report.generate_console_report())


@report_app.command("monte-carlo")
def report_monte_carlo(
    run_id: str = typer.Argument(..., help="Run ID for Monte Carlo simulation"),
    simulations: int = typer.Option(10_000, help="Number of simulations"),
) -> None:
    """Run Monte Carlo simulation on a past backtest."""
    typer.echo("Not yet implemented — coming in v2")


# ---------------------------------------------------------------------------
# PAPER commands
# ---------------------------------------------------------------------------

@paper_app.command("start")
def paper_start(
    strategy: str = typer.Argument(..., help="Strategy reference"),
    capital: float = typer.Option(100_000.0, help="Starting capital"),
) -> None:
    """Start paper trading a strategy (live simulation, no real money)."""
    typer.echo("Paper trading not yet implemented — coming in v2")


@paper_app.command("status")
def paper_status() -> None:
    """Show active paper trading session status."""
    typer.echo("Paper trading not yet implemented — coming in v2")


@paper_app.command("stop")
def paper_stop() -> None:
    """Stop the active paper trading session."""
    typer.echo("Paper trading not yet implemented — coming in v2")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
