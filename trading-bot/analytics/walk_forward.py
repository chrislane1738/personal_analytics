"""Walk-forward optimization engine: rolling window generation, grid search,
and out-of-sample equity curve stitching.

This is a pure computation module — no database persistence or API calls.
The service layer is responsible for persisting results.
"""

from __future__ import annotations

import itertools
import threading
from datetime import date
from typing import Any, Callable, Optional

import numpy as np

from analytics.metrics import compute_all_metrics
from core.engine import BacktestEngine


# ---------------------------------------------------------------------------
# Objective metric mapping
# ---------------------------------------------------------------------------

_OBJECTIVE_MAP: dict[str, str] = {
    "sharpe": "sharpe_ratio",
    "sortino": "sortino_ratio",
    "calmar": "calmar_ratio",
    "total_return": "total_return",
    "profit_factor": "profit_factor",
}


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------

def _add_months(d: date, months: int) -> date:
    """Add *months* calendar months to date *d*, clamping day to month end."""
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    # Clamp day to the last day of the target month
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    day = min(d.day, max_day)
    return date(year, month, day)


def generate_windows(
    start_date: date,
    end_date: date,
    train_months: int,
    oos_months: int,
    step_months: int,
) -> list[dict]:
    """Generate rolling walk-forward window date ranges.

    Each window is a dict with keys:
        train_start, train_end, oos_start, oos_end

    The OOS start equals the train end (no gap).  Windows step forward by
    *step_months*.  Incomplete trailing windows (OOS extends beyond *end_date*)
    are dropped.

    Raises ValueError if fewer than 2 complete windows can be formed.
    """
    windows: list[dict] = []
    offset = 0

    while True:
        train_start = _add_months(start_date, offset)
        train_end = _add_months(start_date, offset + train_months)
        oos_start = train_end
        oos_end = _add_months(start_date, offset + train_months + oos_months)

        # Drop if OOS extends beyond the available data
        if oos_end > end_date:
            break

        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
        })

        offset += step_months

    if len(windows) < 2:
        raise ValueError(
            f"Date range {start_date} to {end_date} with train={train_months}m, "
            f"oos={oos_months}m, step={step_months}m produces {len(windows)} "
            f"window(s); at least 2 are required."
        )

    return windows


# ---------------------------------------------------------------------------
# Parameter grid generation
# ---------------------------------------------------------------------------

def generate_param_grid(
    param_space: dict,
    max_combinations: int = 5000,
) -> list[dict]:
    """Generate all parameter combinations from a param space definition.

    Input format::

        {
            "rsi_thresh": {"type": "strategy", "min": 25, "max": 50, "step": 5},
            "bb_std":     {"type": "strategy", "min": 1.5, "max": 2.5, "step": 0.5},
        }

    Returns a list of dicts, each mapping param name to a concrete value.
    If the param space is empty, returns ``[{}]`` (one combo with no params).

    Raises ValueError if total combinations exceed *max_combinations*.
    """
    if not param_space:
        return [{}]

    names: list[str] = []
    value_lists: list[list] = []

    for name, spec in param_space.items():
        lo = spec["min"]
        hi = spec["max"]
        step = spec["step"]

        # Use numpy arange for correct float handling
        values = np.arange(lo, hi + step * 0.5, step).tolist()
        # Round to avoid floating-point drift
        if isinstance(lo, float) or isinstance(step, float):
            decimals = max(
                len(str(lo).split(".")[-1]) if "." in str(lo) else 0,
                len(str(step).split(".")[-1]) if "." in str(step) else 0,
            )
            values = [round(v, decimals) for v in values]
        else:
            values = [int(round(v)) for v in values]

        names.append(name)
        value_lists.append(values)

    # Check total combinations
    total = 1
    for vals in value_lists:
        total *= len(vals)
    if total > max_combinations:
        raise ValueError(
            f"Parameter grid would produce {total} combinations, "
            f"which would exceed the maximum of {max_combinations}."
        )

    # Generate cartesian product
    grid: list[dict] = []
    for combo in itertools.product(*value_lists):
        grid.append(dict(zip(names, combo)))

    return grid


# ---------------------------------------------------------------------------
# Walk-forward orchestrator
# ---------------------------------------------------------------------------

def run_walk_forward(
    strategy_loader: Callable[[dict], Any],
    param_space: dict,
    database: Any,
    universe: list[str],
    start_date: date,
    end_date: date,
    initial_capital: float,
    benchmark: str,
    train_months: int,
    oos_months: int,
    step_months: int,
    objective: str,
    engine_params: Optional[dict] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Run rolling walk-forward optimization over the given date range.

    Parameters
    ----------
    strategy_loader:
        Callable that accepts a params dict and returns a Strategy instance.
    param_space:
        Dict of parameter definitions.  Each entry has ``type`` ("strategy"
        or "engine"), ``min``, ``max``, ``step``.  Can be empty for fixed-
        parameter runs (no optimization).
    database:
        Database instance with ``get_daily_bars`` for loading bar data.
    universe:
        List of ticker symbols.
    start_date / end_date:
        Overall date range.
    initial_capital:
        Starting cash.
    benchmark:
        Benchmark symbol (e.g. "SPY").
    train_months / oos_months / step_months:
        Window sizing parameters.
    objective:
        Optimization objective: "sharpe", "sortino", "calmar",
        "total_return", or "profit_factor".
    engine_params:
        Optional dict of engine-level params (e.g. ``position_size_pct``).
    cancel_event:
        Optional threading.Event; if set, the walk-forward stops early.
    progress_callback:
        Optional callable(windows_completed, windows_total, phase).

    Returns
    -------
    dict with keys: ``windows``, ``aggregate``, ``stitched_equity_curve``,
    ``parameter_stability``.
    """
    # Resolve the objective metric key
    metric_key = _OBJECTIVE_MAP.get(objective)
    if metric_key is None:
        raise ValueError(
            f"Unknown objective {objective!r}. "
            f"Valid options: {sorted(_OBJECTIVE_MAP)}"
        )

    # 1. Generate windows
    windows = generate_windows(
        start_date, end_date, train_months, oos_months, step_months,
    )
    total_windows = len(windows)

    # Separate strategy params from engine params in the param space
    strategy_param_space: dict = {}
    engine_param_space: dict = {}
    for pname, pspec in param_space.items():
        if pspec.get("type") == "engine":
            engine_param_space[pname] = pspec
        else:
            strategy_param_space[pname] = pspec

    # Base engine kwargs (from engine_params arg, not from grid)
    base_engine_kwargs: dict = dict(engine_params) if engine_params else {}

    # 2. Iterate through windows
    current_capital = initial_capital
    window_results: list[dict] = []
    all_oos_equity: list[tuple[date, float]] = []
    all_oos_benchmark: list[tuple[date, float]] = []
    parameter_stability: dict[str, list] = {
        name: [] for name in param_space
    }

    for win_idx, window in enumerate(windows):
        # Check cancellation
        if cancel_event and cancel_event.is_set():
            break

        train_start = window["train_start"]
        train_end = window["train_end"]
        oos_start = window["oos_start"]
        oos_end = window["oos_end"]

        best_strategy_params: dict = {}
        best_engine_params: dict = dict(base_engine_kwargs)
        train_metrics: dict = {}

        # 3a. Grid search on training window (if param_space is non-empty)
        if param_space:
            # Build combined grid from strategy + engine param spaces
            combined_space = {}
            combined_space.update(strategy_param_space)
            combined_space.update(engine_param_space)
            param_grid = generate_param_grid(combined_space)

            best_score = float("-inf")
            best_combo: dict = {}

            for combo in param_grid:
                # Split combo into strategy params and engine params
                s_params = {
                    k: v for k, v in combo.items()
                    if k in strategy_param_space
                }
                e_params = {
                    k: v for k, v in combo.items()
                    if k in engine_param_space
                }

                # Merge engine grid params with base engine kwargs
                merged_engine = dict(base_engine_kwargs)
                merged_engine.update(e_params)

                strategy = strategy_loader(s_params)
                engine = BacktestEngine(
                    strategy=strategy,
                    database=database,
                    universe=universe,
                    start_date=train_start,
                    end_date=train_end,
                    initial_capital=current_capital,
                    benchmark_symbol=benchmark,
                    quiet=True,
                    **merged_engine,
                )
                metrics = engine.run()

                score = metrics.get(metric_key, float("-inf"))
                if score is None:
                    score = float("-inf")
                if score > best_score:
                    best_score = score
                    best_combo = combo
                    train_metrics = metrics

            # Split best combo
            best_strategy_params = {
                k: v for k, v in best_combo.items()
                if k in strategy_param_space
            }
            best_engine_grid_params = {
                k: v for k, v in best_combo.items()
                if k in engine_param_space
            }
            best_engine_params = dict(base_engine_kwargs)
            best_engine_params.update(best_engine_grid_params)

        # 3b. Run best params on OOS window
        strategy = strategy_loader(best_strategy_params)
        oos_engine = BacktestEngine(
            strategy=strategy,
            database=database,
            universe=universe,
            start_date=oos_start,
            end_date=oos_end,
            initial_capital=current_capital,
            benchmark_symbol=benchmark,
            quiet=True,
            **best_engine_params,
        )
        oos_metrics = oos_engine.run()

        # Collect OOS equity curve and trades
        oos_equity_curve = oos_engine.portfolio.equity_curve
        oos_benchmark_curve = oos_engine.benchmark.equity_curve
        oos_trades = oos_engine.trade_log.get_trade_dicts()

        # 3c. Update current capital to final equity of OOS window
        if oos_equity_curve:
            current_capital = oos_equity_curve[-1][1]
        else:
            # No data in the OOS window — keep capital unchanged
            current_capital = oos_metrics.get("end_value", current_capital)

        # Build OOS equity curve entries for the result
        oos_ec_entries: list[dict] = []
        benchmark_dict: dict[date, float] = {
            d: v for d, v in oos_benchmark_curve
        }
        for d, sv in oos_equity_curve:
            oos_ec_entries.append({
                "date": str(d),
                "strategy_value": sv,
                "benchmark_value": benchmark_dict.get(d, 0.0),
            })

        # Accumulate for stitching
        all_oos_equity.extend(oos_equity_curve)
        all_oos_benchmark.extend(oos_benchmark_curve)

        # Record per-window results
        window_result = {
            "window_index": win_idx,
            "train_start": str(train_start),
            "train_end": str(train_end),
            "oos_start": str(oos_start),
            "oos_end": str(oos_end),
            "best_params": {**best_strategy_params, **{
                k: v for k, v in best_engine_params.items()
                if k in engine_param_space
            }},
            "train_metrics": train_metrics,
            "oos_metrics": oos_metrics,
            "oos_trades": oos_trades,
            "oos_equity_curve": oos_ec_entries,
        }
        window_results.append(window_result)

        # Track parameter stability
        all_best = {**best_strategy_params, **{
            k: v for k, v in best_engine_params.items()
            if k in engine_param_space
        }}
        for pname in parameter_stability:
            parameter_stability[pname].append(all_best.get(pname))

        # Progress callback
        if progress_callback:
            progress_callback(win_idx + 1, total_windows, "oos")

    # 4. Stitch all OOS equity curves
    stitched_equity_curve: list[dict] = []
    stitched_benchmark_dict: dict[date, float] = {
        d: v for d, v in all_oos_benchmark
    }
    for d, sv in all_oos_equity:
        stitched_equity_curve.append({
            "date": str(d),
            "strategy_value": sv,
            "benchmark_value": stitched_benchmark_dict.get(d, 0.0),
        })

    # 5. Compute aggregate metrics on the stitched curve
    all_trades: list[dict] = []
    for w in window_results:
        all_trades.extend(w["oos_trades"])

    aggregate: dict = {}
    if all_oos_equity:
        aggregate = compute_all_metrics(
            equity_curve=all_oos_equity,
            trades=all_trades,
            benchmark_curve=all_oos_benchmark if all_oos_benchmark else None,
            risk_free_rate=0.04,
            initial_capital=initial_capital,
        )
    else:
        aggregate = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
        }

    return {
        "windows": window_results,
        "aggregate": aggregate,
        "stitched_equity_curve": stitched_equity_curve,
        "parameter_stability": parameter_stability,
    }
