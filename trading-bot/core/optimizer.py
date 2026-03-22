"""Parameter optimizer and walk-forward validation for backtest strategies.

Provides grid search over parameter combinations and rolling walk-forward
validation to detect overfitting.
"""

from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date
from itertools import product


@dataclass
class OptimizationResult:
    """Results from a grid search optimization run."""

    param_combinations: list[dict]
    """Every parameter combination that was evaluated."""

    metrics: list[dict]
    """Metrics dict for each combination, in the same order as param_combinations."""

    best_params: dict
    """Parameter combination that produced the best objective value."""

    best_metric_value: float
    """Objective metric value for best_params."""

    objective: str
    """Name of the metric used as the optimization objective."""

    edge_warning: bool
    """True when the best parameter value sits at the boundary of the search grid."""

    ranked_results: list[dict]
    """All results sorted by objective metric, best first.

    Each entry is ``{"params": dict, "metrics": dict}``.
    """


@dataclass
class WalkForwardResult:
    """Results from a rolling walk-forward validation run."""

    windows: list[dict]
    """Per-window results.

    Each entry contains:
    ``train_start``, ``train_end``, ``test_start``, ``test_end``,
    ``best_params``, ``in_sample`` (metric), ``out_of_sample`` (metric).
    """

    avg_in_sample_metric: float
    """Average objective value across all in-sample (training) windows."""

    avg_out_of_sample_metric: float
    """Average objective value across all out-of-sample (test) windows."""

    degradation_pct: float
    """Percentage drop from in-sample to out-of-sample performance.

    Computed as ``(in_sample - out_of_sample) / abs(in_sample) * 100``.
    """

    overfitting_detected: bool
    """True when degradation_pct exceeds the configured threshold."""

    objective: str
    """Name of the metric used as the optimization objective."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deep_copy_config(config: dict) -> dict:
    """Return a deep copy of a config dict."""
    return copy.deepcopy(config)


def _apply_params(config: dict, params: dict) -> None:
    """Apply parameter overrides to a strategy config dict in-place.

    Convention: a param key like ``"rsi_period"`` is split on the *last*
    underscore into ``prefix="rsi"`` and ``param_name="period"``.  The first
    indicator whose name starts with ``prefix`` has its ``param_name`` key
    updated to ``value``.

    Examples::

        "rsi_period"  -> indicators["rsi_14"]["period"] = value
        "sma_period"  -> indicators["sma_20"]["period"] = value
        "bb_std_dev"  -> indicators["bb_20"]["std_dev"] = value
    """
    indicators = config.get("indicators", {})
    for param_key, value in params.items():
        parts = param_key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        prefix, param_name = parts

        for ind_name, ind_config in indicators.items():
            if ind_name.startswith(prefix):
                ind_config[param_name] = value
                break


def _run_backtest(
    config_path: str,
    database,
    universe: list[str],
    start_date: date,
    end_date: date,
    initial_capital: float,
) -> dict:
    """Instantiate a ConfigStrategy + BacktestEngine and return metrics."""
    from core.engine import BacktestEngine
    from strategy.config_strategy import ConfigStrategy

    strategy = ConfigStrategy(config_path)
    engine = BacktestEngine(
        strategy=strategy,
        database=database,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )
    return engine.run()


def _backtest_with_params(
    base_config: dict,
    params: dict,
    database,
    universe: list[str],
    start_date: date,
    end_date: date,
    initial_capital: float,
) -> dict:
    """Apply *params* to a copy of *base_config*, write a temp YAML, and run.

    Returns the metrics dict on success or ``{objective: -inf, error: ...}``
    on failure.
    """
    import yaml

    config = _deep_copy_config(base_config)
    _apply_params(config, params)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(config, tmp)
        tmp_path = tmp.name

    try:
        return _run_backtest(
            tmp_path, database, universe, start_date, end_date, initial_capital
        )
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# ParameterOptimizer
# ---------------------------------------------------------------------------

class ParameterOptimizer:
    """Grid-search and walk-forward validation for config-driven strategies.

    Parameters
    ----------
    strategy_config_path:
        Path to the base YAML strategy config file.
    database:
        Database instance with ``get_daily_bars`` / ``insert_run`` methods.
    universe:
        List of ticker symbols to trade during each backtest.
    initial_capital:
        Starting cash for each backtest run.
    objective:
        Metric to optimise.  One of ``"sharpe_ratio"``, ``"sortino_ratio"``,
        ``"total_return"``, or ``"max_drawdown_pct"``.
        For ``"max_drawdown_pct"`` lower is better; the grid search still
        ranks by the raw value so callers should be aware that a *less
        negative* value wins.
    """

    def __init__(
        self,
        strategy_config_path: str,
        database,
        universe: list[str],
        initial_capital: float = 100_000.0,
        objective: str = "sharpe_ratio",
    ) -> None:
        self.config_path = strategy_config_path
        self.database = database
        self.universe = universe
        self.initial_capital = initial_capital
        self.objective = objective

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grid_search(
        self,
        param_grid: dict[str, list],
        start_date: date,
        end_date: date,
    ) -> OptimizationResult:
        """Evaluate every combination in *param_grid* and return ranked results.

        Parameters
        ----------
        param_grid:
            Mapping from parameter names to lists of candidate values, e.g.
            ``{"rsi_period": [10, 14, 20], "sma_period": [15, 20, 30]}``.
        start_date / end_date:
            Date range for each backtest.

        Returns
        -------
        OptimizationResult
            Contains all combinations tried, their metrics, the best parameter
            set, an edge-of-grid warning flag, and a ranked result list.
        """
        import yaml

        with open(self.config_path) as f:
            base_config = yaml.safe_load(f)

        param_names = list(param_grid.keys())
        param_values = [param_grid[k] for k in param_names]
        combinations = [
            dict(zip(param_names, combo)) for combo in product(*param_values)
        ]

        results: list[dict] = []
        for combo in combinations:
            try:
                metrics = _backtest_with_params(
                    base_config,
                    combo,
                    self.database,
                    self.universe,
                    start_date,
                    end_date,
                    self.initial_capital,
                )
                results.append({"params": combo, "metrics": metrics})
            except Exception as exc:
                results.append(
                    {
                        "params": combo,
                        "metrics": {self.objective: float("-inf")},
                        "error": str(exc),
                    }
                )

        # Rank by objective (higher = better for all supported metrics)
        ranked = sorted(
            results,
            key=lambda r: r["metrics"].get(self.objective, float("-inf")),
            reverse=True,
        )

        best = ranked[0] if ranked else None

        # Edge-of-grid warning: best value is at min or max of any param range
        edge_warning = False
        if best:
            for param_name, values in param_grid.items():
                best_val = best["params"].get(param_name)
                if len(values) > 1 and (
                    best_val == min(values) or best_val == max(values)
                ):
                    edge_warning = True
                    break

        return OptimizationResult(
            param_combinations=combinations,
            metrics=[r["metrics"] for r in results],
            best_params=best["params"] if best else {},
            best_metric_value=best["metrics"].get(self.objective, 0.0)
            if best
            else 0.0,
            objective=self.objective,
            edge_warning=edge_warning,
            ranked_results=ranked,
        )

    def walk_forward(
        self,
        param_grid: dict[str, list],
        start_date: date,
        end_date: date,
        train_months: int = 48,
        test_months: int = 12,
        overfitting_threshold: float = 50.0,
    ) -> WalkForwardResult:
        """Rolling walk-forward validation.

        For each window the optimizer:

        1. Runs a full ``grid_search`` over the training period.
        2. Takes the best parameters and evaluates them on the following test
           period (out-of-sample).
        3. Slides the window forward by *test_months* and repeats.

        Parameters
        ----------
        param_grid:
            Parameter search space (same format as :meth:`grid_search`).
        start_date / end_date:
            Overall date range.  Must be wide enough to contain at least one
            complete train+test window.
        train_months:
            Length of the in-sample training window in months.
        test_months:
            Length of the out-of-sample test window in months.
        overfitting_threshold:
            Degradation percentage above which ``overfitting_detected`` is set
            to ``True`` (default 50 %).

        Returns
        -------
        WalkForwardResult
        """
        import yaml

        windows: list[dict] = []
        window_start = start_date

        while True:
            train_end = _add_months(window_start, train_months)
            test_start = train_end
            test_end = _add_months(test_start, test_months)

            if test_end > end_date:
                break

            # --- In-sample optimisation ---
            train_result = self.grid_search(param_grid, window_start, train_end)
            best_params = train_result.best_params
            in_sample_metric = train_result.best_metric_value

            # --- Out-of-sample validation ---
            with open(self.config_path) as f:
                base_config = yaml.safe_load(f)

            try:
                test_metrics = _backtest_with_params(
                    base_config,
                    best_params,
                    self.database,
                    self.universe,
                    test_start,
                    test_end,
                    self.initial_capital,
                )
                out_of_sample_metric = test_metrics.get(self.objective, 0.0)
            except Exception:
                out_of_sample_metric = 0.0

            windows.append(
                {
                    "train_start": window_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "best_params": best_params,
                    "in_sample": in_sample_metric,
                    "out_of_sample": out_of_sample_metric,
                }
            )

            # Slide window forward by test_months
            window_start = _add_months(window_start, test_months)

        if not windows:
            return WalkForwardResult(
                windows=[],
                avg_in_sample_metric=0.0,
                avg_out_of_sample_metric=0.0,
                degradation_pct=0.0,
                overfitting_detected=False,
                objective=self.objective,
            )

        avg_in = sum(w["in_sample"] for w in windows) / len(windows)
        avg_out = sum(w["out_of_sample"] for w in windows) / len(windows)
        degradation = (
            (avg_in - avg_out) / abs(avg_in) * 100 if avg_in != 0 else 0.0
        )

        return WalkForwardResult(
            windows=windows,
            avg_in_sample_metric=avg_in,
            avg_out_of_sample_metric=avg_out,
            degradation_pct=degradation,
            overfitting_detected=degradation > overfitting_threshold,
            objective=self.objective,
        )


# ---------------------------------------------------------------------------
# Date arithmetic helper (no external dependencies)
# ---------------------------------------------------------------------------

def _add_months(d: date, months: int) -> date:
    """Return *d* offset forward by *months* calendar months.

    Handles month-end clamping (e.g. Jan 31 + 1 month = Feb 28/29).
    """
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    day = min(d.day, last_day)
    return date(year, month, day)
