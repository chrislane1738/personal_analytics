"""Monte Carlo simulation for trading strategy robustness analysis."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MonteCarloResult:
    simulations: int
    median_final_equity: float
    percentile_5: float
    percentile_95: float
    actual_final_equity: float
    probability_of_ruin: float  # % of simulations where equity hit floor
    max_drawdown_median: float
    max_drawdown_95: float
    is_outlier: bool  # True if actual result is outside 5-95 percentile band
    equity_distribution: list[float]  # final equity of each simulation
    drawdown_distribution: list[float]  # max drawdown of each simulation
    percentile_bands: dict[str, list[float]]  # keys p5,p25,p50,p75,p95 — equity per time step
    actual_curve: list[float]  # actual equity at each trade step


def run_monte_carlo(
    trade_pnls: list[float],
    initial_capital: float = 100000.0,
    num_simulations: int = 10000,
    ruin_threshold: float = 0.0,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation by shuffling trade order.

    Takes the actual trade P&Ls, randomly shuffles their order N times,
    and replays the equity curve for each shuffle to test robustness.

    Args:
        trade_pnls: List of P&L values from each trade (e.g., [500, -200, 300, -100])
        initial_capital: Starting portfolio value
        num_simulations: Number of random shuffles to run
        ruin_threshold: Equity level considered "ruin" (default 0 = total loss)
        seed: Random seed for reproducibility (None = random)

    Returns: MonteCarloResult with statistics across all simulations
    """
    rng = np.random.default_rng(seed)

    max_drawdowns = []
    ruin_count = 0

    pnls = np.array(trade_pnls)
    num_trades = len(pnls)

    # Compute actual equity curve (original trade order)
    if num_trades > 0:
        actual_equity = float(initial_capital + np.cumsum(pnls)[-1])
        actual_curve = (initial_capital + np.cumsum(pnls)).tolist()
    else:
        actual_equity = initial_capital
        actual_curve = []

    # Store ALL simulation equity paths in a matrix (num_simulations, num_trades)
    if num_trades > 0:
        all_paths = np.empty((num_simulations, num_trades))
    else:
        all_paths = np.empty((num_simulations, 0))

    for sim_idx in range(num_simulations):
        # Shuffle trade order
        shuffled = rng.permutation(pnls)

        if num_trades == 0:
            # No trades: equity stays at initial_capital throughout
            max_drawdowns.append(0.0)
            continue

        # Replay equity curve
        equity_curve = initial_capital + np.cumsum(shuffled)
        all_paths[sim_idx, :] = equity_curve

        # Compute max drawdown for this simulation
        running_max = np.maximum.accumulate(np.concatenate([[initial_capital], equity_curve]))
        drawdowns = (running_max[1:] - equity_curve) / running_max[1:]  # as positive percentage
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        max_drawdowns.append(max_dd)

        # Check ruin
        if np.any(equity_curve <= ruin_threshold):
            ruin_count += 1

    # Extract final equities from the paths matrix
    if num_trades > 0:
        final_equities = all_paths[:, -1]
    else:
        final_equities = np.full(num_simulations, initial_capital)

    max_drawdowns = np.array(max_drawdowns)

    p5 = float(np.percentile(final_equities, 5))
    p95 = float(np.percentile(final_equities, 95))

    # Compute percentile bands per time step
    if num_trades > 0:
        percentile_bands = {
            "p5": np.percentile(all_paths, 5, axis=0).tolist(),
            "p25": np.percentile(all_paths, 25, axis=0).tolist(),
            "p50": np.percentile(all_paths, 50, axis=0).tolist(),
            "p75": np.percentile(all_paths, 75, axis=0).tolist(),
            "p95": np.percentile(all_paths, 95, axis=0).tolist(),
        }
    else:
        percentile_bands = {"p5": [], "p25": [], "p50": [], "p75": [], "p95": []}

    return MonteCarloResult(
        simulations=num_simulations,
        median_final_equity=float(np.median(final_equities)),
        percentile_5=p5,
        percentile_95=p95,
        actual_final_equity=actual_equity,
        probability_of_ruin=ruin_count / num_simulations,
        max_drawdown_median=float(np.median(max_drawdowns)),
        max_drawdown_95=float(np.percentile(max_drawdowns, 95)),
        is_outlier=bool(actual_equity < p5 or actual_equity > p95),
        equity_distribution=final_equities.tolist(),
        drawdown_distribution=max_drawdowns.tolist(),
        percentile_bands=percentile_bands,
        actual_curve=actual_curve,
    )


# ---------------------------------------------------------------------------
# Helpers for window-level Monte Carlo
# ---------------------------------------------------------------------------


def _compute_sharpe_from_equity(equity_curve: np.ndarray) -> float:
    """Compute annualised Sharpe ratio from an equity curve array."""
    if len(equity_curve) < 2:
        return 0.0
    prev = equity_curve[:-1]
    # Guard against zero/negative equity values causing divide-by-zero
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(prev > 0, np.diff(equity_curve) / prev, 0.0)
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    std = float(np.std(returns))
    if std == 0.0:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(252))


def _compute_max_drawdown(equity_curve: np.ndarray, initial_capital: float) -> float:
    """Compute max drawdown as a positive fraction in [0, 1]."""
    full_curve = np.concatenate([[initial_capital], equity_curve])
    running_max = np.maximum.accumulate(full_curve)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(
            running_max[1:] > 0,
            (running_max[1:] - equity_curve) / running_max[1:],
            0.0,
        )
    drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


def _replay_windows(
    windows: list[list[float]],
    initial_capital: float,
    num_simulations: int,
    ruin_threshold: float,
    rng: np.random.Generator,
    sample_mode: str,  # "shuffle" or "bootstrap"
) -> dict:
    """Core replay logic shared by window shuffle and bootstrap.

    Args:
        windows: list of OOS window trade P&L lists
        initial_capital: starting capital
        num_simulations: number of simulations to run
        ruin_threshold: equity level considered ruin
        rng: numpy random generator
        sample_mode: "shuffle" (permute without replacement) or "bootstrap" (with replacement)

    Returns:
        dict with all result fields
    """
    num_windows = len(windows)

    # Flatten original window order to build actual curve
    flat_pnls = []
    for w in windows:
        flat_pnls.extend(w)
    num_trades = len(flat_pnls)

    # Actual equity curve (original window order)
    if num_trades > 0:
        actual_pnls = np.array(flat_pnls, dtype=np.float64)
        actual_curve = (initial_capital + np.cumsum(actual_pnls)).tolist()
    else:
        actual_curve = []

    # Pre-convert windows to numpy arrays
    window_arrays = [np.array(w, dtype=np.float64) for w in windows]

    # Handle empty windows
    if num_windows == 0 or num_trades == 0:
        empty_bands = {"p5": [], "p25": [], "p50": [], "p75": [], "p95": []}
        return {
            "simulations": num_simulations,
            "percentile_bands": empty_bands,
            "actual_curve": actual_curve,
            "final_equity_distribution": [initial_capital] * num_simulations,
            "drawdown_distribution": [0.0] * num_simulations,
            "probability_of_ruin": 0.0,
            "sharpe_ci": [0.0, 0.0],
            "max_dd_ci": [0.0, 0.0],
        }

    # For shuffle mode all sims have the same length (num_trades).
    # For bootstrap mode sims may have different lengths — use lists.
    fixed_length = sample_mode == "shuffle"
    if fixed_length:
        all_paths = np.empty((num_simulations, num_trades))
    final_equities_list = []
    max_drawdowns = np.empty(num_simulations)
    sharpe_values = np.empty(num_simulations)
    ruin_count = 0

    for sim_idx in range(num_simulations):
        # Sample window indices
        if sample_mode == "shuffle":
            indices = rng.permutation(num_windows)
        else:  # bootstrap
            indices = rng.integers(0, num_windows, size=num_windows)

        # Concatenate P&Ls in sampled window order
        sim_pnls = np.concatenate([window_arrays[i] for i in indices])

        # Replay equity curve
        equity_curve = initial_capital + np.cumsum(sim_pnls)

        if fixed_length:
            all_paths[sim_idx, :] = equity_curve

        final_equities_list.append(float(equity_curve[-1]) if len(equity_curve) > 0 else initial_capital)

        # Max drawdown
        max_drawdowns[sim_idx] = _compute_max_drawdown(equity_curve, initial_capital)

        # Sharpe from equity curve
        sharpe_values[sim_idx] = _compute_sharpe_from_equity(equity_curve)

        # Check ruin
        if np.any(equity_curve <= ruin_threshold):
            ruin_count += 1

    final_equities = np.array(final_equities_list)

    # Percentile bands per time step (only for fixed-length modes)
    if fixed_length:
        percentile_bands = {
            "p5": np.percentile(all_paths, 5, axis=0).tolist(),
            "p25": np.percentile(all_paths, 25, axis=0).tolist(),
            "p50": np.percentile(all_paths, 50, axis=0).tolist(),
            "p75": np.percentile(all_paths, 75, axis=0).tolist(),
            "p95": np.percentile(all_paths, 95, axis=0).tolist(),
        }
    else:
        # For variable-length sims, skip per-step bands — provide empty
        percentile_bands = {"p5": [], "p25": [], "p50": [], "p75": [], "p95": []}

    # Confidence intervals
    ci = compute_metric_confidence_intervals(sharpe_values, max_drawdowns)

    return {
        "simulations": num_simulations,
        "percentile_bands": percentile_bands,
        "actual_curve": actual_curve,
        "final_equity_distribution": final_equities.tolist(),
        "drawdown_distribution": max_drawdowns.tolist(),
        "probability_of_ruin": ruin_count / num_simulations,
        "sharpe_ci": ci["sharpe_ci"],
        "max_dd_ci": ci["max_dd_ci"],
    }


# ---------------------------------------------------------------------------
# Window-level Monte Carlo functions
# ---------------------------------------------------------------------------


def run_window_shuffle(
    windows: list[list[float]],
    initial_capital: float = 100000.0,
    num_simulations: int = 10000,
    seed: int | None = None,
    ruin_threshold: float = 0.0,
) -> dict:
    """Monte Carlo by shuffling the order of OOS windows.

    Each window is a list of trade P&Ls from one walk-forward OOS period.
    Windows are permuted (without replacement) so internal trade order within
    each window is preserved, but the sequence of windows is randomised.

    Args:
        windows: list of OOS window trade P&L lists
        initial_capital: starting capital
        num_simulations: number of shuffled simulations
        seed: random seed for reproducibility
        ruin_threshold: equity level considered ruin

    Returns:
        dict with keys: simulations, percentile_bands, actual_curve,
        final_equity_distribution, drawdown_distribution,
        probability_of_ruin, sharpe_ci, max_dd_ci
    """
    rng = np.random.default_rng(seed)
    return _replay_windows(
        windows, initial_capital, num_simulations, ruin_threshold, rng, "shuffle"
    )


def run_bootstrap(
    windows: list[list[float]],
    initial_capital: float = 100000.0,
    num_simulations: int = 10000,
    seed: int | None = None,
    ruin_threshold: float = 0.0,
) -> dict:
    """Monte Carlo by bootstrap-resampling OOS windows with replacement.

    Samples K windows with replacement (K = len(windows)), so some windows
    may appear multiple times and others may be skipped.

    Args:
        windows: list of OOS window trade P&L lists
        initial_capital: starting capital
        num_simulations: number of bootstrap simulations
        seed: random seed for reproducibility
        ruin_threshold: equity level considered ruin

    Returns:
        dict with keys: simulations, percentile_bands, actual_curve,
        final_equity_distribution, drawdown_distribution,
        probability_of_ruin, sharpe_ci, max_dd_ci
    """
    rng = np.random.default_rng(seed)
    return _replay_windows(
        windows, initial_capital, num_simulations, ruin_threshold, rng, "bootstrap"
    )


def compute_metric_confidence_intervals(
    sharpe_values: np.ndarray,
    max_dd_values: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """Compute confidence intervals for Sharpe ratio and max drawdown arrays.

    Args:
        sharpe_values: array of Sharpe ratios from simulations
        max_dd_values: array of max drawdown values from simulations
        confidence: confidence level (default 0.95 → P2.5 / P97.5)

    Returns:
        dict with sharpe_ci: [lower, upper] and max_dd_ci: [lower, upper]
    """
    alpha = (1.0 - confidence) / 2.0 * 100.0  # e.g. 2.5 for 95% CI
    sharpe_arr = np.asarray(sharpe_values, dtype=np.float64)
    dd_arr = np.asarray(max_dd_values, dtype=np.float64)

    return {
        "sharpe_ci": [
            float(np.percentile(sharpe_arr, alpha)),
            float(np.percentile(sharpe_arr, 100.0 - alpha)),
        ],
        "max_dd_ci": [
            float(np.percentile(dd_arr, alpha)),
            float(np.percentile(dd_arr, 100.0 - alpha)),
        ],
    }
