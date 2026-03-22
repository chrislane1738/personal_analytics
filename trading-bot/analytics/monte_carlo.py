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
