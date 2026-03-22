"""Market regime detection and per-regime performance statistics."""

from __future__ import annotations

import math
from datetime import date
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sma(values: list[float], period: int) -> list[Optional[float]]:
    """Return simple moving averages; prefix is None where insufficient data."""
    result: list[Optional[float]] = []
    for i, _ in enumerate(values):
        if i + 1 < period:
            result.append(None)
        else:
            window = values[i - period + 1 : i + 1]
            result.append(sum(window) / period)
    return result


def _realized_vol(prices: list[float], lookback: int = 20) -> list[Optional[float]]:
    """20-day annualised realised volatility (log-return std * sqrt(252)).

    Returns None where insufficient data.
    """
    result: list[Optional[float]] = [None] * len(prices)
    for i in range(lookback, len(prices)):
        log_rets = [
            math.log(prices[j] / prices[j - 1])
            for j in range(i - lookback + 1, i + 1)
        ]
        mean = sum(log_rets) / len(log_rets)
        variance = sum((r - mean) ** 2 for r in log_rets) / len(log_rets)
        result[i] = math.sqrt(variance) * math.sqrt(252)
    return result


# ---------------------------------------------------------------------------
# detect_regimes
# ---------------------------------------------------------------------------

def detect_regimes(
    prices: list[tuple[date, float]],
    vix_prices: list[tuple[date, float]] = None,
) -> list[tuple[date, MarketRegime]]:
    """Classify each date into a market regime.

    Uses SMA-50 and SMA-200 of the benchmark (prices):

    - Bull     : SMA-50 > SMA-200 and SMA-50 slope positive
    - Bear     : SMA-50 < SMA-200 and SMA-50 slope negative
    - Sideways : SMA-50 ≈ SMA-200 (within 2%) or ambiguous slope
    - High Vol : VIX > 25, or annualised realised vol > 20% (overrides others)

    Args:
        prices:     [(date, close_price), ...] for benchmark (e.g., SPY).
        vix_prices: [(date, close_price), ...] for VIX, optional.

    Returns:
        List of (date, MarketRegime) pairs aligned 1-to-1 with *prices*.
    """
    if not prices:
        return []

    dates = [p[0] for p in prices]
    closes = [p[1] for p in prices]

    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    rvol = _realized_vol(closes, 20)

    # Build VIX lookup by date
    vix_map: dict[date, float] = {}
    if vix_prices:
        for d, v in vix_prices:
            vix_map[d] = v

    # Slope window for SMA-50: compare current SMA-50 to its value 10 bars ago
    SLOPE_WINDOW = 10
    SIDEWAYS_BAND = 0.02   # within 2% → sideways
    VIX_THRESHOLD = 25.0
    RVOL_THRESHOLD = 0.20  # 20% annualised

    regimes: list[tuple[date, MarketRegime]] = []

    for i, d in enumerate(dates):
        s50 = sma50[i]
        s200 = sma200[i]
        rv = rvol[i]
        vix = vix_map.get(d)

        # High-vol override check
        high_vol = False
        if vix is not None and vix > VIX_THRESHOLD:
            high_vol = True
        if rv is not None and rv > RVOL_THRESHOLD:
            high_vol = True

        if high_vol:
            regimes.append((d, MarketRegime.HIGH_VOL))
            continue

        # Need at least SMA-50 to classify; fall back to SIDEWAYS early on
        if s50 is None:
            regimes.append((d, MarketRegime.SIDEWAYS))
            continue

        # SMA-50 slope: compare to SLOPE_WINDOW bars ago
        prev_s50_idx = i - SLOPE_WINDOW
        prev_s50 = sma50[prev_s50_idx] if prev_s50_idx >= 0 else None
        if prev_s50 is not None and s50 != 0:
            slope = (s50 - prev_s50) / s50
        else:
            slope = 0.0

        # Classify
        if s200 is None:
            # Can't compute golden/death cross yet; use slope only
            if slope > 0.001:
                regimes.append((d, MarketRegime.BULL))
            elif slope < -0.001:
                regimes.append((d, MarketRegime.BEAR))
            else:
                regimes.append((d, MarketRegime.SIDEWAYS))
            continue

        gap = (s50 - s200) / s200 if s200 != 0 else 0.0

        if abs(gap) <= SIDEWAYS_BAND and abs(slope) <= 0.001:
            regimes.append((d, MarketRegime.SIDEWAYS))
        elif gap > 0 and slope > 0:
            regimes.append((d, MarketRegime.BULL))
        elif gap < 0 and slope < 0:
            regimes.append((d, MarketRegime.BEAR))
        else:
            # Mixed signals → sideways
            regimes.append((d, MarketRegime.SIDEWAYS))

    return regimes


# ---------------------------------------------------------------------------
# compute_regime_stats
# ---------------------------------------------------------------------------

def compute_regime_stats(
    trades: list,
    regimes: list[tuple[date, MarketRegime]],
) -> dict:
    """Compute per-regime performance statistics.

    *trades* may be either :class:`~analytics.trade_log.CompletedTrade`
    objects (with ``.entry_date`` and ``.pnl`` attributes) or plain dicts
    with ``"entry_date"`` and ``"pnl"`` keys.

    Returns a dict keyed by regime name string::

        {
            "bull":  {"trades": 142, "win_rate": 0.683, "avg_pnl": 150.0, ...},
            "bear":  {"trades": 38,  "win_rate": 0.421, "avg_pnl": -80.0, ...},
            ...
        }
    """
    # Build a lookup: entry_date → regime
    regime_map: dict[date, MarketRegime] = {d: r for d, r in regimes}

    # Group trades by regime name
    groups: dict[str, list] = {r.value: [] for r in MarketRegime}

    for trade in trades:
        if isinstance(trade, dict):
            entry_date = trade["entry_date"]
            pnl = trade["pnl"]
        else:
            entry_date = trade.entry_date
            pnl = trade.pnl

        regime = regime_map.get(entry_date)
        if regime is not None:
            groups[regime.value].append(pnl)

    result: dict[str, dict] = {}
    for regime_name, pnls in groups.items():
        if not pnls:
            result[regime_name] = {"trades": 0}
            continue
        wins = [p for p in pnls if p > 0]
        result[regime_name] = {
            "trades": len(pnls),
            "win_rate": len(wins) / len(pnls),
            "avg_pnl": sum(pnls) / len(pnls),
            "total_pnl": sum(pnls),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
        }

    return result
