"""Options-specific analytics: premium tracking, assignment rates, DTE bucketing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OptionsAnalyticsResult:
    """Aggregated analytics for a set of options trades."""

    total_premium_collected: float      # gross premium received from STO trades
    total_premium_paid: float           # gross premium paid for BTO trades (non-closing)
    net_premium: float                  # collected - paid
    assignment_count: int               # number of short options that were assigned
    total_short_options: int            # total STO trade count
    assignment_rate: float              # assignment_count / total_short_options
    win_rate_by_dte: dict[str, float]   # {"0-7": 0.8, "7-30": 0.65, ...}
    avg_pnl_by_dte: dict[str, float]    # average P&L per DTE bucket
    greeks_timeline: list[dict]         # [{"date": date, "delta": float, ...}]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_options_analytics(
    trades: list,
    greeks_snapshots: list[dict] | None = None,
) -> OptionsAnalyticsResult:
    """Compute options-specific analytics from a list of completed trades.

    Parameters
    ----------
    trades:
        List of trade dicts *or* ``CompletedTrade`` objects.  Each entry is
        expected to have:

        - ``option_type`` – ``"call"`` or ``"put"``; equity trades have
          ``None`` and are skipped.
        - ``direction`` – ``"sell_put"``, ``"sell_call"``, ``"buy_to_close"``,
          ``"sell_to_close"``, etc.
        - ``entry_price`` – premium per share (options are per 100 shares).
        - ``exit_price`` – premium at close (0 if expired worthless).
        - ``pnl`` – realised P&L for the trade.
        - ``quantity`` – number of contracts.
        - ``entry_date`` – :class:`datetime.date` of trade entry.
        - ``exit_date`` – :class:`datetime.date` of trade exit.
        - ``expiration`` – option expiration :class:`datetime.date`.
        - ``exit_reason`` – string; contains ``"assignment"`` when assigned.

    greeks_snapshots:
        Optional list of daily portfolio-level Greeks snapshots:
        ``[{"date": date, "delta": float, "gamma": float, "theta": float,
        "vega": float}, ...]``

    Returns
    -------
    OptionsAnalyticsResult
    """
    options_trades = [t for t in trades if _get_field(t, "option_type") is not None]

    if not options_trades:
        return OptionsAnalyticsResult(
            total_premium_collected=0.0,
            total_premium_paid=0.0,
            net_premium=0.0,
            assignment_count=0,
            total_short_options=0,
            assignment_rate=0.0,
            win_rate_by_dte={},
            avg_pnl_by_dte={},
            greeks_timeline=greeks_snapshots or [],
        )

    # ------------------------------------------------------------------
    # Premium tracking
    # ------------------------------------------------------------------
    premium_collected = 0.0
    premium_paid = 0.0

    for t in options_trades:
        direction = _get_field(t, "direction", "")
        entry_price = _get_field(t, "entry_price", 0.0) or 0.0
        quantity = _get_field(t, "quantity", 0) or 0

        if direction in ("sell_put", "sell_call"):
            # Selling (STO) — we receive the premium.
            premium_collected += entry_price * quantity * 100
        elif direction in ("buy_to_close", "sell_to_close"):
            # Closing trades — not new premium; handled via pnl.
            pass
        else:
            # Buying to open (BTO) or other opening buys.
            premium_paid += entry_price * quantity * 100

    # ------------------------------------------------------------------
    # Assignment tracking
    # ------------------------------------------------------------------
    short_options = [
        t for t in options_trades
        if _get_field(t, "direction", "") in ("sell_put", "sell_call")
    ]
    assignments = [
        t for t in short_options
        if "assignment" in str(_get_field(t, "exit_reason", "") or "").lower()
    ]

    # ------------------------------------------------------------------
    # DTE bucketing: bucket by DTE at entry (expiration - entry_date)
    # ------------------------------------------------------------------
    dte_buckets: dict[str, list[float]] = {
        "0-7": [],
        "7-30": [],
        "30-60": [],
        "60+": [],
    }

    for t in options_trades:
        entry_date = _get_field(t, "entry_date")
        expiration = _get_field(t, "expiration")
        if entry_date is None or expiration is None:
            continue

        dte = (expiration - entry_date).days
        pnl = _get_field(t, "pnl", 0.0) or 0.0

        if dte <= 7:
            dte_buckets["0-7"].append(pnl)
        elif dte <= 30:
            dte_buckets["7-30"].append(pnl)
        elif dte <= 60:
            dte_buckets["30-60"].append(pnl)
        else:
            dte_buckets["60+"].append(pnl)

    win_rate_by_dte: dict[str, float] = {}
    avg_pnl_by_dte: dict[str, float] = {}

    for bucket, pnls in dte_buckets.items():
        if pnls:
            win_rate_by_dte[bucket] = sum(1 for p in pnls if p > 0) / len(pnls)
            avg_pnl_by_dte[bucket] = sum(pnls) / len(pnls)

    total_short = len(short_options)

    return OptionsAnalyticsResult(
        total_premium_collected=premium_collected,
        total_premium_paid=premium_paid,
        net_premium=premium_collected - premium_paid,
        assignment_count=len(assignments),
        total_short_options=total_short,
        assignment_rate=len(assignments) / total_short if total_short > 0 else 0.0,
        win_rate_by_dte=win_rate_by_dte,
        avg_pnl_by_dte=avg_pnl_by_dte,
        greeks_timeline=greeks_snapshots or [],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_field(obj, field: str, default=None):
    """Retrieve *field* from either a dict or an object attribute."""
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)
