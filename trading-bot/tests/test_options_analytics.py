"""Tests for analytics/options_analytics.py."""

from __future__ import annotations

from datetime import date

import pytest

from analytics.options_analytics import OptionsAnalyticsResult, compute_options_analytics
from analytics.trade_log import CompletedTrade


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_trade_dict(
    *,
    option_type: str | None = "put",
    direction: str = "sell_put",
    entry_price: float = 2.50,
    exit_price: float = 0.0,
    pnl: float = 250.0,
    quantity: int = 1,
    entry_date: date = date(2024, 1, 1),
    exit_date: date = date(2024, 1, 20),
    expiration: date = date(2024, 1, 19),
    exit_reason: str = "expired_worthless",
) -> dict:
    """Convenience factory for trade dicts."""
    return {
        "option_type": option_type,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "quantity": quantity,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "expiration": expiration,
        "exit_reason": exit_reason,
    }


def _make_completed_trade(
    *,
    option_type: str | None = "put",
    direction: str = "sell_put",
    entry_price: float = 2.50,
    exit_price: float = 0.0,
    pnl: float = 250.0,
    quantity: int = 1,
    entry_date: date = date(2024, 1, 1),
    exit_date: date = date(2024, 1, 20),
    expiration: date = date(2024, 1, 19),
    exit_reason: str = "expired_worthless",
) -> CompletedTrade:
    """Factory for CompletedTrade objects."""
    holding_days = (exit_date - entry_date).days
    cost_basis = entry_price * quantity
    return CompletedTrade(
        symbol="SPY",
        direction=direction,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        pnl=pnl,
        pnl_pct=pnl / cost_basis if cost_basis else 0.0,
        entry_reason="signal",
        exit_reason=exit_reason,
        holding_days=holding_days,
        option_type=option_type,
        strike=100.0,
        expiration=expiration,
    )


# ---------------------------------------------------------------------------
# 1. Premium tracking: 3 STO sells at $2.50, $3.00, $1.80 (1 contract each)
#    Total collected = (2.50 + 3.00 + 1.80) * 100 = $730
# ---------------------------------------------------------------------------

def test_premium_collected_three_sto_trades():
    trades = [
        _make_trade_dict(entry_price=2.50, direction="sell_put"),
        _make_trade_dict(entry_price=3.00, direction="sell_call"),
        _make_trade_dict(entry_price=1.80, direction="sell_put"),
    ]
    result = compute_options_analytics(trades)

    assert result.total_premium_collected == pytest.approx(730.0)
    assert result.total_premium_paid == pytest.approx(0.0)
    assert result.net_premium == pytest.approx(730.0)


def test_premium_paid_bto_trades():
    """BTO (buy-to-open) trades should increment total_premium_paid."""
    trades = [
        _make_trade_dict(direction="buy_put", entry_price=1.50, option_type="put"),
        _make_trade_dict(direction="buy_call", entry_price=2.00, option_type="call"),
    ]
    result = compute_options_analytics(trades)

    assert result.total_premium_paid == pytest.approx(350.0)   # (1.50 + 2.00) * 100
    assert result.total_premium_collected == pytest.approx(0.0)
    assert result.net_premium == pytest.approx(-350.0)


# ---------------------------------------------------------------------------
# 2. Assignment rate: 3 short puts, 1 assigned → rate = 1/3 ≈ 33.3%
# ---------------------------------------------------------------------------

def test_assignment_rate_one_in_three():
    trades = [
        _make_trade_dict(exit_reason="expired_worthless"),
        _make_trade_dict(exit_reason="assignment"),
        _make_trade_dict(exit_reason="buy_to_close"),
    ]
    result = compute_options_analytics(trades)

    assert result.total_short_options == 3
    assert result.assignment_count == 1
    assert result.assignment_rate == pytest.approx(1 / 3)


def test_assignment_rate_no_assignments():
    trades = [
        _make_trade_dict(exit_reason="expired_worthless"),
        _make_trade_dict(exit_reason="buy_to_close"),
    ]
    result = compute_options_analytics(trades)

    assert result.assignment_count == 0
    assert result.assignment_rate == pytest.approx(0.0)


def test_assignment_rate_case_insensitive():
    """exit_reason matching should be case-insensitive."""
    trades = [_make_trade_dict(exit_reason="Assignment via broker")]
    result = compute_options_analytics(trades)

    assert result.assignment_count == 1


# ---------------------------------------------------------------------------
# 3. DTE bucketing: trades at 5, 20, 45, 90 DTE map to correct buckets
# ---------------------------------------------------------------------------

BASE_ENTRY = date(2024, 1, 1)


def _trade_with_dte(dte: int, pnl: float = 100.0, direction: str = "sell_put") -> dict:
    expiration = date.fromordinal(BASE_ENTRY.toordinal() + dte)
    return _make_trade_dict(
        entry_date=BASE_ENTRY,
        expiration=expiration,
        pnl=pnl,
        direction=direction,
    )


def test_dte_bucket_0_to_7():
    result = compute_options_analytics([_trade_with_dte(5)])
    assert "0-7" in result.win_rate_by_dte
    assert "7-30" not in result.win_rate_by_dte


def test_dte_bucket_7_to_30():
    result = compute_options_analytics([_trade_with_dte(20)])
    assert "7-30" in result.win_rate_by_dte
    assert "0-7" not in result.win_rate_by_dte


def test_dte_bucket_30_to_60():
    result = compute_options_analytics([_trade_with_dte(45)])
    assert "30-60" in result.win_rate_by_dte
    assert "60+" not in result.win_rate_by_dte


def test_dte_bucket_60_plus():
    result = compute_options_analytics([_trade_with_dte(90)])
    assert "60+" in result.win_rate_by_dte
    assert "30-60" not in result.win_rate_by_dte


def test_dte_boundary_exactly_7():
    """A trade with exactly 7 DTE belongs in '0-7'."""
    result = compute_options_analytics([_trade_with_dte(7)])
    assert "0-7" in result.win_rate_by_dte


def test_dte_boundary_exactly_30():
    """A trade with exactly 30 DTE belongs in '7-30'."""
    result = compute_options_analytics([_trade_with_dte(30)])
    assert "7-30" in result.win_rate_by_dte


def test_dte_boundary_exactly_60():
    """A trade with exactly 60 DTE belongs in '30-60'."""
    result = compute_options_analytics([_trade_with_dte(60)])
    assert "30-60" in result.win_rate_by_dte


def test_dte_all_buckets_populated():
    """Trades spanning all buckets should produce four populated bucket keys."""
    trades = [
        _trade_with_dte(5),
        _trade_with_dte(20),
        _trade_with_dte(45),
        _trade_with_dte(90),
    ]
    result = compute_options_analytics(trades)
    assert set(result.win_rate_by_dte.keys()) == {"0-7", "7-30", "30-60", "60+"}


# ---------------------------------------------------------------------------
# 4. Win rate by DTE: 2 trades in '7-30', 1 win ($50) + 1 loss (-$30) → 50%
# ---------------------------------------------------------------------------

def test_win_rate_50_percent_in_7_to_30():
    trades = [
        _trade_with_dte(20, pnl=50.0),
        _trade_with_dte(25, pnl=-30.0),
    ]
    result = compute_options_analytics(trades)

    assert result.win_rate_by_dte["7-30"] == pytest.approx(0.5)


def test_avg_pnl_by_dte_correct():
    trades = [
        _trade_with_dte(20, pnl=50.0),
        _trade_with_dte(25, pnl=-30.0),
    ]
    result = compute_options_analytics(trades)

    assert result.avg_pnl_by_dte["7-30"] == pytest.approx(10.0)  # (50 - 30) / 2


def test_win_rate_100_percent():
    trades = [_trade_with_dte(20, pnl=100.0), _trade_with_dte(25, pnl=50.0)]
    result = compute_options_analytics(trades)
    assert result.win_rate_by_dte["7-30"] == pytest.approx(1.0)


def test_win_rate_0_percent():
    trades = [_trade_with_dte(20, pnl=-20.0), _trade_with_dte(25, pnl=-10.0)]
    result = compute_options_analytics(trades)
    assert result.win_rate_by_dte["7-30"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. No options trades → all zeros / empty collections
# ---------------------------------------------------------------------------

def test_no_options_trades_returns_zeros():
    equity_trades = [
        _make_trade_dict(option_type=None, direction="long"),
        _make_trade_dict(option_type=None, direction="short"),
    ]
    result = compute_options_analytics(equity_trades)

    assert result.total_premium_collected == 0.0
    assert result.total_premium_paid == 0.0
    assert result.net_premium == 0.0
    assert result.assignment_count == 0
    assert result.total_short_options == 0
    assert result.assignment_rate == 0.0
    assert result.win_rate_by_dte == {}
    assert result.avg_pnl_by_dte == {}


def test_empty_trades_list_returns_zeros():
    result = compute_options_analytics([])

    assert result.total_premium_collected == 0.0
    assert result.total_short_options == 0
    assert result.assignment_rate == 0.0
    assert result.win_rate_by_dte == {}


# ---------------------------------------------------------------------------
# 6. Mixed equity + options: equity trades must be filtered out
# ---------------------------------------------------------------------------

def test_mixed_equity_and_options_filtered_correctly():
    trades = [
        # Equity trades — should be ignored
        _make_trade_dict(option_type=None, direction="long", entry_price=50.0, pnl=200.0),
        _make_trade_dict(option_type=None, direction="short", entry_price=40.0, pnl=-100.0),
        # Options trades — should be counted
        _make_trade_dict(option_type="put", direction="sell_put", entry_price=2.50, quantity=1),
        _make_trade_dict(option_type="call", direction="sell_call", entry_price=1.80, quantity=1),
    ]
    result = compute_options_analytics(trades)

    # Only the two STO options trades should count toward premium
    assert result.total_premium_collected == pytest.approx((2.50 + 1.80) * 100)
    assert result.total_short_options == 2


def test_mixed_equity_does_not_inflate_assignment_count():
    trades = [
        _make_trade_dict(option_type=None, direction="long", exit_reason="assignment"),
        _make_trade_dict(option_type="put", direction="sell_put", exit_reason="expired_worthless"),
    ]
    result = compute_options_analytics(trades)

    # The equity "assignment" must not be counted
    assert result.assignment_count == 0


# ---------------------------------------------------------------------------
# 7. Greeks timeline passes through unchanged
# ---------------------------------------------------------------------------

def test_greeks_timeline_passthrough():
    snapshots = [
        {"date": date(2024, 1, 1), "delta": 0.5, "gamma": 0.02, "theta": -0.05, "vega": 0.10},
        {"date": date(2024, 1, 2), "delta": 0.45, "gamma": 0.03, "theta": -0.06, "vega": 0.11},
    ]
    trades = [_make_trade_dict()]
    result = compute_options_analytics(trades, greeks_snapshots=snapshots)

    assert result.greeks_timeline == snapshots
    assert result.greeks_timeline[0]["delta"] == 0.5
    assert result.greeks_timeline[1]["theta"] == pytest.approx(-0.06)


def test_greeks_timeline_defaults_to_empty():
    result = compute_options_analytics([_make_trade_dict()])
    assert result.greeks_timeline == []


def test_greeks_timeline_preserved_when_no_option_trades():
    """Greeks snapshots should be included even if trade list has no options trades."""
    snapshots = [{"date": date(2024, 1, 1), "delta": 0.3}]
    result = compute_options_analytics([], greeks_snapshots=snapshots)
    assert result.greeks_timeline == snapshots


# ---------------------------------------------------------------------------
# 8. CompletedTrade objects (not dicts) work correctly
# ---------------------------------------------------------------------------

def test_works_with_completed_trade_objects():
    trades = [
        _make_completed_trade(entry_price=2.50, direction="sell_put"),
        _make_completed_trade(entry_price=3.00, direction="sell_call"),
        _make_completed_trade(entry_price=1.80, direction="sell_put"),
    ]
    result = compute_options_analytics(trades)

    assert result.total_premium_collected == pytest.approx(730.0)
    assert result.total_short_options == 3


def test_assignment_with_completed_trade_objects():
    trades = [
        _make_completed_trade(exit_reason="expired_worthless"),
        _make_completed_trade(exit_reason="assignment"),
    ]
    result = compute_options_analytics(trades)

    assert result.assignment_count == 1
    assert result.assignment_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 9. Result is an OptionsAnalyticsResult dataclass
# ---------------------------------------------------------------------------

def test_result_type():
    result = compute_options_analytics([_make_trade_dict()])
    assert isinstance(result, OptionsAnalyticsResult)


# ---------------------------------------------------------------------------
# 10. Multi-contract trades scale premium by quantity
# ---------------------------------------------------------------------------

def test_multi_contract_premium_scaling():
    trade = _make_trade_dict(entry_price=2.00, quantity=5, direction="sell_put")
    result = compute_options_analytics([trade])

    # 2.00 * 5 contracts * 100 shares/contract = $1000
    assert result.total_premium_collected == pytest.approx(1000.0)
