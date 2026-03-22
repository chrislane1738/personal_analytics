"""Test fixtures: sample DailyBar instances with known quality issues."""
from datetime import date

from data.storage.models import DailyBar

# ---------------------------------------------------------------------------
# Good bar — passes all validators
# ---------------------------------------------------------------------------
GOOD_BAR = DailyBar(
    symbol="AAPL",
    date=date(2024, 1, 2),
    open=185.0,
    high=187.0,
    low=184.0,
    close=186.0,
    adj_close=185.5,
    volume=16_000_000,
    vwap=185.8,
    data_quality_score=1.0,
)

# ---------------------------------------------------------------------------
# Bad bars — each has exactly one known issue
# ---------------------------------------------------------------------------

# close is negative
NEGATIVE_PRICE_BAR = DailyBar(
    symbol="AAPL",
    date=date(2024, 1, 3),
    open=185.0,
    high=187.0,
    low=-5.0,   # low must be > 0; also close is negative below
    close=-5.0,
    adj_close=-5.0,
    volume=10_000_000,
    vwap=-5.0,
    data_quality_score=1.0,
)

# volume = 0 on a trading day
ZERO_VOLUME_BAR = DailyBar(
    symbol="AAPL",
    date=date(2024, 1, 4),
    open=186.0,
    high=188.0,
    low=185.0,
    close=187.0,
    adj_close=187.0,
    volume=0,
    vwap=186.5,
    data_quality_score=1.0,
)

# high < low (inverted)
HIGH_LOW_INVERTED = DailyBar(
    symbol="AAPL",
    date=date(2024, 1, 5),
    open=184.0,
    high=183.0,  # high < low
    low=185.0,
    close=184.0,
    adj_close=184.0,
    volume=12_000_000,
    vwap=184.0,
    data_quality_score=1.0,
)

# 60% daily move relative to previous close
# Assume prev close was 186.0 → close = 186 * 1.60 = 297.6
SPIKE_BAR = DailyBar(
    symbol="AAPL",
    date=date(2024, 1, 8),
    open=297.0,
    high=300.0,
    low=296.0,
    close=297.6,
    adj_close=297.6,
    volume=20_000_000,
    vwap=298.0,
    data_quality_score=1.0,
)

# Previous bar used when testing SPIKE_BAR
SPIKE_PREV_BAR = DailyBar(
    symbol="AAPL",
    date=date(2024, 1, 5),
    open=185.0,
    high=187.0,
    low=184.0,
    close=186.0,
    adj_close=185.5,
    volume=16_000_000,
    vwap=185.8,
    data_quality_score=1.0,
)

# ---------------------------------------------------------------------------
# List with a missing trading day in the middle
# 2024-01-02 (Tue), then jump to 2024-01-04 (Thu) — 2024-01-03 (Wed) missing
# ---------------------------------------------------------------------------
MISSING_GAP_BARS = [
    DailyBar(
        symbol="AAPL",
        date=date(2024, 1, 2),
        open=185.0,
        high=187.0,
        low=184.0,
        close=186.0,
        adj_close=185.5,
        volume=16_000_000,
        vwap=185.8,
        data_quality_score=1.0,
    ),
    # 2024-01-03 (Wednesday) is intentionally absent
    DailyBar(
        symbol="AAPL",
        date=date(2024, 1, 4),
        open=186.0,
        high=188.5,
        low=185.5,
        close=188.0,
        adj_close=188.0,
        volume=14_000_000,
        vwap=187.0,
        data_quality_score=1.0,
    ),
]
