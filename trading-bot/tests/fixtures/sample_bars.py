"""Deterministic sample bar data for engine integration tests.

Generates 100 daily bars for SPY covering 2024-01-02 through 2024-05-31.
The price series is seeded so every test run sees identical data.

Shape of the series:
  - Starts at ~$470
  - Dips to ~$450 around bars 30-40 (provides mean-reversion entry)
  - Recovers to $490+ by the end
  - Realistic volume in the 50-80 M range
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from data.storage.models import DailyBar

# Seed for reproducibility
_SEED = 42


def generate_spy_bars(
    symbol: str = "SPY",
    num_bars: int = 100,
    start_date: date = date(2024, 1, 2),
) -> list[DailyBar]:
    """Return *num_bars* deterministic DailyBar objects for *symbol*.

    Uses a seeded PRNG so results are identical across invocations.
    """
    rng = random.Random(_SEED)
    bars: list[DailyBar] = []

    price = 470.0
    current = start_date

    for i in range(num_bars):
        # Skip weekends
        while current.weekday() >= 5:
            current += timedelta(days=1)

        # Trend component: dip around bars 30-40, then recover
        if 30 <= i <= 40:
            drift = -0.005  # downtrend (dip to ~$450)
        elif 40 < i <= 60:
            drift = 0.002   # recovery phase
        elif i > 60:
            drift = 0.001   # gentle uptrend to ~$490+
        else:
            drift = 0.0005  # mild uptrend at the start

        # Daily return: drift + noise
        daily_return = drift + rng.gauss(0, 0.008)
        price *= 1 + daily_return

        # Intraday range
        intraday_range = price * rng.uniform(0.005, 0.015)
        open_price = price + rng.uniform(-intraday_range / 3, intraday_range / 3)
        high = max(open_price, price) + rng.uniform(0, intraday_range / 2)
        low = min(open_price, price) - rng.uniform(0, intraday_range / 2)
        close = price
        adj_close = close  # no splits in the sample
        vwap = (high + low + close) / 3

        volume = int(rng.uniform(50_000_000, 80_000_000))

        bars.append(
            DailyBar(
                symbol=symbol,
                date=current,
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                adj_close=round(adj_close, 2),
                volume=volume,
                vwap=round(vwap, 2),
            )
        )

        current += timedelta(days=1)

    return bars


# Alias used by tests that import generate_sample_bars
generate_sample_bars = generate_spy_bars
