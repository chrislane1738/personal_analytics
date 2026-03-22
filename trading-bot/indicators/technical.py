"""
Stateful technical indicator implementations.

All indicators inherit from Indicator (base.py) and maintain internal state
that is updated one price at a time via update(price).
"""
from __future__ import annotations

import math
from collections import deque

from indicators.base import Indicator


# ---------------------------------------------------------------------------
# SMA — Simple Moving Average
# ---------------------------------------------------------------------------

class SMA(Indicator):
    """Simple Moving Average over a rolling window of `period` prices."""

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self._period = period
        self._window: deque[float] = deque(maxlen=period)
        self._sum: float = 0.0

    # ------------------------------------------------------------------
    def update(self, price: float) -> None:
        if len(self._window) == self._period:
            # The oldest value is about to be evicted
            self._sum -= self._window[0]
        self._window.append(price)
        self._sum += price

    @property
    def value(self) -> float | None:
        if len(self._window) < self._period:
            return None
        return self._sum / self._period

    @property
    def warm_up_period(self) -> int:
        return self._period


# ---------------------------------------------------------------------------
# EMA — Exponential Moving Average
# ---------------------------------------------------------------------------

class EMA(Indicator):
    """
    Exponential Moving Average.

    Seed: first EMA = SMA of first `period` values.
    Subsequent: EMA = price * mult + prev_ema * (1 - mult)
    where mult = 2 / (period + 1).
    """

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self._period = period
        self._mult: float = 2.0 / (period + 1)
        self._sma_seed = SMA(period)   # used only during warm-up
        self._ema: float | None = None
        self._count: int = 0

    # ------------------------------------------------------------------
    def update(self, price: float) -> None:
        self._count += 1
        if self._ema is None:
            # Still seeding via SMA
            self._sma_seed.update(price)
            if self._sma_seed.value is not None:
                self._ema = self._sma_seed.value
        else:
            self._ema = price * self._mult + self._ema * (1 - self._mult)

    @property
    def value(self) -> float | None:
        return self._ema

    @property
    def warm_up_period(self) -> int:
        return self._period


# ---------------------------------------------------------------------------
# RSI — Relative Strength Index (Wilder's method)
# ---------------------------------------------------------------------------

class RSI(Indicator):
    """
    Relative Strength Index using Wilder's smoothing.

    warm_up_period = period + 1  (need period+1 prices → period changes)
    """

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self._period = period
        self._prices_seen: int = 0
        self._prev_price: float | None = None

        # Accumulators for the seed (first `period` changes)
        self._seed_gains: list[float] = []
        self._seed_losses: list[float] = []

        # Wilder smoothed averages (set once seed is complete)
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None

    # ------------------------------------------------------------------
    def update(self, price: float) -> None:
        self._prices_seen += 1

        if self._prev_price is None:
            # First price: no change yet
            self._prev_price = price
            return

        change = price - self._prev_price
        self._prev_price = price

        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if self._avg_gain is None:
            # Collect seed changes (we need `period` changes = prices_seen >= period+1)
            self._seed_gains.append(gain)
            self._seed_losses.append(loss)

            if len(self._seed_gains) == self._period:
                # Initialise Wilder averages as simple averages of first period changes
                self._avg_gain = sum(self._seed_gains) / self._period
                self._avg_loss = sum(self._seed_losses) / self._period
        else:
            # Wilder's smoothing
            self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / self._period
            self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / self._period

    @property
    def value(self) -> float | None:
        if self._avg_gain is None:
            return None
        if self._avg_loss == 0.0:
            return 100.0 if self._avg_gain > 0.0 else 50.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @property
    def warm_up_period(self) -> int:
        return self._period + 1


# ---------------------------------------------------------------------------
# MACD — Moving Average Convergence Divergence
# ---------------------------------------------------------------------------

class MACD(Indicator):
    """
    MACD indicator.

    MACD line    = EMA(fast) - EMA(slow)
    Signal line  = EMA(signal) of MACD line values
    Histogram    = MACD line - Signal line

    warm_up_period = slow + signal - 1
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast >= slow:
            raise ValueError("fast period must be less than slow period")
        self._fast_ema = EMA(fast)
        self._slow_ema = EMA(slow)
        self._signal_ema = EMA(signal)
        self._slow = slow
        self._signal_period = signal
        self._macd_line: float | None = None
        self._signal_line_val: float | None = None

    # ------------------------------------------------------------------
    def update(self, price: float) -> None:
        self._fast_ema.update(price)
        self._slow_ema.update(price)

        if self._slow_ema.value is not None and self._fast_ema.value is not None:
            macd_val = self._fast_ema.value - self._slow_ema.value
            self._signal_ema.update(macd_val)
            self._macd_line = macd_val
            if self._signal_ema.value is not None:
                self._signal_line_val = self._signal_ema.value
        # else: slow EMA not ready yet — signal EMA receives no update

    @property
    def value(self) -> float | None:
        """MACD line value (ready only when signal line is also ready)."""
        return self._macd_line if self._signal_line_val is not None else None

    @property
    def signal_line(self) -> float | None:
        return self._signal_line_val

    @property
    def histogram(self) -> float | None:
        if self._macd_line is None or self._signal_line_val is None:
            return None
        return self._macd_line - self._signal_line_val

    @property
    def warm_up_period(self) -> int:
        return self._slow + self._signal_period - 1


# ---------------------------------------------------------------------------
# BollingerBands
# ---------------------------------------------------------------------------

class BollingerBands(Indicator):
    """
    Bollinger Bands.

    Middle = SMA(period)
    Upper  = Middle + std_dev * population_std(window)
    Lower  = Middle - std_dev * population_std(window)
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self._period = period
        self._std_dev = std_dev
        self._window: deque[float] = deque(maxlen=period)

    # ------------------------------------------------------------------
    def update(self, price: float) -> None:
        self._window.append(price)

    def _stats(self) -> tuple[float, float] | None:
        """Return (mean, std) or None if not enough data."""
        if len(self._window) < self._period:
            return None
        n = self._period
        mean = sum(self._window) / n
        variance = sum((x - mean) ** 2 for x in self._window) / n
        return mean, math.sqrt(variance)

    @property
    def value(self) -> float | None:
        """Middle band (same as `middle`)."""
        stats = self._stats()
        return stats[0] if stats is not None else None

    @property
    def middle(self) -> float | None:
        return self.value

    @property
    def upper(self) -> float | None:
        stats = self._stats()
        if stats is None:
            return None
        mean, std = stats
        return mean + self._std_dev * std

    @property
    def lower(self) -> float | None:
        stats = self._stats()
        if stats is None:
            return None
        mean, std = stats
        return mean - self._std_dev * std

    @property
    def warm_up_period(self) -> int:
        return self._period
