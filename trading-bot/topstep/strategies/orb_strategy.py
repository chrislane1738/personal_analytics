"""ORBStrategy — Opening Range Breakout strategy.

Uses daily bar body direction as a proxy for opening range breakout.
A bullish signal is generated when close > open by a threshold scaled by ATR.
A bearish signal is generated when close < open by the same threshold.
"""

from __future__ import annotations

from core.events import BarEvent, SignalEvent
from indicators.technical import ATR, SMA
from strategy.base import Strategy


class ORBStrategy(Strategy):
    """Opening Range Breakout strategy based on daily bar body vs. ATR.

    Parameters
    ----------
    atr_period:
        Lookback period for the ATR indicator used to normalise bar body size.
    sma_period:
        Lookback period for the SMA indicator (used for context; not directly
        gating signals but consumed by update so the indicator stays warm).
    body_threshold:
        Minimum ratio of |close - open| to ATR required before a signal is
        emitted.  Bars whose body is smaller than this fraction of the ATR are
        treated as doji / indecision and ignored.
    state_manager:
        Optional :class:`topstep.state_manager.StateManager`.  When provided,
        the position multiplier is applied to the raw signal strength.
    """

    def __init__(
        self,
        atr_period: int = 14,
        sma_period: int = 5,
        body_threshold: float = 0.3,
        state_manager=None,
    ) -> None:
        super().__init__()
        self._atr = ATR(period=atr_period)
        self._sma = SMA(period=sma_period)
        self._body_threshold = body_threshold
        self._state_manager = state_manager
        self._bar_count: int = 0
        self._warm_up = max(atr_period, sma_period)

    # ------------------------------------------------------------------
    # Strategy ABC
    # ------------------------------------------------------------------

    def warm_up_period(self) -> int:
        return self._warm_up

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        # 1. Update indicators
        self._atr.update(bar.high, bar.low, bar.close)
        self._sma.update(bar.close)

        # 2. Still in warm-up phase
        self._bar_count += 1
        if self._bar_count <= self._warm_up:
            return []

        # 3. Need a valid ATR
        atr_value = self._atr.value
        if atr_value is None or atr_value <= 0:
            return []

        # 4. Compute body and ratio
        body = bar.close - bar.open
        body_ratio = abs(body) / atr_value

        # 5. Filter doji / indecision bars
        if body_ratio < self._body_threshold:
            return []

        # 6. Determine direction
        direction = "long" if body > 0 else "short"

        # 7. Compute strength, capped at 1.0
        raw_strength = min(body_ratio / (self._body_threshold * 3.0), 1.0)

        # 8. Apply state manager multiplier if available
        if self._state_manager is not None:
            equity = portfolio.get_equity()
            pnl = equity - 50_000.0
            multiplier = self._state_manager.get_position_multiplier(pnl)
            strength = min(raw_strength * multiplier, 1.0)
        else:
            strength = raw_strength

        signal = SignalEvent(
            symbol=bar.symbol,
            direction=direction,
            reason="orb_body_atr",
            strength=strength,
        )
        return [signal]

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_parameter_space(cls) -> dict:
        return {
            "atr_period": [10, 14, 20],
            "sma_period": [3, 5, 10],
            "body_threshold": [0.2, 0.3, 0.5],
        }

    @classmethod
    def from_params(cls, params: dict) -> "ORBStrategy":
        return cls(**params)
