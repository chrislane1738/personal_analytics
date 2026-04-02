"""VWAPReversionStrategy — fades deviations from a short-term SMA.

Uses a Simple Moving Average as a VWAP proxy on daily bars.  ATR is used to
normalise the deviation so the threshold is volatility-adjusted.

Long  when close drops > deviation_threshold ATRs below SMA (oversold).
Short when close rises > deviation_threshold ATRs above SMA (overbought).
"""

from __future__ import annotations

from core.events import BarEvent, SignalEvent
from indicators.technical import ATR, SMA
from strategy.base import Strategy
from topstep.state_manager import StateManager


class VWAPReversionStrategy(Strategy):
    """Mean-reversion strategy that fades SMA deviations scaled by ATR.

    Parameters
    ----------
    sma_period : int
        Look-back for the SMA (VWAP proxy).
    atr_period : int
        Look-back for the ATR (volatility normaliser).
    deviation_threshold : float
        Number of ATRs away from SMA required to generate a signal.
    state_manager : StateManager | None
        Optional Topstep state manager that scales signal strength via a
        position multiplier derived from cumulative PnL.
    """

    def __init__(
        self,
        sma_period: int = 5,
        atr_period: int = 14,
        deviation_threshold: float = 1.0,
        state_manager: StateManager | None = None,
    ) -> None:
        super().__init__()
        self._sma = SMA(period=sma_period)
        self._atr = ATR(period=atr_period)
        self._deviation_threshold = deviation_threshold
        self._state_manager = state_manager
        self._bar_count = 0
        self._warm_up = max(sma_period, atr_period)

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def warm_up_period(self) -> int:
        return self._warm_up

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        # 1. Update indicators
        self._sma.update(bar.close)
        self._atr.update(bar.high, bar.low, bar.close)

        # 2. Increment bar count; skip during warm-up
        self._bar_count += 1
        if self._bar_count <= self._warm_up:
            return []

        # 3. Fetch indicator values; skip if not yet ready
        sma_val = self._sma.value
        atr_val = self._atr.value
        if sma_val is None or atr_val is None or atr_val <= 0.0:
            return []

        # 4. Compute normalised deviation
        deviation = (bar.close - sma_val) / atr_val

        # 5. Near fair value — no trade
        if abs(deviation) < self._deviation_threshold:
            return []

        # 6. Determine direction
        direction = "long" if deviation < -self._deviation_threshold else "short"

        # 7. Position multiplier from state manager (uses portfolio equity as proxy for PnL)
        multiplier = 1.0
        if self._state_manager is not None:
            cumulative_pnl = portfolio.get_equity() - 50_000.0  # relative to baseline
            multiplier = self._state_manager.get_position_multiplier(cumulative_pnl)

        # 8. Strength: |deviation| / 3, scaled by multiplier, capped at 1.0
        raw_strength = (abs(deviation) / 3.0) * multiplier
        strength = min(raw_strength, 1.0)

        signal = SignalEvent(
            symbol=bar.symbol,
            direction=direction,
            reason=f"VWAP reversion: deviation={deviation:.2f} ATRs",
            strength=strength,
        )
        return [signal]

    # ------------------------------------------------------------------
    # Optimisation interface
    # ------------------------------------------------------------------

    @classmethod
    def get_parameter_space(cls) -> dict:
        return {
            "sma_period": [3, 5, 10],
            "atr_period": [10, 14, 20],
            "deviation_threshold": [0.8, 1.0, 1.5, 2.0],
        }

    @classmethod
    def from_params(cls, params: dict) -> "VWAPReversionStrategy":
        return cls(**params)
