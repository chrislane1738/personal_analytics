"""Options Wheel Strategy.

Implements the classic "Wheel" options strategy:
  1. Sell cash-secured puts (CSP) when the underlying is oversold (RSI < threshold).
  2. If assigned (portfolio holds the stock), sell covered calls (CC) until called away.
"""

from __future__ import annotations

from datetime import date, timedelta

from core.events import BarEvent, FillEvent, SignalEvent
from indicators.technical import RSI
from strategy.base import BacktestContext, Strategy


class OptionsWheelStrategy(Strategy):
    """Sell cash-secured puts when oversold, take assignment, sell covered calls.

    The Wheel Strategy:
    1. If we don't hold stock and RSI < rsi_threshold: sell a put at
       strike = close * (1 - otm_pct)
    2. If we hold stock (from assignment): sell a covered call at
       strike = close * (1 + otm_pct)

    Parameters
    ----------
    rsi_period:
        Look-back period for the RSI indicator (default 14).
    rsi_threshold:
        RSI level below which a cash-secured put is sold (default 40.0).
    otm_pct:
        Fraction out-of-the-money for both put and call strikes (default 0.05).
    dte_target:
        Target days-to-expiration when choosing the options expiry (default 30).
        Currently used informally — the strategy always selects the next
        standard monthly expiry (3rd Friday of the following month).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_threshold: float = 40.0,
        otm_pct: float = 0.05,
        dte_target: int = 30,
    ) -> None:
        super().__init__()
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.otm_pct = otm_pct
        self.dte_target = dte_target

        self.indicators["rsi"] = RSI(rsi_period)

        # symbol -> bool: tracks whether we notionally "own" the stock via
        # prior assignment (used when the real portfolio object is None in tests).
        self._has_stock: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        """Return 0 or 1 SignalEvent for *bar* depending on RSI and position state."""
        self.indicators["rsi"].update(bar.close)

        if not self.indicators["rsi"].is_ready:
            return []

        signals: list[SignalEvent] = []
        rsi_val = self.indicators["rsi"].value

        # Determine whether we currently hold the underlying equity
        if portfolio is not None:
            has_position = portfolio.has_position(bar.symbol)
        else:
            has_position = self._has_stock.get(bar.symbol, False)

        if not has_position:
            # Phase 1: sell a cash-secured put when the stock is oversold
            if rsi_val < self.rsi_threshold:
                strike = round(bar.close * (1 - self.otm_pct), 2)
                expiration = self._next_monthly_expiry(bar.date)
                signals.append(
                    SignalEvent(
                        symbol=bar.symbol,
                        direction="sell_put",
                        reason=(
                            f"RSI={rsi_val:.1f} < {self.rsi_threshold},"
                            f" sell CSP at ${strike}"
                        ),
                        strength=1.0,
                        option_type="put",
                        strike=strike,
                        expiration=expiration,
                    )
                )
        else:
            # Phase 2: sell a covered call against the held stock
            strike = round(bar.close * (1 + self.otm_pct), 2)
            expiration = self._next_monthly_expiry(bar.date)
            signals.append(
                SignalEvent(
                    symbol=bar.symbol,
                    direction="sell_call",
                    reason=f"Holding stock, sell CC at ${strike}",
                    strength=1.0,
                    option_type="call",
                    strike=strike,
                    expiration=expiration,
                )
            )

        return signals

    def warm_up_period(self) -> int:
        """RSI needs rsi_period + 1 prices before it produces a value."""
        return self.rsi_period + 1

    def on_fill(self, fill: FillEvent) -> None:
        """Track stock assignment / call-away via fill events."""
        if fill.action in ("BUY", "BTO"):
            self._has_stock[fill.symbol] = True
        elif fill.action in ("SELL", "STC"):
            self._has_stock[fill.symbol] = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_monthly_expiry(current_date: date) -> date:
        """Return the 3rd Friday of the month following *current_date*.

        Standard U.S. equity options expire on the 3rd Friday of each month.
        """
        # Advance to the next calendar month
        if current_date.month == 12:
            year, month = current_date.year + 1, 1
        else:
            year, month = current_date.year, current_date.month + 1

        # Find the first Friday of that month (weekday 4 == Friday)
        first_day = date(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)

        # The 3rd Friday is two weeks later
        third_friday = first_friday + timedelta(weeks=2)
        return third_friday
