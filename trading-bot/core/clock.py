"""Clock: iterates over trading dates within a date range.

Supports either an explicit list of trading days (e.g., from an exchange
calendar) or a simple weekday approximation (Mon-Fri).
"""

from __future__ import annotations

from datetime import date, timedelta


class Clock:
    """Iterates over trading dates between *start* and *end* inclusive.

    Parameters
    ----------
    start:
        First date of the range (inclusive).
    end:
        Last date of the range (inclusive).
    trading_days:
        Optional explicit list of known trading days.  When provided, only
        dates within [start, end] that appear in this list are used.
        When omitted, weekdays (Mon–Fri) are generated as an approximation.
    """

    def __init__(
        self,
        start: date,
        end: date,
        trading_days: list[date] | None = None,
    ) -> None:
        if trading_days is not None:
            self.dates = sorted(d for d in trading_days if start <= d <= end)
        else:
            # Generate weekday dates (simple approximation — ignores holidays)
            self.dates: list[date] = []
            current = start
            while current <= end:
                if current.weekday() < 5:  # Mon=0 … Fri=4
                    self.dates.append(current)
                current += timedelta(days=1)

    def __iter__(self):
        return iter(self.dates)

    def __len__(self) -> int:
        return len(self.dates)
