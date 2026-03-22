"""General-purpose helper utilities for the trading bot."""
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# US market holidays (fixed + observed) 2020-2030
# Source: NYSE holiday schedule
# ---------------------------------------------------------------------------

_US_MARKET_HOLIDAYS: frozenset[date] = frozenset(
    [
        # --- 2020 ---
        date(2020, 1, 1),   # New Year's Day
        date(2020, 1, 20),  # MLK Day
        date(2020, 2, 17),  # Presidents' Day
        date(2020, 4, 10),  # Good Friday
        date(2020, 5, 25),  # Memorial Day
        date(2020, 7, 3),   # Independence Day (observed Fri)
        date(2020, 9, 7),   # Labor Day
        date(2020, 11, 26), # Thanksgiving
        date(2020, 12, 25), # Christmas
        # --- 2021 ---
        date(2021, 1, 1),
        date(2021, 1, 18),
        date(2021, 2, 15),
        date(2021, 4, 2),
        date(2021, 5, 31),
        date(2021, 7, 5),   # Independence Day observed Mon
        date(2021, 9, 6),
        date(2021, 11, 25),
        date(2021, 12, 24), # Christmas observed Fri
        # --- 2022 ---
        date(2022, 1, 17),
        date(2022, 2, 21),
        date(2022, 4, 15),
        date(2022, 5, 30),
        date(2022, 6, 20),  # Juneteenth observed Mon
        date(2022, 7, 4),
        date(2022, 9, 5),
        date(2022, 11, 24),
        date(2022, 12, 26), # Christmas observed Mon
        # --- 2023 ---
        date(2023, 1, 2),   # New Year's observed Mon
        date(2023, 1, 16),
        date(2023, 2, 20),
        date(2023, 4, 7),
        date(2023, 5, 29),
        date(2023, 6, 19),
        date(2023, 7, 4),
        date(2023, 9, 4),
        date(2023, 11, 23),
        date(2023, 12, 25),
        # --- 2024 ---
        date(2024, 1, 1),
        date(2024, 1, 15),
        date(2024, 2, 19),
        date(2024, 3, 29),
        date(2024, 5, 27),
        date(2024, 6, 19),
        date(2024, 7, 4),
        date(2024, 9, 2),
        date(2024, 11, 28),
        date(2024, 12, 25),
        # --- 2025 ---
        date(2025, 1, 1),
        date(2025, 1, 9),   # National Day of Mourning (Carter)
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 4, 18),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 11, 27),
        date(2025, 12, 25),
        # --- 2026 ---
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),   # Independence Day observed Fri
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
        # --- 2027 ---
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),
        date(2027, 5, 31),
        date(2027, 6, 18),  # Juneteenth observed Fri
        date(2027, 7, 5),   # Independence Day observed Mon
        date(2027, 9, 6),
        date(2027, 11, 25),
        date(2027, 12, 24), # Christmas observed Fri
        # --- 2028 ---
        date(2028, 1, 17),
        date(2028, 2, 21),
        date(2028, 4, 14),
        date(2028, 5, 29),
        date(2028, 6, 19),
        date(2028, 7, 4),
        date(2028, 9, 4),
        date(2028, 11, 23),
        date(2028, 12, 25),
        # --- 2029 ---
        date(2029, 1, 1),
        date(2029, 1, 15),
        date(2029, 2, 19),
        date(2029, 3, 30),
        date(2029, 5, 28),
        date(2029, 6, 19),
        date(2029, 7, 4),
        date(2029, 9, 3),
        date(2029, 11, 22),
        date(2029, 12, 25),
        # --- 2030 ---
        date(2030, 1, 1),
        date(2030, 1, 21),
        date(2030, 2, 18),
        date(2030, 4, 19),
        date(2030, 5, 27),
        date(2030, 6, 19),
        date(2030, 7, 4),
        date(2030, 9, 2),
        date(2030, 11, 28),
        date(2030, 12, 25),
    ]
)


def normalize_symbol(symbol: str) -> str:
    """Return *symbol* uppercased with surrounding whitespace stripped.

    Examples
    --------
    >>> normalize_symbol("  aapl  ")
    'AAPL'
    """
    return symbol.strip().upper()


def is_trading_day(d: date) -> bool:
    """Return ``True`` if *d* is a US market trading day.

    A day is NOT a trading day when it falls on:
    * Saturday (weekday == 5) or Sunday (weekday == 6)
    * A known US market holiday

    Parameters
    ----------
    d:
        The date to check.
    """
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return d not in _US_MARKET_HOLIDAYS


def get_trading_days(start: date, end: date) -> list[date]:
    """Return every trading day between *start* and *end* inclusive.

    Parameters
    ----------
    start:
        First date of the range (inclusive).
    end:
        Last date of the range (inclusive).

    Returns
    -------
    list[date]
        Sorted list of trading days.  Empty list if ``start > end``.
    """
    if start > end:
        return []
    days: list[date] = []
    current = start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days
