"""Data cleaners that repair common DailyBar quality issues."""
from __future__ import annotations

import logging
from copy import copy
from datetime import timedelta

from data.storage.models import DailyBar
from data.validation.validators import (
    ValidationResult,
    _expected_trading_days,
    validate_bar,
    validate_gap,
)

logger = logging.getLogger(__name__)


def forward_fill_gaps(
    bars: list[DailyBar], max_gap_days: int = 2
) -> list[DailyBar]:
    """
    Fill short gaps (1 to max_gap_days missing trading days) by forward-filling
    the previous bar's close price into synthetic bars with volume=0 and
    data_quality_score=0.5.

    Gaps larger than max_gap_days are left unfilled (a warning is logged).
    """
    if len(bars) < 2:
        return list(bars)

    sorted_bars = sorted(bars, key=lambda b: b.date)
    result: list[DailyBar] = [sorted_bars[0]]

    for i in range(1, len(sorted_bars)):
        prev_bar = sorted_bars[i - 1]
        curr_bar = sorted_bars[i]
        missing = _expected_trading_days(prev_bar.date, curr_bar.date)

        if not missing:
            result.append(curr_bar)
            continue

        if len(missing) <= max_gap_days:
            for missing_date in missing:
                synthetic = DailyBar(
                    symbol=prev_bar.symbol,
                    date=missing_date,
                    open=prev_bar.close,
                    high=prev_bar.close,
                    low=prev_bar.close,
                    close=prev_bar.close,
                    adj_close=prev_bar.adj_close,
                    volume=0,
                    vwap=prev_bar.close,
                    data_quality_score=0.5,
                )
                result.append(synthetic)
        else:
            logger.warning(
                "Gap of %d trading days between %s and %s is too large to fill "
                "(max_gap_days=%d). Leaving gap unfilled.",
                len(missing),
                prev_bar.date,
                curr_bar.date,
                max_gap_days,
            )

        result.append(curr_bar)

    return result


def flag_and_clean(
    bars: list[DailyBar],
) -> tuple[list[DailyBar], list[ValidationResult]]:
    """
    Run all validators against every bar, apply forward-fill cleaning, and
    return the cleaned bar list together with all issues found.
    """
    if not bars:
        return [], []

    sorted_bars = sorted(bars, key=lambda b: b.date)
    all_issues: list[ValidationResult] = []

    # Per-bar validation
    for i, bar in enumerate(sorted_bars):
        prev_bar = sorted_bars[i - 1] if i > 0 else None
        # Compute 20-day average volume if enough history is available
        if i >= 20:
            avg_vol = sum(b.volume for b in sorted_bars[i - 20 : i]) / 20
        else:
            avg_vol = None
        issues = validate_bar(bar, prev_bar=prev_bar, avg_volume_20d=avg_vol)
        all_issues.extend(issues)

    # Gap validation across the full series
    gap_issues = validate_gap(sorted_bars)
    all_issues.extend(gap_issues)

    # Apply forward-fill to patch small gaps
    cleaned_bars = forward_fill_gaps(sorted_bars)

    return cleaned_bars, all_issues
