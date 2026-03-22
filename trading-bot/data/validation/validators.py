"""Validators for DailyBar data quality checks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from data.storage.models import DailyBar


@dataclass
class ValidationResult:
    is_valid: bool
    issue_type: str
    severity: str  # 'warning' or 'error'
    details: str


def validate_schema(bar: DailyBar) -> Optional[ValidationResult]:
    """Check all required fields are present and have correct types."""
    required_fields = {
        "symbol": str,
        "date": date,
        "open": (int, float),
        "high": (int, float),
        "low": (int, float),
        "close": (int, float),
        "adj_close": (int, float),
        "volume": (int,),
        "vwap": (int, float),
        "data_quality_score": (int, float),
    }
    for field_name, expected_type in required_fields.items():
        value = getattr(bar, field_name, None)
        if value is None and field_name != "data_quality_score":
            return ValidationResult(
                is_valid=False,
                issue_type="MISSING_FIELD",
                severity="error",
                details=f"Field '{field_name}' is missing or None",
            )
        if value is not None and not isinstance(value, expected_type):
            return ValidationResult(
                is_valid=False,
                issue_type="WRONG_TYPE",
                severity="error",
                details=(
                    f"Field '{field_name}' expected {expected_type}, "
                    f"got {type(value).__name__}"
                ),
            )
    return None


def validate_range(bar: DailyBar) -> Optional[ValidationResult]:
    """Check price and volume are within valid ranges."""
    # All price fields must be positive
    for field_name in ("open", "high", "low", "close", "adj_close", "vwap"):
        value = getattr(bar, field_name)
        if value <= 0:
            return ValidationResult(
                is_valid=False,
                issue_type="NEGATIVE_OR_ZERO_PRICE",
                severity="error",
                details=f"Field '{field_name}' must be positive, got {value}",
            )

    # Volume must be non-negative
    if bar.volume < 0:
        return ValidationResult(
            is_valid=False,
            issue_type="NEGATIVE_VOLUME",
            severity="error",
            details=f"Volume must be >= 0, got {bar.volume}",
        )

    # High must be >= low
    if bar.high < bar.low:
        return ValidationResult(
            is_valid=False,
            issue_type="HIGH_LOW_INVERTED",
            severity="error",
            details=f"High ({bar.high}) < Low ({bar.low})",
        )

    # High must be >= open and close
    if bar.high < bar.open:
        return ValidationResult(
            is_valid=False,
            issue_type="HIGH_BELOW_OPEN",
            severity="error",
            details=f"High ({bar.high}) < Open ({bar.open})",
        )
    if bar.high < bar.close:
        return ValidationResult(
            is_valid=False,
            issue_type="HIGH_BELOW_CLOSE",
            severity="error",
            details=f"High ({bar.high}) < Close ({bar.close})",
        )

    # Low must be <= open and close
    if bar.low > bar.open:
        return ValidationResult(
            is_valid=False,
            issue_type="LOW_ABOVE_OPEN",
            severity="error",
            details=f"Low ({bar.low}) > Open ({bar.open})",
        )
    if bar.low > bar.close:
        return ValidationResult(
            is_valid=False,
            issue_type="LOW_ABOVE_CLOSE",
            severity="error",
            details=f"Low ({bar.low}) > Close ({bar.close})",
        )

    return None


def validate_spike(
    bar: DailyBar, prev_bar: Optional[DailyBar]
) -> Optional[ValidationResult]:
    """Flag if the daily move vs previous close exceeds 50%."""
    if prev_bar is None:
        return None

    if prev_bar.close == 0:
        return None

    move = abs(bar.close - prev_bar.close) / prev_bar.close
    if move > 0.50:
        return ValidationResult(
            is_valid=False,
            issue_type="PRICE_SPIKE",
            severity="warning",
            details=(
                f"Daily move of {move:.1%} exceeds 50% threshold "
                f"(prev close={prev_bar.close}, close={bar.close})"
            ),
        )
    return None


def validate_volume_anomaly(
    bar: DailyBar, avg_volume_20d: Optional[float]
) -> Optional[ValidationResult]:
    """Flag zero volume on a trading day, or volume > 10x 20-day average."""
    if bar.volume == 0:
        return ValidationResult(
            is_valid=False,
            issue_type="ZERO_VOLUME",
            severity="warning",
            details="Volume is zero on a trading day",
        )

    if avg_volume_20d is not None and avg_volume_20d > 0:
        ratio = bar.volume / avg_volume_20d
        if ratio > 10.0:
            return ValidationResult(
                is_valid=False,
                issue_type="VOLUME_SPIKE",
                severity="warning",
                details=(
                    f"Volume ({bar.volume:,}) is {ratio:.1f}x the "
                    f"20-day average ({avg_volume_20d:,.0f})"
                ),
            )

    return None


def validate_bar(
    bar: DailyBar,
    prev_bar: Optional[DailyBar] = None,
    avg_volume_20d: Optional[float] = None,
) -> list[ValidationResult]:
    """Run all validators against a single bar and return all issues found."""
    issues: list[ValidationResult] = []

    schema_result = validate_schema(bar)
    if schema_result is not None:
        issues.append(schema_result)
        # Cannot safely run further checks if schema is broken
        return issues

    range_result = validate_range(bar)
    if range_result is not None:
        issues.append(range_result)

    spike_result = validate_spike(bar, prev_bar)
    if spike_result is not None:
        issues.append(spike_result)

    volume_result = validate_volume_anomaly(bar, avg_volume_20d)
    if volume_result is not None:
        issues.append(volume_result)

    return issues


def _is_weekend(d: date) -> bool:
    """Return True if the date falls on a Saturday or Sunday."""
    return d.weekday() >= 5  # 5=Saturday, 6=Sunday


def _expected_trading_days(start: date, end: date) -> list[date]:
    """Return weekday dates between start (exclusive) and end (exclusive)."""
    days: list[date] = []
    current = start + timedelta(days=1)
    while current < end:
        if not _is_weekend(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def validate_gap(bars: list[DailyBar]) -> list[ValidationResult]:
    """Check for missing trading days (weekdays) between consecutive bars."""
    if len(bars) < 2:
        return []

    # Sort bars by date to ensure consistent ordering
    sorted_bars = sorted(bars, key=lambda b: b.date)
    bar_dates = {b.date for b in sorted_bars}
    issues: list[ValidationResult] = []

    for i in range(len(sorted_bars) - 1):
        bar_a = sorted_bars[i]
        bar_b = sorted_bars[i + 1]
        missing = _expected_trading_days(bar_a.date, bar_b.date)
        # Filter out dates that actually exist in the list
        truly_missing = [d for d in missing if d not in bar_dates]
        if truly_missing:
            issues.append(
                ValidationResult(
                    is_valid=False,
                    issue_type="MISSING_TRADING_DAY",
                    severity="warning",
                    details=(
                        f"Missing {len(truly_missing)} trading day(s) between "
                        f"{bar_a.date} and {bar_b.date}: "
                        + ", ".join(str(d) for d in truly_missing)
                    ),
                )
            )

    return issues
