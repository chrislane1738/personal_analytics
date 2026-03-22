"""Tests for the data validation pipeline (Task 4)."""
from __future__ import annotations

from datetime import date

import pytest

from data.storage.models import DailyBar
from data.validation.cleaners import flag_and_clean, forward_fill_gaps
from data.validation.quality_report import compute_quality_score, generate_quality_report
from data.validation.validators import (
    ValidationResult,
    validate_bar,
    validate_gap,
    validate_range,
    validate_spike,
    validate_volume_anomaly,
)
from tests.fixtures.bad_data import (
    GOOD_BAR,
    HIGH_LOW_INVERTED,
    MISSING_GAP_BARS,
    NEGATIVE_PRICE_BAR,
    SPIKE_BAR,
    SPIKE_PREV_BAR,
    ZERO_VOLUME_BAR,
)


# ---------------------------------------------------------------------------
# validate_range
# ---------------------------------------------------------------------------


class TestValidateRange:
    def test_good_bar_passes(self):
        result = validate_range(GOOD_BAR)
        assert result is None

    def test_negative_close_caught(self):
        result = validate_range(NEGATIVE_PRICE_BAR)
        assert result is not None
        assert result.is_valid is False
        assert result.severity == "error"
        assert "NEGATIVE_OR_ZERO_PRICE" in result.issue_type or "NEGATIVE" in result.issue_type

    def test_inverted_high_low_caught(self):
        result = validate_range(HIGH_LOW_INVERTED)
        assert result is not None
        assert result.is_valid is False
        assert result.severity == "error"
        assert "HIGH_LOW_INVERTED" in result.issue_type

    def test_high_below_open_caught(self):
        bar = DailyBar(
            symbol="TEST",
            date=date(2024, 1, 2),
            open=190.0,
            high=185.0,  # below open
            low=184.0,
            close=186.0,
            adj_close=186.0,
            volume=1_000_000,
            vwap=186.0,
        )
        result = validate_range(bar)
        assert result is not None
        assert "HIGH_BELOW_OPEN" in result.issue_type

    def test_low_above_close_caught(self):
        bar = DailyBar(
            symbol="TEST",
            date=date(2024, 1, 2),
            open=188.0,
            high=189.0,
            low=187.0,  # above close but below open — triggers LOW_ABOVE_CLOSE
            close=185.0,
            adj_close=185.0,
            volume=1_000_000,
            vwap=186.0,
        )
        result = validate_range(bar)
        assert result is not None
        assert "LOW_ABOVE_CLOSE" in result.issue_type


# ---------------------------------------------------------------------------
# validate_spike
# ---------------------------------------------------------------------------


class TestValidateSpike:
    def test_sixty_percent_move_caught(self):
        result = validate_spike(SPIKE_BAR, SPIKE_PREV_BAR)
        assert result is not None
        assert result.is_valid is False
        assert result.severity == "warning"
        assert "PRICE_SPIKE" in result.issue_type

    def test_five_percent_move_allowed(self):
        small_move_bar = DailyBar(
            symbol="AAPL",
            date=date(2024, 1, 3),
            open=186.0,
            high=196.0,
            low=185.0,
            close=195.3,  # ~5% above GOOD_BAR close of 186.0
            adj_close=195.3,
            volume=15_000_000,
            vwap=190.0,
        )
        result = validate_spike(small_move_bar, GOOD_BAR)
        assert result is None

    def test_no_prev_bar_returns_none(self):
        result = validate_spike(GOOD_BAR, None)
        assert result is None

    def test_exactly_fifty_pct_boundary(self):
        """Exactly 50% move should NOT trigger (threshold is strictly > 50%)."""
        bar_50 = DailyBar(
            symbol="AAPL",
            date=date(2024, 1, 3),
            open=279.0,
            high=280.0,
            low=278.0,
            close=279.0,  # exactly 50% above 186.0
            adj_close=279.0,
            volume=10_000_000,
            vwap=279.0,
        )
        result = validate_spike(bar_50, GOOD_BAR)
        assert result is None


# ---------------------------------------------------------------------------
# validate_volume_anomaly
# ---------------------------------------------------------------------------


class TestValidateVolumeAnomaly:
    def test_zero_volume_caught(self):
        result = validate_volume_anomaly(ZERO_VOLUME_BAR, None)
        assert result is not None
        assert result.is_valid is False
        assert result.severity == "warning"
        assert "ZERO_VOLUME" in result.issue_type

    def test_fifteen_times_avg_volume_caught(self):
        bar = DailyBar(
            symbol="AAPL",
            date=date(2024, 1, 2),
            open=185.0,
            high=187.0,
            low=184.0,
            close=186.0,
            adj_close=185.5,
            volume=150_000_000,  # 15x avg of 10_000_000
            vwap=185.8,
        )
        result = validate_volume_anomaly(bar, avg_volume_20d=10_000_000.0)
        assert result is not None
        assert result.is_valid is False
        assert "VOLUME_SPIKE" in result.issue_type

    def test_normal_volume_passes(self):
        result = validate_volume_anomaly(GOOD_BAR, avg_volume_20d=15_000_000.0)
        assert result is None

    def test_no_avg_volume_no_spike_flagged(self):
        """Without a 20-day avg, only zero-volume check should apply."""
        result = validate_volume_anomaly(GOOD_BAR, avg_volume_20d=None)
        assert result is None


# ---------------------------------------------------------------------------
# validate_gap
# ---------------------------------------------------------------------------


class TestValidateGap:
    def test_missing_trading_day_caught(self):
        issues = validate_gap(MISSING_GAP_BARS)
        assert len(issues) >= 1
        assert any("MISSING_TRADING_DAY" in i.issue_type for i in issues)

    def test_weekend_not_flagged(self):
        """Friday -> Monday should produce no gap issues."""
        fri = DailyBar(
            symbol="AAPL",
            date=date(2024, 1, 5),  # Friday
            open=185.0, high=187.0, low=184.0, close=186.0,
            adj_close=186.0, volume=1_000_000, vwap=186.0,
        )
        mon = DailyBar(
            symbol="AAPL",
            date=date(2024, 1, 8),  # Monday
            open=186.0, high=188.0, low=185.0, close=187.0,
            adj_close=187.0, volume=1_000_000, vwap=187.0,
        )
        issues = validate_gap([fri, mon])
        assert issues == []

    def test_single_bar_no_issues(self):
        issues = validate_gap([GOOD_BAR])
        assert issues == []

    def test_consecutive_days_no_issues(self):
        """Tuesday -> Wednesday (no gap) should not be flagged."""
        tue = DailyBar(
            symbol="AAPL",
            date=date(2024, 1, 2),  # Tuesday
            open=185.0, high=187.0, low=184.0, close=186.0,
            adj_close=186.0, volume=1_000_000, vwap=186.0,
        )
        wed = DailyBar(
            symbol="AAPL",
            date=date(2024, 1, 3),  # Wednesday
            open=186.0, high=188.0, low=185.0, close=187.0,
            adj_close=187.0, volume=1_000_000, vwap=187.0,
        )
        issues = validate_gap([tue, wed])
        assert issues == []


# ---------------------------------------------------------------------------
# forward_fill_gaps
# ---------------------------------------------------------------------------


class TestForwardFillGaps:
    def test_fills_one_day_gap(self):
        result = forward_fill_gaps(MISSING_GAP_BARS)
        dates = [b.date for b in result]
        assert date(2024, 1, 3) in dates, "Wednesday gap should be filled"

    def test_filled_bar_properties(self):
        result = forward_fill_gaps(MISSING_GAP_BARS)
        filled = next(b for b in result if b.date == date(2024, 1, 3))
        prev_close = MISSING_GAP_BARS[0].close  # 186.0
        assert filled.open == prev_close
        assert filled.high == prev_close
        assert filled.low == prev_close
        assert filled.close == prev_close
        assert filled.volume == 0
        assert filled.data_quality_score == 0.5

    def test_does_not_fill_five_day_gap(self):
        """A 5-day gap exceeds max_gap_days=2 and should be left unfilled."""
        bar_a = DailyBar(
            symbol="AAPL",
            date=date(2024, 1, 2),  # Tuesday
            open=185.0, high=187.0, low=184.0, close=186.0,
            adj_close=186.0, volume=1_000_000, vwap=186.0,
        )
        # Jump to 2024-01-09 (next Tuesday) — 3 missing weekdays: Wed/Thu/Fri
        # Use a bigger jump: 2024-01-02 -> 2024-01-10 = Wed/Thu/Fri/Mon/Tue missing (5 days)
        bar_b = DailyBar(
            symbol="AAPL",
            date=date(2024, 1, 11),  # Thursday — 7 calendar days later
            open=190.0, high=192.0, low=189.0, close=191.0,
            adj_close=191.0, volume=1_000_000, vwap=190.5,
        )
        result = forward_fill_gaps([bar_a, bar_b], max_gap_days=2)
        result_dates = {b.date for b in result}
        # Missing weekdays: Jan 3,4,5,8,9,10 — all 6 should NOT be filled
        for missing_date in [
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
            date(2024, 1, 8),
            date(2024, 1, 9),
            date(2024, 1, 10),
        ]:
            assert missing_date not in result_dates

    def test_empty_list_returns_empty(self):
        assert forward_fill_gaps([]) == []

    def test_single_bar_returns_unchanged(self):
        result = forward_fill_gaps([GOOD_BAR])
        assert len(result) == 1
        assert result[0] == GOOD_BAR


# ---------------------------------------------------------------------------
# compute_quality_score
# ---------------------------------------------------------------------------


class TestComputeQualityScore:
    def test_clean_bar_returns_one(self):
        score = compute_quality_score(GOOD_BAR, [])
        assert score == 1.0

    def test_single_error_deducts_0_3(self):
        error = ValidationResult(
            is_valid=False, issue_type="TEST_ERROR", severity="error", details=""
        )
        score = compute_quality_score(GOOD_BAR, [error])
        assert abs(score - 0.7) < 1e-9

    def test_single_warning_deducts_0_1(self):
        warning = ValidationResult(
            is_valid=False, issue_type="TEST_WARNING", severity="warning", details=""
        )
        score = compute_quality_score(GOOD_BAR, [warning])
        assert abs(score - 0.9) < 1e-9

    def test_multiple_errors_cannot_go_below_zero(self):
        errors = [
            ValidationResult(
                is_valid=False, issue_type=f"ERR_{i}", severity="error", details=""
            )
            for i in range(10)
        ]
        score = compute_quality_score(GOOD_BAR, errors)
        assert score == 0.0

    def test_mixed_issues_deduct_correctly(self):
        issues = [
            ValidationResult(is_valid=False, issue_type="E1", severity="error", details=""),
            ValidationResult(is_valid=False, issue_type="W1", severity="warning", details=""),
        ]
        score = compute_quality_score(GOOD_BAR, issues)
        # 1.0 - 0.3 - 0.1 = 0.6
        assert abs(score - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# generate_quality_report
# ---------------------------------------------------------------------------


class TestGenerateQualityReport:
    def test_report_structure(self):
        issues = [
            ValidationResult(
                is_valid=False, issue_type="ZERO_VOLUME", severity="warning", details=""
            )
        ]
        report = generate_quality_report([GOOD_BAR], issues)
        assert "total_bars" in report
        assert "total_issues" in report
        assert "issues_by_type" in report
        assert "issues_by_severity" in report
        assert "avg_quality_score" in report
        assert "worst_bars" in report

    def test_total_bars_count(self):
        report = generate_quality_report([GOOD_BAR, ZERO_VOLUME_BAR], [])
        assert report["total_bars"] == 2

    def test_issues_by_type_counts(self):
        issues = [
            ValidationResult(is_valid=False, issue_type="A", severity="warning", details=""),
            ValidationResult(is_valid=False, issue_type="A", severity="warning", details=""),
            ValidationResult(is_valid=False, issue_type="B", severity="error", details=""),
        ]
        report = generate_quality_report([GOOD_BAR], issues)
        assert report["issues_by_type"]["A"] == 2
        assert report["issues_by_type"]["B"] == 1

    def test_empty_input(self):
        report = generate_quality_report([], [])
        assert report["total_bars"] == 0
        assert report["total_issues"] == 0
        assert report["avg_quality_score"] == 0.0


# ---------------------------------------------------------------------------
# Full pipeline: flag_and_clean
# ---------------------------------------------------------------------------


class TestFlagAndClean:
    def _make_bar(self, sym: str, d: date, close: float = 186.0,
                  volume: int = 1_000_000) -> DailyBar:
        return DailyBar(
            symbol=sym, date=d,
            open=close, high=close + 1.0, low=close - 1.0, close=close,
            adj_close=close, volume=volume, vwap=close,
        )

    def test_good_bars_produce_no_issues(self):
        bars = [
            self._make_bar("AAPL", date(2024, 1, 2)),
            self._make_bar("AAPL", date(2024, 1, 3)),
            self._make_bar("AAPL", date(2024, 1, 4)),
        ]
        cleaned, issues = flag_and_clean(bars)
        assert issues == []
        assert len(cleaned) == 3

    def test_zero_volume_bar_is_flagged(self):
        bars = [
            self._make_bar("AAPL", date(2024, 1, 2)),
            self._make_bar("AAPL", date(2024, 1, 3), volume=0),
        ]
        _, issues = flag_and_clean(bars)
        assert any("ZERO_VOLUME" in i.issue_type for i in issues)

    def test_gap_is_filled_in_cleaned_output(self):
        """A one-day missing trading day should be filled after flag_and_clean."""
        cleaned, issues = flag_and_clean(MISSING_GAP_BARS)
        cleaned_dates = {b.date for b in cleaned}
        assert date(2024, 1, 3) in cleaned_dates
        assert any("MISSING_TRADING_DAY" in i.issue_type for i in issues)

    def test_mixed_data_all_issues_caught(self):
        """Run the full pipeline on a mix of good and bad bars."""
        bars = [
            GOOD_BAR,
            ZERO_VOLUME_BAR,
            HIGH_LOW_INVERTED,
        ]
        _, issues = flag_and_clean(bars)
        issue_types = {i.issue_type for i in issues}
        assert "ZERO_VOLUME" in issue_types
        # HIGH_LOW_INVERTED triggers HIGH_LOW_INVERTED
        assert any("HIGH_LOW" in t or "HIGH_BELOW" in t or "LOW_ABOVE" in t
                   for t in issue_types)

    def test_empty_input(self):
        cleaned, issues = flag_and_clean([])
        assert cleaned == []
        assert issues == []

    def test_spike_bar_flagged(self):
        bars = [SPIKE_PREV_BAR, SPIKE_BAR]
        _, issues = flag_and_clean(bars)
        assert any("PRICE_SPIKE" in i.issue_type for i in issues)
