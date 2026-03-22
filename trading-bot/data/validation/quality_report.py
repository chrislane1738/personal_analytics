"""Quality scoring and report generation for DailyBar collections."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from data.storage.models import DailyBar
from data.validation.validators import ValidationResult

# Score deductions per issue severity
_DEDUCTIONS = {
    "error": 0.3,
    "warning": 0.1,
}


def compute_quality_score(
    bar: DailyBar, issues: list[ValidationResult]
) -> float:
    """
    Return a quality score in [0.0, 1.0] for a bar.

    Starts at 1.0 and deducts 0.3 for each error and 0.1 for each warning.
    """
    score = 1.0
    for issue in issues:
        score -= _DEDUCTIONS.get(issue.severity, 0.0)
    return max(0.0, score)


def generate_quality_report(
    bars: list[DailyBar], issues: list[ValidationResult]
) -> dict[str, Any]:
    """
    Produce a summary dict describing the overall quality of a bar collection.

    Keys:
        total_bars          — number of bars analysed
        total_issues        — total number of ValidationResults
        issues_by_type      — {issue_type: count} mapping
        issues_by_severity  — {severity: count} mapping
        avg_quality_score   — mean data_quality_score across all bars
        worst_bars          — list of (symbol, date, data_quality_score) for bars
                              whose score < 1.0, sorted ascending by score
    """
    issues_by_type: dict[str, int] = defaultdict(int)
    issues_by_severity: dict[str, int] = defaultdict(int)

    for issue in issues:
        issues_by_type[issue.issue_type] += 1
        issues_by_severity[issue.severity] += 1

    avg_quality_score = (
        sum(b.data_quality_score for b in bars) / len(bars) if bars else 0.0
    )

    worst_bars = sorted(
        [
            {
                "symbol": b.symbol,
                "date": b.date,
                "data_quality_score": b.data_quality_score,
            }
            for b in bars
            if b.data_quality_score < 1.0
        ],
        key=lambda x: x["data_quality_score"],
    )

    return {
        "total_bars": len(bars),
        "total_issues": len(issues),
        "issues_by_type": dict(issues_by_type),
        "issues_by_severity": dict(issues_by_severity),
        "avg_quality_score": avg_quality_score,
        "worst_bars": worst_bars,
    }
