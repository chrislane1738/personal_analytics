"""Tests for grappling metric feature computation."""
import pytest
import pandas as pd
from features.grappling import compute_grappling_averages, compute_grappling_features

def test_grappling_averages_time_aware():
    fight_stats = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-01-01", "2023-06-01"]),
        "fighter": ["A", "A"],
        "td_avg": [3.0, 5.0],
        "td_acc": [0.40, 0.60],
        "td_def": [0.70, 0.80],
        "sub_avg": [1.0, 2.0],
    })
    avgs = compute_grappling_averages(fight_stats, "A", pd.Timestamp("2024-01-01"))
    assert avgs["td_avg"] == pytest.approx(4.0)
    assert avgs["sub_avg"] == pytest.approx(1.5)

def test_grappling_differentials():
    a = {"td_avg": 4.0, "td_acc": 0.50, "td_def": 0.75, "sub_avg": 1.5}
    b = {"td_avg": 2.0, "td_acc": 0.35, "td_def": 0.80, "sub_avg": 0.5}
    result = compute_grappling_features(a, b)
    assert result["td_avg_diff"] == pytest.approx(2.0)
    assert result["grappling_advantage"] == pytest.approx(4.0 * 0.50 - 2.0 * 0.35, abs=0.01)
