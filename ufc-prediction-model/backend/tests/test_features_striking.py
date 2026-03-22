"""Tests for striking metric feature computation."""
import pytest
import pandas as pd
from features.striking import compute_striking_averages, compute_striking_features

def test_striking_averages_time_aware():
    fight_stats = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-01-01", "2023-06-01", "2024-01-01"]),
        "fighter": ["A", "A", "A"],
        "slpm": [4.0, 6.0, 8.0],
        "sapm": [3.0, 4.0, 2.0],
        "str_acc": [0.45, 0.55, 0.60],
        "str_def": [0.55, 0.60, 0.65],
    })
    avgs = compute_striking_averages(fight_stats, "A", pd.Timestamp("2024-01-01"))
    assert avgs["slpm"] == pytest.approx(5.0, abs=0.01)
    assert avgs["sapm"] == pytest.approx(3.5, abs=0.01)

def test_striking_differentials():
    a_avgs = {"slpm": 5.0, "sapm": 3.0, "str_acc": 0.50, "str_def": 0.60}
    b_avgs = {"slpm": 4.0, "sapm": 4.5, "str_acc": 0.45, "str_def": 0.55}
    result = compute_striking_features(a_avgs, b_avgs)
    assert result["slpm_diff"] == pytest.approx(1.0)
    assert result["str_acc_diff"] == pytest.approx(0.05)
    assert result["a_strike_differential"] == pytest.approx(2.0)
    assert result["b_strike_differential"] == pytest.approx(-0.5)
