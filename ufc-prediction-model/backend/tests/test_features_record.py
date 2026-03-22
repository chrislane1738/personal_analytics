"""Tests for career record feature computation."""
import pytest
import pandas as pd
from features.record import compute_record_features, compute_career_stats_at_date

def test_compute_career_stats_at_date():
    """Stats must only include fights BEFORE the target date (leakage prevention)."""
    fights = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-01-01", "2023-06-01", "2024-01-01", "2024-06-01"]),
        "fighter": ["A", "A", "A", "A"],
        "result": ["Win", "Win", "Loss", "Win"],
        "method": ["KO", "Decision", "KO", "Submission"],
    })
    stats = compute_career_stats_at_date(fights, "A", pd.Timestamp("2024-01-01"))
    assert stats["wins"] == 2
    assert stats["losses"] == 0
    assert stats["win_streak"] == 2
    assert stats["ko_win_pct"] == 0.5

def test_win_streak_resets_on_loss():
    fights = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-01-01", "2023-03-01", "2023-06-01"]),
        "fighter": ["A", "A", "A"],
        "result": ["Win", "Loss", "Win"],
        "method": ["KO", "KO", "Decision"],
    })
    stats = compute_career_stats_at_date(fights, "A", pd.Timestamp("2024-01-01"))
    assert stats["win_streak"] == 1
    assert stats["loss_streak"] == 0

def test_record_differentials():
    a_stats = {"wins": 15, "losses": 3, "win_rate": 0.83, "finish_rate": 0.6, "ufc_fights": 18, "win_streak": 3, "loss_streak": 0}
    b_stats = {"wins": 10, "losses": 5, "win_rate": 0.67, "finish_rate": 0.4, "ufc_fights": 15, "win_streak": 1, "loss_streak": 0}
    result = compute_record_features(a_stats, b_stats)
    assert result["win_rate_diff"] == pytest.approx(0.16, abs=0.01)
    assert result["experience_diff"] == 3
