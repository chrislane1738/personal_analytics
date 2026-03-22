"""Tests for career record feature computation."""
import pytest
import pandas as pd
from features.record import compute_record_features, compute_career_stats_at_date, compute_experience_features

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
    a_stats = {"wins": 15, "losses": 3, "win_rate": 0.83, "finish_rate": 0.6, "ufc_fights": 18,
               "win_streak": 3, "loss_streak": 0, "five_round_fights": 4}
    b_stats = {"wins": 10, "losses": 5, "win_rate": 0.67, "finish_rate": 0.4, "ufc_fights": 15,
               "win_streak": 1, "loss_streak": 0, "five_round_fights": 1}
    result = compute_record_features(a_stats, b_stats)
    assert result["win_rate_diff"] == pytest.approx(0.16, abs=0.01)
    assert result["experience_diff"] == 3
    assert result["five_round_diff"] == 3


def test_five_round_fight_count():
    """Should count scheduled 5-round fights from prior history."""
    fights = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01"]),
        "fighter": ["A", "A", "A"],
        "result": ["Win", "Win", "Win"],
        "method": ["Decision", "KO", "Decision"],
        "rounds_scheduled": [3, 3, 5],
    })
    stats = compute_career_stats_at_date(fights, "A", pd.Timestamp("2024-01-01"))
    assert stats["five_round_fights"] == 1


def test_experience_features_strength_of_schedule():
    """Fighter who beat strong opponents should have higher experience score."""
    fights = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01"]),
        "fighter": ["A", "A", "A"],
        "opponent": ["Strong1", "Strong2", "Strong3"],
        "result": ["Win", "Win", "Win"],
    })
    # Mock opponent records: all have 80% win rate (strong opponents)
    strong_records = {
        "Strong1": {"win_rate": 0.80},
        "Strong2": {"win_rate": 0.75},
        "Strong3": {"win_rate": 0.85},
    }
    exp = compute_experience_features(fights, "A", pd.Timestamp("2024-01-01"), strong_records)
    assert exp["avg_opponent_win_rate"] == pytest.approx(0.80, abs=0.01)
    assert exp["experience_score"] > 0

    # Compare with fighter who beat weak opponents
    weak_fights = fights.copy()
    weak_fights["opponent"] = ["Weak1", "Weak2", "Weak3"]
    weak_records = {
        "Weak1": {"win_rate": 0.30},
        "Weak2": {"win_rate": 0.25},
        "Weak3": {"win_rate": 0.35},
    }
    weak_exp = compute_experience_features(weak_fights, "A", pd.Timestamp("2024-01-01"), weak_records)
    assert weak_exp["avg_opponent_win_rate"] == pytest.approx(0.30, abs=0.01)
    # Strong schedule fighter should have higher experience score
    assert exp["experience_score"] > weak_exp["experience_score"]


def test_experience_features_empty_fighter():
    fights = pd.DataFrame(columns=["event_date", "fighter", "opponent", "result"])
    exp = compute_experience_features(fights, "A", pd.Timestamp("2024-01-01"))
    assert exp["avg_opponent_win_rate"] == 0.0
    assert exp["experience_score"] == 0.0
