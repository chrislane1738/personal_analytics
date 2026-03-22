"""Tests for recent form feature computation."""
import pytest
import pandas as pd
from features.form import compute_form_features

def test_recent_form_last_3():
    fights = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]),
        "fighter": ["A", "A", "A", "A"],
        "result": ["Win", "Loss", "Win", "Win"],
    })
    form = compute_form_features(fights, "A", pd.Timestamp("2024-01-01"))
    assert form["last_3_wins"] == 2
    assert form["last_3_losses"] == 1
    assert form["days_since_last_fight"] == 92

def test_debut_fighter_form():
    fights = pd.DataFrame(columns=["event_date", "fighter", "result"])
    form = compute_form_features(fights, "A", pd.Timestamp("2024-01-01"))
    assert form["last_3_wins"] == 0
    assert form["is_debut"] is True
