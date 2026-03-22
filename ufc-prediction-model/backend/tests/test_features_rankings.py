"""Tests for rankings feature computation."""
import pytest
import pandas as pd
from features.rankings import get_ranking_at_date, compute_rankings_features

def test_ranking_lookup_uses_most_recent_before_date():
    rankings = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
        "fighter": ["Jon Jones", "Jon Jones", "Jon Jones"],
        "rank": [1, 1, 2],
        "weight_class": ["Heavyweight", "Heavyweight", "Heavyweight"],
    })
    rank = get_ranking_at_date(rankings, "Jon Jones", "Heavyweight", pd.Timestamp("2024-01-10"))
    assert rank == 1

def test_unranked_fighter():
    rankings = pd.DataFrame(columns=["date", "fighter", "rank", "weight_class"])
    rank = get_ranking_at_date(rankings, "Unknown", "Lightweight", pd.Timestamp("2024-01-01"))
    assert rank == 0
