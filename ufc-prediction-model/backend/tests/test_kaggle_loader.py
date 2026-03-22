"""Tests for Kaggle dataset loading and merging."""
import pytest
import pandas as pd
from scrapers.kaggle_loader import (
    load_fight_data, load_rankings_data, clean_fight_data,
    exclude_non_binary_outcomes, validate_required_columns,
)

def test_exclude_non_binary_outcomes():
    """Draws, no-contests, DQs must be excluded per spec §3.1 Rule #6."""
    df = pd.DataFrame({
        "winner": ["Fighter A", "Fighter B", "Draw", "NC", "Fighter A"],
        "result": ["KO", "Decision", "Draw", "No Contest", "DQ"],
        "fighter_a": ["A", "B", "C", "D", "E"],
        "fighter_b": ["F", "G", "H", "I", "J"],
    })
    result = exclude_non_binary_outcomes(df)
    assert len(result) == 2
    assert "Draw" not in result["result"].values
    assert "No Contest" not in result["result"].values
    assert "DQ" not in result["result"].values

def test_validate_required_columns_passes():
    df = pd.DataFrame({
        "fighter_a": ["A"], "fighter_b": ["B"], "winner": ["A"],
        "event_date": ["2024-01-01"], "weight_class": ["Lightweight"],
    })
    validate_required_columns(df, ["fighter_a", "fighter_b", "winner", "event_date", "weight_class"])

def test_validate_required_columns_fails():
    df = pd.DataFrame({"fighter_a": ["A"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(df, ["fighter_a", "fighter_b", "winner"])

def test_clean_fight_data_parses_dates():
    df = pd.DataFrame({
        "event_date": ["January 20, 2024", "2024-03-15"],
        "fighter_a": ["A", "B"], "fighter_b": ["C", "D"],
        "winner": ["A", "B"], "weight_class": ["Lightweight", "Welterweight"],
        "result": ["KO", "Decision"],
    })
    result = clean_fight_data(df)
    assert pd.api.types.is_datetime64_any_dtype(result["event_date"])
