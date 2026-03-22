"""Tests for physical attribute feature computation."""
import pytest
from features.physical import compute_physical_features

def test_basic_differentials():
    fighter_a = {"height_cm": 185, "reach_cm": 193, "age": 30, "stance": "Orthodox"}
    fighter_b = {"height_cm": 178, "reach_cm": 185, "age": 34, "stance": "Southpaw"}
    result = compute_physical_features(fighter_a, fighter_b)
    assert result["height_diff"] == 7
    assert result["reach_diff"] == 8
    assert result["age_diff"] == -4
    assert result["a_stance"] == "Orthodox"
    assert result["b_stance"] == "Southpaw"

def test_missing_reach_handled():
    fighter_a = {"height_cm": 185, "reach_cm": None, "age": 30, "stance": "Orthodox"}
    fighter_b = {"height_cm": 178, "reach_cm": 185, "age": 34, "stance": "Orthodox"}
    result = compute_physical_features(fighter_a, fighter_b)
    assert result["reach_diff"] == 0
