"""Tests for full dataset build with cold-start imputation."""
import pytest
import pandas as pd
from features.build_dataset import impute_cold_start


def test_impute_cold_start_uses_weight_class_medians():
    features = {"a_slpm": 0.0, "a_td_avg": 0.0, "a_is_debut": True, "weight_class": "Lightweight"}
    medians = {"Lightweight": {"slpm": 4.5, "td_avg": 2.0}}
    result = impute_cold_start(features, medians)
    assert result["a_slpm"] == 4.5
    assert result["a_td_avg"] == 2.0


def test_impute_cold_start_sets_balanced_style():
    features = {"a_style_striker": 0.0, "a_style_wrestler": 0.0,
                "a_style_grappler": 0.0, "a_style_balanced": 0.0,
                "a_is_debut": True, "weight_class": "Flyweight"}
    medians = {"Flyweight": {}}
    result = impute_cold_start(features, medians)
    assert result["a_style_striker"] == 0.25
    assert result["a_style_wrestler"] == 0.25


def test_non_debut_not_imputed():
    features = {"a_slpm": 5.0, "a_is_debut": False, "weight_class": "Lightweight"}
    medians = {"Lightweight": {"slpm": 4.5}}
    result = impute_cold_start(features, medians)
    assert result["a_slpm"] == 5.0  # Unchanged
