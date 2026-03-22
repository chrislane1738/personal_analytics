"""Tests for model training."""
import pytest
import pandas as pd
import numpy as np
from models.train import train_model, get_feature_columns

def test_get_feature_columns_excludes_target():
    df = pd.DataFrame({"feat_1": [1], "feat_2": [2], "target": [1], "event_date": ["2024-01-01"]})
    cols = get_feature_columns(df, include_odds=False)
    assert "target" not in cols
    assert "event_date" not in cols
    assert "feat_1" in cols

def test_get_feature_columns_excludes_odds_for_model_a():
    df = pd.DataFrame({"feat_1": [1], "a_implied_prob": [0.6], "odds_diff": [0.1], "target": [1]})
    cols = get_feature_columns(df, include_odds=False)
    assert "a_implied_prob" not in cols
    assert "odds_diff" not in cols

def test_train_model_returns_fitted():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({"feat_1": np.random.randn(n), "feat_2": np.random.randn(n), "target": np.random.randint(0, 2, n)})
    model = train_model(df, ["feat_1", "feat_2"])
    preds = model.predict_proba(df[["feat_1", "feat_2"]])
    assert preds.shape == (n, 2)
