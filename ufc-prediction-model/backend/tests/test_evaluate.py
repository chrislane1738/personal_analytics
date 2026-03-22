"""Tests for model evaluation."""
import pytest
import numpy as np
from models.evaluate import compute_metrics, expanding_window_cv_splits

def test_compute_metrics():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    y_prob = np.array([0.8, 0.3, 0.7, 0.4, 0.2])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 0.8
    assert "auc_roc" in metrics
    assert "log_loss" in metrics

def test_expanding_window_splits():
    years = [2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022]
    splits = expanding_window_cv_splits(years, start_year=2020)
    assert len(splits) == 3
    train_idx, val_idx = splits[0]
    assert all(years[i] < 2020 for i in train_idx)
    assert all(years[i] == 2020 for i in val_idx)
