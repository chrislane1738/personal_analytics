"""Model evaluation: metrics, cross-validation, calibration."""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "n_samples": len(y_true),
    }

def expanding_window_cv_splits(years: list[int], start_year: int = 2020) -> list[tuple[list[int], list[int]]]:
    unique_years = sorted(set(y for y in years if y >= start_year))
    splits = []
    for val_year in unique_years:
        train_idx = [i for i, y in enumerate(years) if y < val_year]
        val_idx = [i for i, y in enumerate(years) if y == val_year]
        if train_idx and val_idx:
            splits.append((train_idx, val_idx))
    return splits
