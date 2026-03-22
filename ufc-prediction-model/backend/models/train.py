"""LightGBM model training for both model variants."""
import logging
from pathlib import Path
from datetime import date
import lightgbm as lgb
import joblib
import pandas as pd
import numpy as np
from config import MODELS_DIR, RANDOM_SEED, CV_START_YEAR

logger = logging.getLogger(__name__)

ODDS_COLUMNS = {"a_implied_prob", "b_implied_prob", "odds_diff", "market_confidence"}
EXCLUDE_COLUMNS = {"target", "event_date", "fighter_a", "fighter_b", "winner", "event_name"}

def get_feature_columns(df: pd.DataFrame, include_odds: bool = True) -> list[str]:
    excluded = EXCLUDE_COLUMNS.copy()
    if not include_odds:
        excluded.update(ODDS_COLUMNS)
    return [c for c in df.columns if c not in excluded]

def train_model(df: pd.DataFrame, feature_cols: list[str], params: dict | None = None) -> lgb.LGBMClassifier:
    if params is None:
        params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "num_leaves": 31, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 5, "reg_alpha": 0.1, "reg_lambda": 0.1,
            "random_state": RANDOM_SEED, "verbose": -1,
        }
    model = lgb.LGBMClassifier(**params)
    X = df[feature_cols]
    y = df["target"]
    model.fit(X, y)
    logger.info(f"Model trained on {len(X)} samples with {len(feature_cols)} features")
    return model

def save_model(model: lgb.LGBMClassifier, variant: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"model_{variant}_{date.today().isoformat()}.joblib"
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path

def train_both_variants(df: pd.DataFrame, tune: bool = False) -> tuple:
    cols_a = get_feature_columns(df, include_odds=False)
    cols_b = get_feature_columns(df, include_odds=True)
    params = None
    if tune:
        logger.info("Running Optuna hyperparameter tuning...")
        params = tune_hyperparameters(df, cols_a)
    logger.info(f"Training Model A (no odds) with {len(cols_a)} features...")
    model_a = train_model(df, cols_a, params=params)
    path_a = save_model(model_a, "a_no_odds")
    logger.info(f"Training Model B (with odds) with {len(cols_b)} features...")
    model_b = train_model(df, cols_b, params=params)
    path_b = save_model(model_b, "b_with_odds")
    return (model_a, path_a), (model_b, path_b)

def tune_hyperparameters(df: pd.DataFrame, feature_cols: list[str], n_trials: int = 50) -> dict:
    import optuna
    from models.evaluate import expanding_window_cv_splits
    years = df["event_date"].dt.year.tolist() if "event_date" in df.columns else [2020] * len(df)
    splits = expanding_window_cv_splits(years, start_year=CV_START_YEAR)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_SEED, "verbose": -1,
        }
        accuracies = []
        for train_idx, val_idx in splits:
            model = lgb.LGBMClassifier(**params)
            model.fit(df.iloc[train_idx][feature_cols], df.iloc[train_idx]["target"])
            preds = model.predict(df.iloc[val_idx][feature_cols])
            accuracies.append((preds == df.iloc[val_idx]["target"]).mean())
        return np.mean(accuracies)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params
