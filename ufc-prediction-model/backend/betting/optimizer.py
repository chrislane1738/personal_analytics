"""Automated profile optimization: sensitivity sweep + Bayesian search."""
import logging
import time
from typing import Callable

import numpy as np
from lightgbm import LGBMClassifier

from betting.profiles import (
    ALL_GROUPS,
    apply_profile,
    prepare_backtest_context,
)

logger = logging.getLogger(__name__)

ACTIVE_GROUPS = [g for g in ALL_GROUPS if g != "odds"]
DEFAULT_MULTIPLIERS = {g: 1.0 for g in ALL_GROUPS}


def evaluate_multipliers(multipliers: dict, ctx: dict) -> dict:
    """Run one expanding-window backtest with given multipliers.
    Returns {accuracy, hc_accuracy, hc_fights, total_fights}."""
    profile = {"multipliers": multipliers, "disabled_features": []}
    df = ctx["df"]
    feature_cols = ctx["feature_cols"]
    feature_groups = ctx["feature_groups"]
    splits = ctx["splits"]
    params = dict(ctx["model_params"])  # copy to avoid mutation

    # Ensure reproducibility
    params.setdefault("random_state", 42)
    params["verbose"] = -1

    all_y_true = []
    all_y_prob = []

    for train_idx, val_idx in splits:
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_val = val_df[feature_cols]

        X_val_transformed = apply_profile(X_val, profile, feature_groups)

        fold_model = LGBMClassifier(**params)
        fold_model.fit(X_train, y_train)

        proba = fold_model.predict_proba(X_val_transformed)[:, 1]

        all_y_true.extend(val_df["target"].tolist())
        all_y_prob.extend(proba.tolist())

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = (all_y_prob >= 0.5).astype(int)

    accuracy = float((all_y_pred == all_y_true).mean())
    total_fights = len(all_y_true)

    confidence = np.maximum(all_y_prob, 1 - all_y_prob)
    hc_mask = confidence >= 0.70
    hc_fights = int(hc_mask.sum())
    hc_accuracy = float((all_y_pred[hc_mask] == all_y_true[hc_mask]).mean()) if hc_fights > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "hc_accuracy": round(hc_accuracy, 4),
        "hc_fights": hc_fights,
        "total_fights": total_fights,
    }


def run_sensitivity_sweep(
    progress_callback: Callable[[dict], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
) -> dict:
    """Phase 1: Sweep each active group's multiplier independently."""
    ctx = prepare_backtest_context("a")

    steps = [round(0.1 + i * 0.1, 1) for i in range(20)]  # 0.1 to 2.0
    total_trials = len(ACTIVE_GROUPS) * len(steps)

    curves: dict[str, list[dict]] = {g: [] for g in ACTIVE_GROUPS}
    best_per_group: dict[str, float] = {}
    best_accuracy = 0.0
    best_hc_accuracy = 0.0
    trial_num = 0
    start_time = time.time()
    cancelled = False

    for group in ACTIVE_GROUPS:
        if cancelled:
            break

        group_best_acc = 0.0
        group_best_mult = 1.0

        for mult in steps:
            if cancel_flag and cancel_flag():
                logger.info("Sweep cancelled")
                cancelled = True
                break

            trial_num += 1
            multipliers = {**DEFAULT_MULTIPLIERS, group: mult}
            result = evaluate_multipliers(multipliers, ctx)

            curves[group].append({"multiplier": mult, **result})

            if result["accuracy"] > group_best_acc:
                group_best_acc = result["accuracy"]
                group_best_mult = mult
            if result["accuracy"] > best_accuracy:
                best_accuracy = result["accuracy"]
            if result["hc_accuracy"] > best_hc_accuracy:
                best_hc_accuracy = result["hc_accuracy"]

            elapsed = time.time() - start_time
            eta = (elapsed / trial_num) * (total_trials - trial_num) if trial_num > 0 else 0

            if progress_callback:
                progress_callback({
                    "phase": "sweep",
                    "current_trial": trial_num,
                    "total_trials": total_trials,
                    "current_group": group,
                    "current_multiplier": mult,
                    "best_accuracy": best_accuracy,
                    "best_hc_accuracy": best_hc_accuracy,
                    "eta_seconds": round(eta),
                })

            logger.info(f"Sweep [{trial_num}/{total_trials}] {group}={mult:.1f}: {result['accuracy']*100:.1f}%")

        best_per_group[group] = group_best_mult

    best_combined = {**DEFAULT_MULTIPLIERS}
    best_combined.update(best_per_group)

    return {
        "curves": curves,
        "best_per_group": best_per_group,
        "best_combined": best_combined,
    }


def run_optuna_optimization(
    n_trials: int = 300,
    sweep_best: dict | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_flag: Callable[[], bool] | None = None,
) -> dict:
    """Phase 2: Bayesian optimization over active groups."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ctx = prepare_backtest_context("a")

    all_trials: list[dict] = []
    best_accuracy = 0.0
    best_hc_accuracy = 0.0
    best_params: dict = {}
    start_time = time.time()

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_accuracy, best_hc_accuracy, best_params

        if cancel_flag and cancel_flag():
            raise optuna.exceptions.OptunaError("Cancelled")

        multipliers = {g: 1.0 for g in ALL_GROUPS}
        for group in ACTIVE_GROUPS:
            multipliers[group] = trial.suggest_float(group, 0.1, 3.0)

        result = evaluate_multipliers(multipliers, ctx)

        trial_data = {
            "trial_number": trial.number,
            "multipliers": {g: round(multipliers[g], 2) for g in ACTIVE_GROUPS},
            **result,
        }
        all_trials.append(trial_data)

        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            best_hc_accuracy = result["hc_accuracy"]
            best_params = {g: round(multipliers[g], 2) for g in ACTIVE_GROUPS}

        elapsed = time.time() - start_time
        completed = len(all_trials)
        eta = (elapsed / completed) * (n_trials - completed) if completed > 0 else 0

        if progress_callback:
            progress_callback({
                "phase": "optuna",
                "current_trial": completed,
                "total_trials": n_trials,
                "best_accuracy": best_accuracy,
                "best_hc_accuracy": best_hc_accuracy,
                "best_params": best_params,
                "eta_seconds": round(eta),
            })

        logger.info(f"Optuna [{completed}/{n_trials}] acc={result['accuracy']*100:.1f}% best={best_accuracy*100:.1f}%")
        return result["accuracy"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=30),
    )

    if sweep_best:
        study.enqueue_trial({g: sweep_best.get(g, 1.0) for g in ACTIVE_GROUPS})
    study.enqueue_trial({g: 1.0 for g in ACTIVE_GROUPS})

    try:
        study.optimize(objective, n_trials=n_trials)
    except optuna.exceptions.OptunaError:
        logger.info("Optuna optimization cancelled")

    param_importance = {}
    try:
        importance = optuna.importance.get_param_importances(study)
        param_importance = {k: round(v * 100, 1) for k, v in importance.items()}
    except Exception:
        logger.warning("Could not compute parameter importance")

    return {
        "best_params": best_params,
        "best_accuracy": best_accuracy,
        "best_hc_accuracy": best_hc_accuracy,
        "trials": all_trials,
        "param_importance": param_importance,
    }
