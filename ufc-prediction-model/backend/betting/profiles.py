"""Model profiles: feature group weighting and backtesting for betting strategies."""
import json
import re
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from config import MODELS_DIR, PROCESSED_DATA_DIR, CV_START_YEAR, DATA_DIR
from models.evaluate import expanding_window_cv_splits

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXED_FEATURES = {
    "a_stance", "b_stance", "weight_class", "rounds_scheduled",
    "is_title_fight", "is_five_rounder", "a_is_debut", "b_is_debut",
    "a_is_near_debut", "b_is_near_debut", "a_is_ranked", "b_is_ranked",
    "a_stepping_up", "b_stepping_up",
}

ALL_GROUPS = [
    "striking", "grappling", "record", "physical", "form",
    "rankings", "experience_quality", "style", "odds",
]

# Substring patterns for assigning features to groups.
# Order matters: patterns are checked top-down; first match wins.
# More specific patterns must come before less specific ones.
_GROUP_PATTERNS: list[tuple[str, list[str]]] = [
    # Order matters: more specific patterns must come before less specific ones.
    # e.g. "experience_quality" must precede "record" (opponent_win_rate contains win_rate)
    # and "rankings" (ranked_opponent_pct contains rank).
    # "form" must precede "record" (last_3_wins contains wins, recent_finish contains finish).
    ("striking", ["slpm", "str_acc", "str_def", "sapm", "strike_differential"]),
    ("grappling", ["td_avg", "td_acc", "td_def", "sub_avg", "grappling_advantage", "sub_threat"]),
    ("experience_quality", [
        "opponent_win_rate", "ranked_opponent_pct", "experience_score",
        "avg_opp_wr", "ranked_opp_pct",
    ]),
    ("form", ["last_3", "last_5", "days_since", "recent_finish", "momentum"]),
    ("record", [
        "wins", "losses", "draws", "ufc_fights", "win_rate", "finish_rate",
        "ko_win_pct", "sub_win_pct", "dec_win_pct", "win_streak", "loss_streak",
        "kd_rate", "five_round",
    ]),
    ("physical", ["height", "reach", "age"]),
    ("rankings", ["rank"]),
    ("style", ["style_"]),
    ("odds", ["implied_prob", "odds_diff", "market_confidence"]),
]

EXCLUDE_COLUMNS = {"target", "event_date", "fighter_a", "fighter_b", "winner", "event_name"}
ODDS_COLUMNS = {"a_implied_prob", "b_implied_prob", "odds_diff", "market_confidence"}

# Columns dropped during dataset preparation (constant / near-zero-variance)
_DROP_COLUMNS = {"sapm_diff", "str_def_diff", "td_def_diff", "a_sapm", "b_sapm"}

PROFILES_PATH = DATA_DIR / "profiles.json"

# ---------------------------------------------------------------------------
# Feature Groups
# ---------------------------------------------------------------------------

def build_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """Assign each feature column to a named group using substring patterns.

    Features in FIXED_FEATURES are skipped entirely.
    Features not matching any pattern are left ungrouped (pass-through).
    """
    groups: dict[str, list[str]] = {g: [] for g in ALL_GROUPS}

    for col in feature_cols:
        if col in FIXED_FEATURES:
            continue

        matched = False
        for group_name, patterns in _GROUP_PATTERNS:
            for pattern in patterns:
                if pattern in col:
                    groups[group_name].append(col)
                    matched = True
                    break
            if matched:
                break
        # Features not matched by any pattern are left ungrouped and pass through
        # untransformed during apply_profile.

    return groups


# ---------------------------------------------------------------------------
# Profile CRUD
# ---------------------------------------------------------------------------

def _default_profile() -> dict:
    return {
        "multipliers": {g: 1.0 for g in ALL_GROUPS},
        "disabled_features": [],
        "custom_adjustments": {"ov_ev_bonus": 0.0},
        "created": datetime.utcnow().isoformat(),
        "last_backtest": None,
    }


def load_profiles() -> dict:
    """Load profiles from disk. Creates default file if missing."""
    if not PROFILES_PATH.exists():
        PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        profiles = {"Default": _default_profile()}
        save_profiles(profiles)
        return profiles

    with open(PROFILES_PATH, "r") as f:
        return json.load(f)


def save_profiles(profiles: dict) -> None:
    """Persist profiles to disk."""
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=2)


# ---------------------------------------------------------------------------
# Name-based adjustments
# ---------------------------------------------------------------------------

_OV_EV_RE = re.compile(r"(ov|ev|ova|eva)$", re.IGNORECASE)


def has_ov_ev_name(name: str) -> bool:
    """Check if a fighter's last name ends in ov/ev/ova/eva."""
    last = name.strip().split()[-1] if name.strip() else ""
    return bool(_OV_EV_RE.search(last))


def apply_name_adjustments(
    proba: np.ndarray,
    fighter_a_names: list[str],
    fighter_b_names: list[str],
    adjustments: dict,
) -> np.ndarray:
    """Apply post-prediction probability adjustments based on fighter names.

    Returns adjusted probabilities clipped to [0.01, 0.99].
    """
    adjusted = proba.copy()
    ov_ev_bonus = adjustments.get("ov_ev_bonus", 0.0)

    if ov_ev_bonus != 0.0:
        for i, (a_name, b_name) in enumerate(zip(fighter_a_names, fighter_b_names)):
            a_has = has_ov_ev_name(a_name)
            b_has = has_ov_ev_name(b_name)
            if a_has and not b_has:
                adjusted[i] += ov_ev_bonus
            elif b_has and not a_has:
                adjusted[i] -= ov_ev_bonus

    return np.clip(adjusted, 0.01, 0.99)


# ---------------------------------------------------------------------------
# Profile Application
# ---------------------------------------------------------------------------

def apply_profile(df: pd.DataFrame, profile: dict, feature_groups: dict[str, list[str]]) -> pd.DataFrame:
    """Return a copy of *df* with feature values transformed per *profile*.

    1. Fixed features are left unchanged.
    2. Disabled features are zeroed out.
    3. Grouped features are multiplied by their group's multiplier.
    4. Ungrouped features pass through unchanged.
    """
    out = df.copy()
    disabled = set(profile.get("disabled_features", []))
    multipliers = profile.get("multipliers", {})

    # Build a reverse mapping: feature -> group
    feat_to_group: dict[str, str] = {}
    for group_name, feats in feature_groups.items():
        for feat in feats:
            feat_to_group[feat] = group_name

    for col in out.columns:
        if col in FIXED_FEATURES:
            continue
        if col in disabled:
            out[col] = 0
            continue
        group = feat_to_group.get(col)
        if group is not None:
            mult = multipliers.get(group, 1.0)
            if mult != 1.0:
                out[col] = out[col] * mult

    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"^[A-Za-z0-9 \-]{1,50}$")


def validate_profile(profile: dict, feature_cols: list[str]) -> list[str]:
    """Return a list of error messages. Empty list means valid."""
    errors: list[str] = []

    # Name validation (if present)
    name = profile.get("name")
    if name is not None:
        if name == "Default":
            errors.append("Cannot use reserved name 'Default'")
        elif not _NAME_RE.match(name):
            errors.append("Name must be 1-50 chars, alphanumeric/spaces/hyphens only")

    # Multiplier validation
    multipliers = profile.get("multipliers", {})
    for group in ALL_GROUPS:
        val = multipliers.get(group)
        if val is not None:
            if not (0.1 <= val <= 3.0):
                errors.append(f"Multiplier for '{group}' must be between 0.1 and 3.0 (got {val})")

    # Disabled features validation
    feature_set = set(feature_cols)
    for feat in profile.get("disabled_features", []):
        if feat not in feature_set:
            errors.append(f"Disabled feature '{feat}' not found in model feature columns")
        if feat in FIXED_FEATURES:
            errors.append(f"Cannot disable fixed feature '{feat}'")

    return errors


# ---------------------------------------------------------------------------
# Profile Backtesting
# ---------------------------------------------------------------------------

def _find_latest_model(variant: str = "a") -> Path:
    """Return path to the most recent model joblib for the given variant."""
    pattern = f"model_{variant}_*.joblib"
    candidates = sorted(MODELS_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No model files matching {pattern} in {MODELS_DIR}")
    return candidates[-1]


def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the training dataset for backtesting."""
    # Drop constant / near-zero-variance columns
    cols_to_drop = [c for c in _DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    # Fill NaN with 0
    df = df.fillna(0)

    return df


def prepare_backtest_context(variant: str = "a") -> dict:
    """Load and prepare everything needed for backtest evaluation.
    Returns a context dict reusable across many evaluations."""
    model_path = _find_latest_model(variant)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    params = model.get_params()

    df = pd.read_csv(PROCESSED_DATA_DIR / "training_dataset.csv")
    df = _prepare_dataset(df)

    if variant == "a":
        feature_cols = [c for c in feature_cols if c not in ODDS_COLUMNS]

    feature_groups = build_feature_groups(feature_cols)

    df["event_date"] = pd.to_datetime(df["event_date"])
    years = df["event_date"].dt.year.tolist()
    splits = expanding_window_cv_splits(years, start_year=CV_START_YEAR)

    return {
        "model_params": params,
        "feature_cols": feature_cols,
        "feature_groups": feature_groups,
        "df": df,
        "splits": splits,
    }


def run_profile_backtest(profile: dict, variant: str = "a") -> dict:
    """Run expanding-window cross-validation with profile-transformed features.

    Retrains a fresh LightGBM model per fold to avoid data leakage.
    Returns accuracy metrics including per-year breakdown, confidence tiers,
    and feature importance.
    """
    ctx = prepare_backtest_context(variant)
    df = ctx["df"]
    feature_cols = ctx["feature_cols"]
    feature_groups = ctx["feature_groups"]
    splits = ctx["splits"]
    params = ctx["model_params"]
    logger.info(f"Loaded model config with {len(feature_cols)} features")
    logger.info(f"Created {len(splits)} CV folds starting from {CV_START_YEAR}")

    # 6. Run cross-validation: retrain per fold to avoid data leakage
    all_y_true = []
    all_y_prob = []
    all_years = []
    fold_model = None  # will hold the last fold's trained model

    custom_adjustments = profile.get("custom_adjustments", {})

    for train_idx, val_idx in splits:
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_val = val_df[feature_cols]
        y_val = val_df["target"]

        # Apply profile transformation to validation features only
        X_val_transformed = apply_profile(X_val, profile, feature_groups)

        # Train a fresh model with the same hyperparameters
        fold_model = LGBMClassifier(**params)
        fold_model.fit(X_train, y_train)

        proba = fold_model.predict_proba(X_val_transformed)[:, 1]

        # Apply name-based adjustments
        if any(v != 0.0 for v in custom_adjustments.values()):
            proba = apply_name_adjustments(
                proba,
                val_df["fighter_a"].tolist(),
                val_df["fighter_b"].tolist(),
                custom_adjustments,
            )

        all_y_true.extend(y_val.tolist())
        all_y_prob.extend(proba.tolist())
        all_years.extend(val_df["event_date"].dt.year.tolist())

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_years = np.array(all_years)
    all_y_pred = (all_y_prob >= 0.5).astype(int)

    # 7. Compute overall accuracy
    overall_accuracy = float((all_y_pred == all_y_true).mean())
    total_fights = len(all_y_true)

    # 8. Accuracy by year
    accuracy_by_year = {}
    for year in sorted(set(all_years)):
        mask = all_years == year
        year_acc = float((all_y_pred[mask] == all_y_true[mask]).mean())
        accuracy_by_year[str(int(year))] = round(year_acc, 4)

    # 9. Confidence-accuracy tiers
    thresholds = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]
    confidence_accuracy = []
    for thresh in thresholds:
        # Confidence = max(prob, 1-prob), i.e. how confident the model is
        confidence = np.maximum(all_y_prob, 1 - all_y_prob)
        mask = confidence >= thresh
        if mask.sum() > 0:
            tier_acc = float((all_y_pred[mask] == all_y_true[mask]).mean())
            confidence_accuracy.append({
                "threshold": thresh,
                "accuracy": round(tier_acc, 4),
                "fights": int(mask.sum()),
            })

    # 10. Feature importance from last fold's model
    feature_importance = {}
    grouped_importance = {}
    if fold_model is not None:
        importance = fold_model.feature_importances_
        feature_importance = {col: float(imp) for col, imp in zip(feature_cols, importance)}

        # Group importances
        grouped_importance = {}
        for group_name, group_features in feature_groups.items():
            group_imp = sum(feature_importance.get(f, 0) for f in group_features)
            grouped_importance[group_name] = round(group_imp, 2)

        # Normalize to percentages
        total_imp = sum(grouped_importance.values()) or 1
        grouped_importance = {k: round(v / total_imp * 100, 1) for k, v in grouped_importance.items()}

    logger.info(f"Backtest complete: {overall_accuracy:.4f} accuracy on {total_fights} fights")

    return {
        "accuracy": round(overall_accuracy, 4),
        "total_fights": total_fights,
        "accuracy_by_year": accuracy_by_year,
        "confidence_accuracy": confidence_accuracy,
        "feature_importance": feature_importance,
        "grouped_importance": grouped_importance,
    }
