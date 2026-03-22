"""Feature engineering pipeline: orchestrates all feature modules into a single feature vector."""
import pandas as pd
import numpy as np
from features.physical import compute_physical_features
from features.record import compute_record_features
from features.striking import compute_striking_features
from features.grappling import compute_grappling_features
from features.rankings import compute_rankings_features
from features.context import compute_context_features
from features.odds import compute_odds_features
from features.style import compute_all_style_features

def build_feature_row(
    fighter_a_info: dict, fighter_b_info: dict,
    a_record: dict, b_record: dict,
    a_striking: dict, b_striking: dict,
    a_grappling: dict, b_grappling: dict,
    a_form: dict, b_form: dict,
    a_rank: int, b_rank: int,
    context: dict,
    a_odds: float | None = None, b_odds: float | None = None,
) -> dict:
    features = {}
    features.update(compute_physical_features(fighter_a_info, fighter_b_info))
    features.update(compute_record_features(a_record, b_record))
    features.update(compute_striking_features(a_striking, b_striking))
    features.update(compute_grappling_features(a_grappling, b_grappling))
    for prefix, form in [("a", a_form), ("b", b_form)]:
        for key, val in form.items():
            features[f"{prefix}_{key}"] = val
    features["days_since_fight_diff"] = a_form["days_since_last_fight"] - b_form["days_since_last_fight"]
    features["momentum_diff"] = a_form["last_3_wins"] - b_form["last_3_wins"]
    features.update(compute_rankings_features(a_rank, b_rank))
    features.update(context)
    features.update(compute_odds_features(a_odds, b_odds))
    a_style_stats = {**a_striking, **a_grappling, **a_record}
    b_style_stats = {**b_striking, **b_grappling, **b_record}
    a_style = compute_all_style_features(a_style_stats)
    b_style = compute_all_style_features(b_style_stats)
    for key in a_style:
        features[f"a_style_{key}"] = a_style[key]
        features[f"b_style_{key}"] = b_style[key]
        features[f"style_{key}_diff"] = a_style[key] - b_style[key]
    return features

def augment_with_swap(df: pd.DataFrame) -> pd.DataFrame:
    swapped = df.copy()
    a_cols = [c for c in df.columns if c.startswith("a_")]
    b_cols = [c for c in df.columns if c.startswith("b_")]
    for a_col in a_cols:
        b_col = "b_" + a_col[2:]
        if b_col in swapped.columns:
            swapped[a_col], swapped[b_col] = df[b_col].values, df[a_col].values
    diff_cols = [c for c in df.columns if c.endswith("_diff")]
    for col in diff_cols:
        swapped[col] = -df[col]
    if "target" in swapped.columns:
        swapped["target"] = 1 - df["target"]
    return pd.concat([df, swapped], ignore_index=True)
