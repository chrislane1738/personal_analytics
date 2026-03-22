"""Build the full feature-engineered dataset from raw fight data."""
import logging
import pandas as pd
from pathlib import Path
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DEBUT_FIGHT_THRESHOLD

logger = logging.getLogger(__name__)

STYLE_KEYS = ["striker", "wrestler", "grappler", "balanced"]
STAT_KEYS = ["slpm", "sapm", "str_acc", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]


def compute_weight_class_medians(df: pd.DataFrame) -> dict:
    medians = {}
    for wc in df["weight_class"].unique():
        wc_df = df[df["weight_class"] == wc]
        medians[wc] = {}
        for stat in STAT_KEYS:
            col = f"a_{stat}"
            if col in wc_df.columns:
                medians[wc][stat] = float(wc_df[col].median())
    return medians


def impute_cold_start(features: dict, medians: dict) -> dict:
    features = features.copy()
    wc = features.get("weight_class", "")
    for prefix in ["a", "b"]:
        if features.get(f"{prefix}_is_debut") or features.get(f"{prefix}_is_near_debut"):
            wc_meds = medians.get(wc, {})
            for stat in STAT_KEYS:
                key = f"{prefix}_{stat}"
                if features.get(key, 0) == 0.0 and stat in wc_meds:
                    features[key] = wc_meds[stat]
            for style in STYLE_KEYS:
                key = f"{prefix}_style_{style}"
                if features.get(key, 0) == 0.0:
                    features[key] = 0.25
    return features
