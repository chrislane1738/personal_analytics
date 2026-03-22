"""Striking metric features: SLpM, SApM, accuracy, defense, differentials."""
import pandas as pd

def compute_striking_averages(fight_stats: pd.DataFrame, fighter_name: str, target_date: pd.Timestamp) -> dict:
    prior = fight_stats[
        (fight_stats["fighter"] == fighter_name) & (fight_stats["event_date"] < target_date)
    ]
    if prior.empty:
        return {"slpm": 0.0, "sapm": 0.0, "str_acc": 0.0, "str_def": 0.0}
    return {"slpm": prior["slpm"].mean(), "sapm": prior["sapm"].mean(),
            "str_acc": prior["str_acc"].mean(), "str_def": prior["str_def"].mean()}

def compute_striking_features(a_avgs: dict, b_avgs: dict) -> dict:
    features = {}
    for prefix, avgs in [("a", a_avgs), ("b", b_avgs)]:
        for key, val in avgs.items():
            features[f"{prefix}_{key}"] = val
        features[f"{prefix}_strike_differential"] = avgs["slpm"] - avgs["sapm"]
    features["slpm_diff"] = a_avgs["slpm"] - b_avgs["slpm"]
    features["sapm_diff"] = a_avgs["sapm"] - b_avgs["sapm"]
    features["str_acc_diff"] = a_avgs["str_acc"] - b_avgs["str_acc"]
    features["str_def_diff"] = a_avgs["str_def"] - b_avgs["str_def"]
    return features
