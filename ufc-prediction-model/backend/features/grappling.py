"""Grappling metric features: takedown avg, accuracy, defense, submissions."""
import pandas as pd

def compute_grappling_averages(fight_stats: pd.DataFrame, fighter_name: str, target_date: pd.Timestamp) -> dict:
    prior = fight_stats[
        (fight_stats["fighter"] == fighter_name) & (fight_stats["event_date"] < target_date)
    ]
    if prior.empty:
        return {"td_avg": 0.0, "td_acc": 0.0, "td_def": 0.0, "sub_avg": 0.0}
    return {"td_avg": prior["td_avg"].mean(), "td_acc": prior["td_acc"].mean(),
            "td_def": prior["td_def"].mean(), "sub_avg": prior["sub_avg"].mean()}

def compute_grappling_features(a_avgs: dict, b_avgs: dict) -> dict:
    features = {}
    for prefix, avgs in [("a", a_avgs), ("b", b_avgs)]:
        for key, val in avgs.items():
            features[f"{prefix}_{key}"] = val
    features["td_avg_diff"] = a_avgs["td_avg"] - b_avgs["td_avg"]
    features["td_acc_diff"] = a_avgs["td_acc"] - b_avgs["td_acc"]
    features["td_def_diff"] = a_avgs["td_def"] - b_avgs["td_def"]
    features["sub_avg_diff"] = a_avgs["sub_avg"] - b_avgs["sub_avg"]
    features["grappling_advantage"] = (a_avgs["td_avg"] * a_avgs["td_acc"] - b_avgs["td_avg"] * b_avgs["td_acc"])
    features["sub_threat_diff"] = a_avgs["sub_avg"] - b_avgs["sub_avg"]
    return features
