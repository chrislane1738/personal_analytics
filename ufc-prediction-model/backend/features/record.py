"""Career record features: wins, losses, streaks, finish rates."""
import pandas as pd

def compute_career_stats_at_date(fights_df: pd.DataFrame, fighter_name: str, target_date: pd.Timestamp) -> dict:
    """Compute cumulative career stats using only fights BEFORE target_date."""
    prior = fights_df[
        (fights_df["fighter"] == fighter_name) & (fights_df["event_date"] < target_date)
    ].sort_values("event_date")

    if prior.empty:
        return _empty_stats()

    wins = (prior["result"] == "Win").sum()
    losses = (prior["result"] == "Loss").sum()
    draws = (prior["result"] == "Draw").sum()
    total = len(prior)

    win_streak = 0
    loss_streak = 0
    for result in prior["result"].iloc[::-1]:
        if result == "Win":
            if loss_streak == 0:
                win_streak += 1
            else:
                break
        elif result == "Loss":
            if win_streak == 0:
                loss_streak += 1
            else:
                break
        else:
            break

    win_fights = prior[prior["result"] == "Win"]
    ko_wins = win_fights["method"].str.contains("KO|TKO", case=False, na=False).sum()
    sub_wins = win_fights["method"].str.contains("Sub", case=False, na=False).sum()

    # Career knockdown rate (spec §3.1 Rule 4)
    kd_scored = prior["knockdowns_scored"].sum() if "knockdowns_scored" in prior.columns else 0
    kd_rate = kd_scored / total if total > 0 else 0.0

    return {
        "wins": int(wins), "losses": int(losses), "draws": int(draws),
        "ufc_fights": int(total),
        "win_rate": wins / total if total > 0 else 0.0,
        "win_streak": int(win_streak), "loss_streak": int(loss_streak),
        "ko_win_pct": ko_wins / wins if wins > 0 else 0.0,
        "sub_win_pct": sub_wins / wins if wins > 0 else 0.0,
        "dec_win_pct": (wins - ko_wins - sub_wins) / wins if wins > 0 else 0.0,
        "finish_rate": (ko_wins + sub_wins) / wins if wins > 0 else 0.0,
        "kd_rate": kd_rate,
    }

def _empty_stats() -> dict:
    return {"wins": 0, "losses": 0, "draws": 0, "ufc_fights": 0,
            "win_rate": 0.0, "win_streak": 0, "loss_streak": 0,
            "ko_win_pct": 0.0, "sub_win_pct": 0.0, "dec_win_pct": 0.0,
            "finish_rate": 0.0, "kd_rate": 0.0}

def compute_record_features(a_stats: dict, b_stats: dict) -> dict:
    features = {}
    for prefix, stats in [("a", a_stats), ("b", b_stats)]:
        for key, val in stats.items():
            features[f"{prefix}_{key}"] = val
    features["win_rate_diff"] = a_stats.get("win_rate", 0) - b_stats.get("win_rate", 0)
    features["finish_rate_diff"] = a_stats.get("finish_rate", 0) - b_stats.get("finish_rate", 0)
    features["experience_diff"] = a_stats.get("ufc_fights", 0) - b_stats.get("ufc_fights", 0)
    features["streak_diff"] = a_stats.get("win_streak", 0) - b_stats.get("win_streak", 0)
    return features
