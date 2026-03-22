"""Career record features: wins, losses, streaks, finish rates, experience quality."""
import math
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

    # Five-round fight experience (championship rounds)
    five_round_fights = int(
        prior["rounds_scheduled"].eq(5).sum()
    ) if "rounds_scheduled" in prior.columns else 0

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
        "five_round_fights": five_round_fights,
    }

def _empty_stats() -> dict:
    return {"wins": 0, "losses": 0, "draws": 0, "ufc_fights": 0,
            "win_rate": 0.0, "win_streak": 0, "loss_streak": 0,
            "ko_win_pct": 0.0, "sub_win_pct": 0.0, "dec_win_pct": 0.0,
            "finish_rate": 0.0, "kd_rate": 0.0, "five_round_fights": 0}


def compute_experience_features(
    fights_df: pd.DataFrame, fighter_name: str, target_date: pd.Timestamp,
    all_fighter_records: dict[str, dict] | None = None,
) -> dict:
    """Compute experience quality features: strength of schedule, championship experience.

    Addresses the problem where a new fighter with great ratios against weak
    opponents is overvalued vs an experienced fighter with worse ratios earned
    against elite competition.
    """
    prior = fights_df[
        (fights_df["fighter"] == fighter_name) & (fights_df["event_date"] < target_date)
    ].sort_values("event_date")

    total = len(prior)
    if total == 0:
        return {
            "avg_opponent_win_rate": 0.0,
            "ranked_opponent_pct": 0.0,
            "experience_score": 0.0,
        }

    # Average opponent win rate (strength of schedule)
    avg_opp_wr = 0.0
    if all_fighter_records and "opponent" in prior.columns:
        opp_win_rates = []
        for _, row in prior.iterrows():
            opp = row["opponent"]
            opp_record = all_fighter_records.get(opp, {})
            opp_wr = opp_record.get("win_rate", 0.5)  # Default to 0.5 if unknown
            opp_win_rates.append(opp_wr)
        avg_opp_wr = sum(opp_win_rates) / len(opp_win_rates) if opp_win_rates else 0.0

    # Percentage of fights against ranked opponents
    ranked_opp_pct = 0.0
    if "opponent_ranked" in prior.columns:
        ranked_opp_pct = prior["opponent_ranked"].sum() / total

    # Composite experience score: log(fights) * avg_opponent_quality
    # log(fights) grows quickly at first then tapers — captures diminishing returns
    # Multiplied by opponent quality so experience against cans counts less
    experience_score = math.log1p(total) * max(avg_opp_wr, 0.3)  # Floor at 0.3 to avoid zeroing out

    return {
        "avg_opponent_win_rate": avg_opp_wr,
        "ranked_opponent_pct": ranked_opp_pct,
        "experience_score": experience_score,
    }

def compute_record_features(
    a_stats: dict, b_stats: dict,
    a_exp: dict | None = None, b_exp: dict | None = None,
) -> dict:
    """Compute record features + experience quality differentials."""
    features = {}
    for prefix, stats in [("a", a_stats), ("b", b_stats)]:
        for key, val in stats.items():
            features[f"{prefix}_{key}"] = val
    features["win_rate_diff"] = a_stats.get("win_rate", 0) - b_stats.get("win_rate", 0)
    features["finish_rate_diff"] = a_stats.get("finish_rate", 0) - b_stats.get("finish_rate", 0)
    features["experience_diff"] = a_stats.get("ufc_fights", 0) - b_stats.get("ufc_fights", 0)
    features["streak_diff"] = a_stats.get("win_streak", 0) - b_stats.get("win_streak", 0)
    features["five_round_diff"] = a_stats.get("five_round_fights", 0) - b_stats.get("five_round_fights", 0)

    # Experience quality features
    if a_exp and b_exp:
        for prefix, exp in [("a", a_exp), ("b", b_exp)]:
            for key, val in exp.items():
                features[f"{prefix}_{key}"] = val
        features["avg_opp_wr_diff"] = a_exp.get("avg_opponent_win_rate", 0) - b_exp.get("avg_opponent_win_rate", 0)
        features["experience_score_diff"] = a_exp.get("experience_score", 0) - b_exp.get("experience_score", 0)
        features["ranked_opp_pct_diff"] = a_exp.get("ranked_opponent_pct", 0) - b_exp.get("ranked_opponent_pct", 0)

    return features
