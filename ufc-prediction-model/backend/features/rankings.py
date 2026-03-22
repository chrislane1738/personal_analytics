"""UFC rankings features: rank at fight time, rank differentials."""
import pandas as pd

def get_ranking_at_date(rankings_df: pd.DataFrame, fighter_name: str, weight_class: str, target_date: pd.Timestamp) -> int:
    if rankings_df.empty:
        return 0
    mask = ((rankings_df["fighter"] == fighter_name) & (rankings_df["weight_class"] == weight_class) & (rankings_df["date"] < target_date))
    relevant = rankings_df[mask].sort_values("date")
    if relevant.empty:
        return 0
    return int(relevant["rank"].iloc[-1])

def compute_rankings_features(a_rank: int, b_rank: int) -> dict:
    return {
        "a_rank": a_rank, "b_rank": b_rank,
        "a_is_ranked": a_rank > 0, "b_is_ranked": b_rank > 0,
        "rank_diff": a_rank - b_rank if (a_rank > 0 and b_rank > 0) else (
            -b_rank if (a_rank == 0 and b_rank > 0) else (a_rank if (b_rank == 0 and a_rank > 0) else 0)),
        "a_stepping_up": a_rank == 0 and b_rank > 0,
        "b_stepping_up": b_rank == 0 and a_rank > 0,
    }
