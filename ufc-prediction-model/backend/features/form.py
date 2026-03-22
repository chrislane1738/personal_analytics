"""Recent form features: days since fight, recent results, momentum."""
import pandas as pd
from config import DEBUT_FIGHT_THRESHOLD

def compute_form_features(fights_df: pd.DataFrame, fighter_name: str, target_date: pd.Timestamp) -> dict:
    prior = fights_df[
        (fights_df["fighter"] == fighter_name) & (fights_df["event_date"] < target_date)
    ].sort_values("event_date")
    total_fights = len(prior)
    is_debut = total_fights == 0
    is_near_debut = total_fights < DEBUT_FIGHT_THRESHOLD
    if is_debut:
        return {"last_3_wins": 0, "last_3_losses": 0, "last_5_wins": 0, "last_5_losses": 0,
                "days_since_last_fight": 365, "recent_finish_rate": 0.0,
                "is_debut": True, "is_near_debut": True}
    last_3 = prior.tail(3)
    last_5 = prior.tail(5)
    last_fight_date = prior["event_date"].iloc[-1]
    return {
        "last_3_wins": int((last_3["result"] == "Win").sum()),
        "last_3_losses": int((last_3["result"] == "Loss").sum()),
        "last_5_wins": int((last_5["result"] == "Win").sum()),
        "last_5_losses": int((last_5["result"] == "Loss").sum()),
        "days_since_last_fight": (target_date - last_fight_date).days,
        "recent_finish_rate": (last_5["method"].str.contains("KO|TKO|Sub", case=False, na=False).sum() / len(last_5)
                               if "method" in last_5.columns and len(last_5) > 0 else 0.0),
        "is_debut": is_debut, "is_near_debut": is_near_debut,
    }
