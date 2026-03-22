"""Load, clean, and merge Kaggle UFC datasets."""
import logging
from pathlib import Path
import pandas as pd
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

REQUIRED_FIGHT_COLUMNS = ["fighter_a", "fighter_b", "winner", "event_date", "weight_class"]
NON_BINARY_RESULTS = {"Draw", "No Contest", "NC", "DQ", "Disqualification", "No contest"}
NON_BINARY_WINNERS = {"Draw", "NC", "No Contest", ""}

def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def exclude_non_binary_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    mask_result = ~df["result"].isin(NON_BINARY_RESULTS) if "result" in df.columns else True
    mask_winner = ~df["winner"].isin(NON_BINARY_WINNERS) if "winner" in df.columns else True
    filtered = df[mask_result & mask_winner].copy()
    removed = len(df) - len(filtered)
    if removed > 0:
        logger.info(f"Excluded {removed} non-binary outcome fights (draws/NC/DQ)")
    return filtered

def clean_fight_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], format="mixed", dayfirst=False)
    df = df.dropna(subset=["fighter_a", "fighter_b", "winner"])
    style_cols = [c for c in df.columns if "membership" in c.lower() or "style_score" in c.lower()]
    if style_cols:
        logger.info(f"Dropping pre-computed style columns (Rule 5): {style_cols}")
        df = df.drop(columns=style_cols)
    df = df.sort_values("event_date").reset_index(drop=True)
    return df

def load_fight_data(filename: str = "ufc_fights.csv") -> pd.DataFrame:
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Fight data not found at {filepath}.")
    df = pd.read_csv(filepath)
    validate_required_columns(df, REQUIRED_FIGHT_COLUMNS)
    df = clean_fight_data(df)
    df = exclude_non_binary_outcomes(df)
    return df

def load_rankings_data(filename: str = "ufc_rankings.csv") -> pd.DataFrame:
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        logger.warning(f"Rankings data not found at {filepath}.")
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df
