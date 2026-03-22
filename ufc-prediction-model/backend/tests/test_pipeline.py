"""Tests for the feature engineering pipeline orchestrator."""
import pytest
import pandas as pd
from features.pipeline import build_feature_row, augment_with_swap

def test_augment_with_swap():
    """Data augmentation: each fight produces two rows with swapped fighters."""
    df = pd.DataFrame({
        "a_slpm": [5.0], "b_slpm": [3.0],
        "slpm_diff": [2.0], "target": [1],
    })
    result = augment_with_swap(df)
    assert len(result) == 2
    assert result.iloc[1]["a_slpm"] == 3.0
    assert result.iloc[1]["b_slpm"] == 5.0
    assert result.iloc[1]["slpm_diff"] == -2.0
    assert result.iloc[1]["target"] == 0

def test_build_feature_row_returns_dict():
    fighter_a = {"height_cm": 185, "reach_cm": 193, "age": 30, "stance": "Orthodox"}
    fighter_b = {"height_cm": 178, "reach_cm": 185, "age": 34, "stance": "Southpaw"}
    row = build_feature_row(
        fighter_a_info=fighter_a, fighter_b_info=fighter_b,
        a_record={"wins": 10, "losses": 2, "win_rate": 0.83, "finish_rate": 0.5, "ufc_fights": 12,
                   "win_streak": 3, "loss_streak": 0, "ko_win_pct": 0.3, "sub_win_pct": 0.1, "dec_win_pct": 0.6, "kd_rate": 0.1, "draws": 0},
        b_record={"wins": 8, "losses": 4, "win_rate": 0.67, "finish_rate": 0.4, "ufc_fights": 12,
                   "win_streak": 1, "loss_streak": 0, "ko_win_pct": 0.2, "sub_win_pct": 0.2, "dec_win_pct": 0.6, "kd_rate": 0.05, "draws": 0},
        a_striking={"slpm": 5.0, "sapm": 3.0, "str_acc": 0.50, "str_def": 0.60},
        b_striking={"slpm": 4.0, "sapm": 4.0, "str_acc": 0.45, "str_def": 0.55},
        a_grappling={"td_avg": 2.0, "td_acc": 0.40, "td_def": 0.70, "sub_avg": 0.5},
        b_grappling={"td_avg": 1.0, "td_acc": 0.35, "td_def": 0.65, "sub_avg": 0.3},
        a_form={"last_3_wins": 3, "last_3_losses": 0, "last_5_wins": 4, "last_5_losses": 1,
                "days_since_last_fight": 90, "recent_finish_rate": 0.4, "is_debut": False, "is_near_debut": False},
        b_form={"last_3_wins": 1, "last_3_losses": 2, "last_5_wins": 2, "last_5_losses": 3,
                "days_since_last_fight": 180, "recent_finish_rate": 0.2, "is_debut": False, "is_near_debut": False},
        a_rank=5, b_rank=0,
        context={"weight_class": "Lightweight", "rounds_scheduled": 3,
                 "is_title_fight": False, "card_position": "prelim", "is_five_rounder": False},
        a_odds=None, b_odds=None,
    )
    assert isinstance(row, dict)
    assert "height_diff" in row
    assert "slpm_diff" in row
    assert "weight_class" in row
