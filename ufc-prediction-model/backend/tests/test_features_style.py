"""Tests for fighting style classification system."""
import pytest
from features.style import compute_style_scores, get_sub_scores

def test_pure_striker_classification():
    stats = {
        "slpm": 6.0, "sapm": 3.0, "str_acc": 0.55, "str_def": 0.60,
        "td_avg": 0.5, "td_acc": 0.30, "td_def": 0.80, "sub_avg": 0.1,
        "ko_win_pct": 0.70, "sub_win_pct": 0.05, "dec_win_pct": 0.25,
        "finish_rate": 0.75, "kd_rate": 0.3,
    }
    scores = compute_style_scores(stats)
    assert scores["striker"] > 0.6
    assert scores["wrestler"] < 0.3
    assert scores["grappler"] < 0.2

def test_pure_wrestler_classification():
    stats = {
        "slpm": 2.5, "sapm": 2.0, "str_acc": 0.42, "str_def": 0.55,
        "td_avg": 5.0, "td_acc": 0.55, "td_def": 0.75, "sub_avg": 0.3,
        "ko_win_pct": 0.10, "sub_win_pct": 0.10, "dec_win_pct": 0.80,
        "finish_rate": 0.20, "kd_rate": 0.05,
    }
    scores = compute_style_scores(stats)
    assert scores["wrestler"] > 0.6
    assert scores["striker"] < 0.4

def test_sub_scores_only_above_threshold():
    scores = {"striker": 0.8, "wrestler": 0.3, "grappler": 0.1, "balanced": 0.2}
    stats = {
        "slpm": 6.0, "sapm": 3.0, "str_acc": 0.55, "str_def": 0.60,
        "ko_win_pct": 0.70, "kd_rate": 0.3, "finish_rate": 0.75,
    }
    sub = get_sub_scores(scores, stats, threshold=0.5)
    assert "power_puncher" in sub
    assert sub.get("control_wrestler", 0) == 0
    assert sub.get("sub_hunter", 0) == 0

def test_all_scores_between_0_and_1():
    stats = {
        "slpm": 4.0, "sapm": 3.5, "str_acc": 0.48, "str_def": 0.55,
        "td_avg": 2.5, "td_acc": 0.40, "td_def": 0.65, "sub_avg": 1.0,
        "ko_win_pct": 0.35, "sub_win_pct": 0.25, "dec_win_pct": 0.40,
        "finish_rate": 0.60, "kd_rate": 0.15,
    }
    scores = compute_style_scores(stats)
    for key, val in scores.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} outside [0,1]"
