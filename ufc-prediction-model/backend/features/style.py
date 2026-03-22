"""Fighting style classification: 4 primary archetypes + sub-types with fuzzy scores.

Hierarchy (spec §4):
  Striker → Power Puncher, Counter-Striker, Pressure Fighter
  Wrestler → Control Wrestler, Ground & Pound
  Grappler → Submission Hunter, Positional Grappler
  Balanced → Adaptive, Defense-First
"""
import numpy as np
from config import STYLE_THRESHOLD

def _clamp(val: float) -> float:
    return max(0.0, min(1.0, val))

def compute_style_scores(stats: dict) -> dict:
    slpm = stats.get("slpm", 0)
    td_avg = stats.get("td_avg", 0)
    sub_avg = stats.get("sub_avg", 0)
    str_acc = stats.get("str_acc", 0)
    td_acc = stats.get("td_acc", 0)
    td_def = stats.get("td_def", 0)
    str_def = stats.get("str_def", 0)
    ko_pct = stats.get("ko_win_pct", 0)
    sub_pct = stats.get("sub_win_pct", 0)

    strike_signal = _clamp(slpm / 8.0) * 0.4 + _clamp(str_acc / 0.65) * 0.3 + ko_pct * 0.3
    wrestle_signal = _clamp(td_avg / 6.0) * 0.4 + _clamp(td_acc / 0.60) * 0.3 + (1 - ko_pct - sub_pct) * 0.3
    grapple_signal = _clamp(sub_avg / 3.0) * 0.4 + sub_pct * 0.4 + _clamp(td_avg / 6.0) * 0.2

    max_signal = max(strike_signal, wrestle_signal, grapple_signal, 0.01)
    balance_signal = 1.0 - (max_signal - min(strike_signal, wrestle_signal, grapple_signal)) / max_signal
    balance_signal = balance_signal * 0.6 + _clamp(str_def / 0.65) * 0.2 + _clamp(td_def / 0.75) * 0.2

    return {"striker": _clamp(strike_signal), "wrestler": _clamp(wrestle_signal),
            "grappler": _clamp(grapple_signal), "balanced": _clamp(balance_signal)}

def get_sub_scores(primary_scores: dict, stats: dict, threshold: float = STYLE_THRESHOLD) -> dict:
    sub = {}
    slpm = stats.get("slpm", 0)
    sapm = stats.get("sapm", 0)
    str_acc = stats.get("str_acc", 0)
    str_def = stats.get("str_def", 0)
    ko_pct = stats.get("ko_win_pct", 0)
    kd_rate = stats.get("kd_rate", 0)
    finish_rate = stats.get("finish_rate", 0)

    if primary_scores.get("striker", 0) >= threshold:
        sub["power_puncher"] = _clamp(ko_pct * 0.4 + _clamp(kd_rate / 0.4) * 0.3 + finish_rate * 0.3)
        sub["counter_striker"] = _clamp(_clamp(str_acc / 0.60) * 0.3 + (1 - _clamp(sapm / 5.0)) * 0.4 + _clamp(str_def / 0.65) * 0.3)
        sub["pressure_fighter"] = _clamp(_clamp(slpm / 7.0) * 0.4 + _clamp(sapm / 5.0) * 0.3 + (1 - str_acc) * 0.3)
    else:
        sub["power_puncher"] = 0.0
        sub["counter_striker"] = 0.0
        sub["pressure_fighter"] = 0.0

    if primary_scores.get("wrestler", 0) >= threshold:
        td_avg = stats.get("td_avg", 0)
        dec_pct = stats.get("dec_win_pct", 0)
        sub["control_wrestler"] = _clamp(dec_pct * 0.5 + _clamp(td_avg / 5.0) * 0.5)
        sub["ground_and_pound"] = _clamp(ko_pct * 0.4 + _clamp(td_avg / 5.0) * 0.3 + finish_rate * 0.3)
    else:
        sub["control_wrestler"] = 0.0
        sub["ground_and_pound"] = 0.0

    if primary_scores.get("grappler", 0) >= threshold:
        sub_avg_val = stats.get("sub_avg", 0)
        sub_pct = stats.get("sub_win_pct", 0)
        sub["sub_hunter"] = _clamp(sub_pct * 0.5 + _clamp(sub_avg_val / 2.5) * 0.5)
        sub["positional_grappler"] = _clamp(_clamp(stats.get("td_avg", 0) / 5.0) * 0.5 + (1 - sub_pct) * 0.5)
    else:
        sub["sub_hunter"] = 0.0
        sub["positional_grappler"] = 0.0

    if primary_scores.get("balanced", 0) >= threshold:
        sub["adaptive"] = _clamp(
            (1 - abs(ko_pct - stats.get("sub_win_pct", 0) - stats.get("dec_win_pct", 0))) * 0.5
            + _clamp(slpm / 5.0) * 0.25 + _clamp(stats.get("td_avg", 0) / 3.0) * 0.25)
        sub["defense_first"] = _clamp(
            _clamp(str_def / 0.65) * 0.3 + _clamp(stats.get("td_def", 0) / 0.75) * 0.3
            + (1 - _clamp(sapm / 4.0)) * 0.4)
    else:
        sub["adaptive"] = 0.0
        sub["defense_first"] = 0.0

    return sub

def compute_all_style_features(stats: dict) -> dict:
    primary = compute_style_scores(stats)
    subs = get_sub_scores(primary, stats)
    return {**primary, **subs}
