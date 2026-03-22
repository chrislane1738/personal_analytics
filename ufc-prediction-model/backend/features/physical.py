"""Physical attribute features: height, reach, age, stance differentials."""

def _safe_diff(a_val, b_val) -> float:
    if a_val is None or b_val is None:
        return 0.0
    try:
        return float(a_val) - float(b_val)
    except (TypeError, ValueError):
        return 0.0

def compute_physical_features(fighter_a: dict, fighter_b: dict) -> dict:
    return {
        "a_height_cm": fighter_a.get("height_cm"),
        "b_height_cm": fighter_b.get("height_cm"),
        "a_reach_cm": fighter_a.get("reach_cm"),
        "b_reach_cm": fighter_b.get("reach_cm"),
        "a_age": fighter_a.get("age"),
        "b_age": fighter_b.get("age"),
        "a_stance": fighter_a.get("stance", "Unknown"),
        "b_stance": fighter_b.get("stance", "Unknown"),
        "height_diff": _safe_diff(fighter_a.get("height_cm"), fighter_b.get("height_cm")),
        "reach_diff": _safe_diff(fighter_a.get("reach_cm"), fighter_b.get("reach_cm")),
        "age_diff": _safe_diff(fighter_a.get("age"), fighter_b.get("age")),
    }
