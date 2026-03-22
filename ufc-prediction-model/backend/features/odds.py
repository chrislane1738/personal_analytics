"""Betting odds features (Model B only)."""
def compute_odds_features(a_odds: float | None, b_odds: float | None) -> dict:
    if a_odds is None or b_odds is None:
        return {"a_implied_prob": None, "b_implied_prob": None, "odds_diff": None, "market_confidence": None}
    a_prob = _american_to_probability(a_odds)
    b_prob = _american_to_probability(b_odds)
    return {"a_implied_prob": a_prob, "b_implied_prob": b_prob,
            "odds_diff": a_prob - b_prob, "market_confidence": abs(a_prob - b_prob)}

def _american_to_probability(odds: float) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)
