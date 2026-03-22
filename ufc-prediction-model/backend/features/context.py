"""Fight context features: weight class, rounds, card position."""
def compute_context_features(weight_class: str, rounds_scheduled: int, is_title_fight: bool, card_position: str) -> dict:
    return {"weight_class": weight_class, "rounds_scheduled": rounds_scheduled,
            "is_title_fight": is_title_fight, "card_position": card_position,
            "is_five_rounder": rounds_scheduled == 5}
