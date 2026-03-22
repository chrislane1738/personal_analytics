"""Pydantic response models for the API."""
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    model_a_loaded: bool
    model_b_loaded: bool

class KeyFactor(BaseModel):
    feature: str
    impact: float
    abs_impact: float
    direction: str

class FightPrediction(BaseModel):
    fighter_a: str
    fighter_b: str
    weight_class: str
    card_position: str
    fighter_a_win_prob: float
    fighter_b_win_prob: float
    predicted_winner: str
    confidence: float
    key_factors: list[KeyFactor]
    a_style_primary: str
    b_style_primary: str

class EventPredictions(BaseModel):
    event_name: str
    event_date: str
    predictions: list[FightPrediction]
    model_variant: str

class FighterProfile(BaseModel):
    name: str
    height_cm: float | None
    reach_cm: float | None
    stance: str
    record: str
    style_scores: dict[str, float]
    style_sub_scores: dict[str, float]

class ModelPerformance(BaseModel):
    overall_accuracy: float
    accuracy_by_weight_class: dict[str, float]
    accuracy_by_card_position: dict[str, float]
    feature_importance: list[dict[str, float]]

class FeatureImportance(BaseModel):
    feature: str
    importance: float
