"""Generate predictions for upcoming fights."""
import json
import logging
from datetime import date
from pathlib import Path
import joblib
import pandas as pd
from config import MODELS_DIR, PREDICTIONS_DIR
from models.explain import compute_shap_values, get_top_factors

logger = logging.getLogger(__name__)

def load_latest_model(variant: str) -> tuple:
    pattern = f"model_{variant}_*.joblib"
    files = sorted(MODELS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No model found for variant '{variant}' in {MODELS_DIR}")
    path = files[-1]
    return joblib.load(path), path

def predict_fight(model, features: pd.DataFrame, feature_names: list[str]) -> dict:
    X = features[feature_names]
    prob = model.predict_proba(X)[0]
    shap_vals = compute_shap_values(model, X)
    factors = get_top_factors(shap_vals[0], feature_names, top_n=5)
    return {
        "fighter_a_win_prob": float(prob[1]), "fighter_b_win_prob": float(prob[0]),
        "predicted_winner": "A" if prob[1] > 0.5 else "B",
        "confidence": float(max(prob)), "key_factors": factors,
    }

def save_predictions(predictions: list[dict], event_name: str) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{date.today().isoformat()}_{event_name.replace(' ', '_')}.json"
    path = PREDICTIONS_DIR / filename
    path.write_text(json.dumps(predictions, indent=2, default=str))
    return path
