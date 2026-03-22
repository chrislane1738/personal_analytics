"""SHAP-based model explainability for per-prediction key factors."""
import shap
import numpy as np
import pandas as pd

def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values

def get_top_factors(shap_vals: np.ndarray, feature_names: list[str], top_n: int = 5) -> list[dict]:
    abs_vals = np.abs(shap_vals)
    top_indices = np.argsort(abs_vals)[::-1][:top_n]
    return [{"feature": feature_names[idx], "impact": float(shap_vals[idx]),
             "abs_impact": float(abs_vals[idx]),
             "direction": "positive" if shap_vals[idx] > 0 else "negative"} for idx in top_indices]

def get_global_importance(model, feature_names: list[str]) -> list[dict]:
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return [{"feature": name, "importance": float(imp)} for name, imp in ranked]
