"""Profile routes: CRUD for model profiles and profile backtesting."""
import logging
from datetime import datetime

import joblib
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from betting.profiles import (
    ALL_GROUPS,
    FIXED_FEATURES,
    build_feature_groups,
    load_profiles,
    save_profiles,
    validate_profile,
    run_profile_backtest,
)
from config import MODELS_DIR
import threading
from api.app_state import (
    is_optimizer_running, set_optimizer_thread, clear_optimizer_cancel,
    get_optimizer_progress, write_optimizer_progress,
    request_optimizer_cancel, is_optimizer_cancelled,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ProfileCreate(BaseModel):
    name: str
    multipliers: dict[str, float] = Field(default_factory=lambda: {g: 1.0 for g in ALL_GROUPS})
    disabled_features: list[str] = Field(default_factory=list)
    custom_adjustments: dict[str, float] = Field(default_factory=lambda: {"ov_ev_bonus": 0.0})


class ProfileUpdate(BaseModel):
    multipliers: dict[str, float] | None = None
    disabled_features: list[str] | None = None
    custom_adjustments: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_cols() -> list[str]:
    """Load feature columns from the latest no-odds model bundle."""
    candidates = sorted(MODELS_DIR.glob("model_a_no_odds_*.joblib"))
    if not candidates:
        raise HTTPException(status_code=500, detail="No model artifacts found")
    bundle = joblib.load(candidates[-1])
    return bundle["feature_cols"]


# ---------------------------------------------------------------------------
# GET /  — List all profiles
# ---------------------------------------------------------------------------

@router.get("/")
def list_profiles():
    """List all profiles with their last_backtest data."""
    profiles = load_profiles()
    return profiles


# ---------------------------------------------------------------------------
# GET /groups  — Feature groups and fixed features
# (must be before /{name} routes to avoid matching "groups" as a name)
# ---------------------------------------------------------------------------

@router.get("/groups")
def get_feature_groups():
    """Return feature groups and fixed features from the latest model."""
    feature_cols = _get_feature_cols()
    groups = build_feature_groups(feature_cols)
    return {"groups": groups, "fixed_features": list(FIXED_FEATURES)}


# ---------------------------------------------------------------------------
# GET /importance  — Feature importance from the latest saved model
# (must be before /{name} routes to avoid path parameter conflicts)
# ---------------------------------------------------------------------------

@router.get("/importance")
def get_feature_importance():
    """Return feature importances from the latest saved model."""
    feature_cols = _get_feature_cols()
    candidates = sorted(MODELS_DIR.glob("model_a_no_odds_*.joblib"))
    if not candidates:
        raise HTTPException(status_code=500, detail="No model artifacts found")
    bundle = joblib.load(candidates[-1])
    model = bundle["model"]

    importance = model.feature_importances_
    feature_importance = {col: float(imp) for col, imp in zip(feature_cols, importance)}

    groups = build_feature_groups(feature_cols)
    grouped = {}
    for group_name, group_features in groups.items():
        grouped[group_name] = sum(feature_importance.get(f, 0) for f in group_features)

    total = sum(grouped.values()) or 1
    grouped = {k: round(v / total * 100, 1) for k, v in grouped.items()}

    return {"feature_importance": feature_importance, "grouped_importance": grouped}


# ---------------------------------------------------------------------------
# POST /optimize/sweep  — Start sensitivity sweep
# GET  /optimize/status — Poll optimizer progress
# POST /optimize/optuna — Start Optuna optimization
# POST /optimize/stop   — Cancel running optimization
# (must be before /{name} routes to avoid path parameter conflicts)
# ---------------------------------------------------------------------------

@router.post("/optimize/sweep")
def start_sweep():
    """Start sensitivity sweep in background."""
    if is_optimizer_running():
        raise HTTPException(status_code=409, detail="An optimization job is already running")

    clear_optimizer_cancel()

    def run():
        try:
            from betting.optimizer import run_sensitivity_sweep

            write_optimizer_progress({"status": "running", "phase": "sweep", "current_trial": 0, "total_trials": 160})

            result = run_sensitivity_sweep(
                progress_callback=lambda s: write_optimizer_progress({"status": "running", **s}),
                cancel_flag=is_optimizer_cancelled,
            )

            # Auto-save best profile
            profiles = load_profiles()
            profiles["Sweep Best"] = {
                "multipliers": result["best_combined"],
                "disabled_features": [],
                "custom_adjustments": {"ov_ev_bonus": 0.0},
                "created": datetime.utcnow().isoformat(),
                "last_backtest": None,
            }
            save_profiles(profiles)

            write_optimizer_progress({"status": "completed", "phase": "sweep", "result": result})
        except Exception as e:
            logger.error(f"Sweep failed: {e}", exc_info=True)
            write_optimizer_progress({"status": "error", "error": str(e)})

    t = threading.Thread(target=run, daemon=True)
    set_optimizer_thread(t)
    t.start()

    return {"status": "running", "phase": "sweep", "total_trials": 160}


@router.post("/optimize/optuna")
def start_optuna(n_trials: int = Query(300, ge=10, le=1000)):
    """Start Optuna optimization in background."""
    if is_optimizer_running():
        raise HTTPException(status_code=409, detail="An optimization job is already running")

    clear_optimizer_cancel()

    # Check if sweep results exist for warm-start
    progress = get_optimizer_progress()
    sweep_best = None
    if progress.get("status") == "completed" and progress.get("phase") == "sweep":
        sweep_best = progress.get("result", {}).get("best_combined")

    def run():
        try:
            from betting.optimizer import run_optuna_optimization

            write_optimizer_progress({"status": "running", "phase": "optuna", "current_trial": 0, "total_trials": n_trials})

            result = run_optuna_optimization(
                n_trials=n_trials,
                sweep_best=sweep_best,
                progress_callback=lambda s: write_optimizer_progress({"status": "running", **s}),
                cancel_flag=is_optimizer_cancelled,
            )

            # Auto-save best profile
            profiles = load_profiles()
            best_multipliers = {g: 1.0 for g in ALL_GROUPS}
            best_multipliers.update(result["best_params"])
            profiles["Optuna Best"] = {
                "multipliers": best_multipliers,
                "disabled_features": [],
                "custom_adjustments": {"ov_ev_bonus": 0.0},
                "created": datetime.utcnow().isoformat(),
                "last_backtest": None,
            }
            save_profiles(profiles)

            write_optimizer_progress({"status": "completed", "phase": "optuna", "result": result})
        except Exception as e:
            logger.error(f"Optuna failed: {e}", exc_info=True)
            write_optimizer_progress({"status": "error", "error": str(e)})

    t = threading.Thread(target=run, daemon=True)
    set_optimizer_thread(t)
    t.start()

    return {"status": "running", "phase": "optuna", "total_trials": n_trials}


@router.get("/optimize/status")
def get_optimizer_status():
    """Get current optimizer progress."""
    progress = get_optimizer_progress()
    progress["running"] = is_optimizer_running()
    return progress


@router.post("/optimize/stop")
def stop_optimizer():
    """Stop the currently running optimization."""
    if not is_optimizer_running():
        raise HTTPException(status_code=400, detail="No optimization is running")
    request_optimizer_cancel()
    return {"status": "stopping"}


# ---------------------------------------------------------------------------
# POST /  — Create a new profile
# ---------------------------------------------------------------------------

@router.post("/")
def create_profile(body: ProfileCreate):
    """Create a new profile with custom multipliers and disabled features."""
    if body.name == "Default":
        raise HTTPException(status_code=400, detail="Cannot create profile with reserved name 'Default'")

    feature_cols = _get_feature_cols()
    profile_data = body.model_dump()
    errors = validate_profile(profile_data, feature_cols)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    profiles = load_profiles()
    if body.name in profiles:
        raise HTTPException(status_code=400, detail=f"Profile '{body.name}' already exists")

    profile = {
        "multipliers": body.multipliers,
        "disabled_features": body.disabled_features,
        "custom_adjustments": body.custom_adjustments,
        "created": datetime.utcnow().isoformat(),
        "last_backtest": None,
    }
    profiles[body.name] = profile
    save_profiles(profiles)

    return {"name": body.name, **profile}


# ---------------------------------------------------------------------------
# PUT /{name}  — Update a profile
# ---------------------------------------------------------------------------

@router.put("/{name}")
def update_profile(name: str, body: ProfileUpdate):
    """Update a profile's multipliers and/or disabled features."""
    if name == "Default":
        raise HTTPException(status_code=400, detail="Cannot modify the Default profile")

    profiles = load_profiles()
    if name not in profiles:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")

    profile = profiles[name]

    if body.multipliers is not None:
        profile["multipliers"] = body.multipliers
    if body.disabled_features is not None:
        profile["disabled_features"] = body.disabled_features
    if body.custom_adjustments is not None:
        profile["custom_adjustments"] = body.custom_adjustments

    # Validate updated profile
    feature_cols = _get_feature_cols()
    validation_data = {"name": name, **profile}
    errors = validate_profile(validation_data, feature_cols)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    # Preserve created timestamp, clear last_backtest
    profile["last_backtest"] = None
    profiles[name] = profile
    save_profiles(profiles)

    return {"name": name, **profile}


# ---------------------------------------------------------------------------
# DELETE /{name}  — Delete a profile
# ---------------------------------------------------------------------------

@router.delete("/{name}")
def delete_profile(name: str):
    """Delete a profile by name."""
    if name == "Default":
        raise HTTPException(status_code=400, detail="Cannot delete the Default profile")

    profiles = load_profiles()
    if name not in profiles:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")

    del profiles[name]
    save_profiles(profiles)

    return {"status": "deleted", "name": name}


# ---------------------------------------------------------------------------
# POST /{name}/backtest  — Run backtest with profile
# ---------------------------------------------------------------------------

@router.post("/{name}/backtest")
async def backtest_profile(
    name: str,
    variant: str = Query("a", description="Model variant: 'a' (no odds) or 'b' (with odds)"),
):
    """Run expanding-window backtest using a profile's feature transformations."""
    import asyncio

    profiles = load_profiles()
    if name not in profiles:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")

    profile = profiles[name]

    try:
        # Run in thread pool to avoid blocking the event loop (~25s)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, run_profile_backtest, profile, variant
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Profile backtest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Save backtest result to the profile
    profile["last_backtest"] = result
    profiles[name] = profile
    save_profiles(profiles)

    return result
