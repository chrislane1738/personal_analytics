"""Application state: loaded models, fighter data, fight history for predictions."""
import logging
import numpy as np
import pandas as pd
import joblib
from config import MODELS_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self):
        self.model_a = None
        self.model_b = None
        self.feature_cols_a = None
        self.feature_cols_b = None
        self.fighters_df = None
        self.eval_results = None
        self.global_importance_a = None
        self.global_importance_b = None
        # Pre-loaded data for predictions
        self.fight_history = None
        self.running_records = None
        self.backtest_results = None
        self.odds_lookup = None

    def load_models(self):
        """Load latest trained models from artifacts directory."""
        try:
            for variant, attr_model, attr_cols in [
                ("a_no_odds", "model_a", "feature_cols_a"),
                ("b_with_odds", "model_b", "feature_cols_b"),
            ]:
                files = sorted(MODELS_DIR.glob(f"model_{variant}_*.joblib"))
                if files:
                    bundle = joblib.load(files[-1])
                    if isinstance(bundle, dict):
                        setattr(self, attr_model, bundle["model"])
                        setattr(self, attr_cols, bundle.get("feature_cols", []))
                    else:
                        setattr(self, attr_model, bundle)
                    logger.info(f"Loaded {attr_model} from {files[-1].name}")

            eval_path = MODELS_DIR / "eval_results.joblib"
            if eval_path.exists():
                self.eval_results = joblib.load(eval_path)

            imp_path = MODELS_DIR / "feature_importance.joblib"
            if imp_path.exists():
                imp = joblib.load(imp_path)
                self.global_importance_a = imp.get("model_a", [])
                self.global_importance_b = imp.get("model_b", [])
            backtest_path = MODELS_DIR / "backtest_results.json"
            if backtest_path.exists():
                import json
                self.backtest_results = json.loads(backtest_path.read_text())
                logger.info(f"Loaded backtest results ({self.backtest_results.get('total_fights', 0)} fights)")

        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def load_fighter_data(self):
        """Load fighter lookup + fight history for predictions."""
        csv_path = RAW_DATA_DIR / "ufc-master.csv"
        if not csv_path.exists():
            return

        try:
            df = pd.read_csv(csv_path)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

            # Build fight history and running records for the column mapper
            from data.column_mapper import build_fight_history, build_running_records
            self.fight_history = build_fight_history(df)
            self.running_records = build_running_records(df)
            logger.info("Built fight history and running records")

            # Build fighter profiles with stats (last row = most recent)
            fighters = {}
            for _, row in df.iterrows():
                for prefix, name_col in [("Red", "RedFighter"), ("Blue", "BlueFighter")]:
                    name = row[name_col]
                    if name not in fighters:
                        fighters[name] = {
                            "name": name, "height_cm": None, "reach_cm": None,
                            "stance": "Unknown", "wins": 0, "losses": 0,
                            "weight_class": "",
                            "slpm": 0, "str_acc": 0, "td_avg": 0,
                            "td_acc": 0, "sub_avg": 0,
                            "ko_wins": 0, "sub_wins": 0, "dec_wins": 0,
                        }
                    rec = fighters[name]
                    rec["height_cm"] = row.get(f"{prefix}HeightCms") or rec["height_cm"]
                    rec["reach_cm"] = row.get(f"{prefix}ReachCms") or rec["reach_cm"]
                    rec["stance"] = row.get(f"{prefix}Stance") or rec["stance"]
                    rec["wins"] = int(row.get(f"{prefix}Wins", 0) or 0)
                    rec["losses"] = int(row.get(f"{prefix}Losses", 0) or 0)
                    rec["weight_class"] = row.get("WeightClass", "") or rec["weight_class"]
                    rec["slpm"] = float(row.get(f"{prefix}AvgSigStrLanded", 0) or 0)
                    rec["str_acc"] = float(row.get(f"{prefix}AvgSigStrPct", 0) or 0)
                    rec["td_avg"] = float(row.get(f"{prefix}AvgTDLanded", 0) or 0)
                    rec["td_acc"] = float(row.get(f"{prefix}AvgTDPct", 0) or 0)
                    rec["sub_avg"] = float(row.get(f"{prefix}AvgSubAtt", 0) or 0)
                    rec["ko_wins"] = int(row.get(f"{prefix}WinsByKO", 0) or 0)
                    rec["sub_wins"] = int(row.get(f"{prefix}WinsBySubmission", 0) or 0)
                    dec = int(row.get(f"{prefix}WinsByDecisionMajority", 0) or 0) + \
                          int(row.get(f"{prefix}WinsByDecisionSplit", 0) or 0) + \
                          int(row.get(f"{prefix}WinsByDecisionUnanimous", 0) or 0)
                    rec["dec_wins"] = dec

            self.fighters_df = fighters
            logger.info(f"Loaded {len(fighters)} unique fighters")

            # Build odds lookup for betting simulator
            import numpy as np
            odds_lookup = {}
            for _, row in df.iterrows():
                date_str = row["Date"].strftime("%Y-%m-%d")
                key = f"{date_str}|{row['RedFighter']}|{row['BlueFighter']}"
                odds_lookup[key] = {
                    "red_odds": float(row["RedOdds"]) if pd.notna(row.get("RedOdds")) else None,
                    "blue_odds": float(row["BlueOdds"]) if pd.notna(row.get("BlueOdds")) else None,
                    "red_dec_odds": float(row["RedDecOdds"]) if pd.notna(row.get("RedDecOdds")) else None,
                    "blue_dec_odds": float(row["BlueDecOdds"]) if pd.notna(row.get("BlueDecOdds")) else None,
                    "r_ko_odds": float(row["RKOOdds"]) if pd.notna(row.get("RKOOdds")) else None,
                    "b_ko_odds": float(row["BKOOdds"]) if pd.notna(row.get("BKOOdds")) else None,
                    "r_sub_odds": float(row["RSubOdds"]) if pd.notna(row.get("RSubOdds")) else None,
                    "b_sub_odds": float(row["BSubOdds"]) if pd.notna(row.get("BSubOdds")) else None,
                    "winner": row.get("Winner"),
                    "finish": row.get("Finish"),
                }
            self.odds_lookup = odds_lookup
            logger.info(f"Built odds lookup ({len(odds_lookup)} fights)")
        except Exception as e:
            logger.error(f"Error loading fighter data: {e}", exc_info=True)

    @property
    def models_loaded(self) -> bool:
        return self.model_a is not None or self.model_b is not None


# Singleton
state = AppState()


# --- Optimizer job state ---
import json as _json
import threading as _threading
from config import DATA_DIR as _DATA_DIR

OPTIMIZER_PROGRESS_FILE = _DATA_DIR / "optimizer_progress.json"

_optimizer_thread: _threading.Thread | None = None
_optimizer_cancel = _threading.Event()


def is_optimizer_running() -> bool:
    return _optimizer_thread is not None and _optimizer_thread.is_alive()


def set_optimizer_thread(t: _threading.Thread):
    global _optimizer_thread
    _optimizer_thread = t


def clear_optimizer_cancel():
    _optimizer_cancel.clear()


def request_optimizer_cancel():
    _optimizer_cancel.set()


def is_optimizer_cancelled() -> bool:
    return _optimizer_cancel.is_set()


def get_optimizer_progress() -> dict:
    if OPTIMIZER_PROGRESS_FILE.exists():
        try:
            with open(OPTIMIZER_PROGRESS_FILE) as f:
                return _json.load(f)
        except (_json.JSONDecodeError, IOError):
            pass
    return {"status": "idle"}


def write_optimizer_progress(prog: dict):
    with open(OPTIMIZER_PROGRESS_FILE, "w") as f:
        _json.dump(prog, f)
