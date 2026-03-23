"""Background walk-forward study runner service.

Maintains a registry of active/completed walk-forward studies and spawns
each study in a background thread so the API can return immediately.

Follows the same thread-safety pattern as BacktestRunner.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

from analytics.monte_carlo import run_bootstrap, run_monte_carlo, run_window_shuffle
from analytics.walk_forward import run_walk_forward
from data.storage.database import Database
from data.storage.models import WalkForwardStudy
from strategy.config_strategy import ConfigStrategy

logger = logging.getLogger(__name__)

# Resolve strategy directory relative to project root
_STRATEGIES_DIR = Path(__file__).resolve().parents[2] / "config" / "strategies"


class StudyStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StudyInfo:
    """Track the state of a single walk-forward study."""

    study_id: str
    status: StudyStatus = StudyStatus.RUNNING
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    error: str | None = None
    windows_completed: int = 0
    windows_total: int = 0
    current_phase: str = "initializing"


class WalkForwardRunner:
    """Manages background walk-forward studies."""

    def __init__(self) -> None:
        self._studies: dict[str, StudyInfo] = {}
        self._lock = threading.Lock()

    def start_study(
        self,
        *,
        strategy_name: str,
        config_dict: dict,
        param_space: dict,
        universe: list[str],
        start_date: date,
        end_date: date,
        initial_capital: float = 100_000.0,
        benchmark: str = "SPY",
        train_months: int = 24,
        oos_months: int = 6,
        step_months: int = 3,
        objective: str = "sharpe",
        monte_carlo_modes: list[str] | None = None,
        simulations: int = 1000,
        database: Database,
    ) -> str:
        """Launch a walk-forward study in a background thread. Returns the study_id."""
        study_id = str(uuid.uuid4())[:8]
        cancel_event = threading.Event()

        info = StudyInfo(study_id=study_id, cancel_event=cancel_event)

        thread = threading.Thread(
            target=self._run_study,
            args=(
                info,
                strategy_name,
                config_dict,
                param_space,
                universe,
                start_date,
                end_date,
                initial_capital,
                benchmark,
                train_months,
                oos_months,
                step_months,
                objective,
                monte_carlo_modes or [],
                simulations,
                database,
            ),
            daemon=True,
            name=f"walk-forward-{study_id}",
        )
        info.thread = thread

        with self._lock:
            self._studies[study_id] = info

        thread.start()
        return study_id

    def get_status(self, study_id: str) -> StudyInfo | None:
        """Return the StudyInfo for a given study_id, or None."""
        with self._lock:
            return self._studies.get(study_id)

    def stop_study(self, study_id: str) -> bool:
        """Signal a running study to cancel. Returns True if found."""
        with self._lock:
            info = self._studies.get(study_id)
        if info is None:
            return False
        info.cancel_event.set()
        info.status = StudyStatus.CANCELLED
        return True

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_study(
        self,
        info: StudyInfo,
        strategy_name: str,
        config_dict: dict,
        param_space: dict,
        universe: list[str],
        start_date: date,
        end_date: date,
        initial_capital: float,
        benchmark: str,
        train_months: int,
        oos_months: int,
        step_months: int,
        objective: str,
        monte_carlo_modes: list[str],
        simulations: int,
        database: Database,
    ) -> None:
        """Execute the walk-forward study (runs in a worker thread)."""
        thread_db: Database | None = None
        try:
            # Create a fresh DB connection for this thread to avoid SQLite
            # thread-safety issues with the shared API connection.
            thread_db = Database(database._db_path)
            thread_db.create_tables()

            # Insert study record with status=running
            now = datetime.utcnow()
            study = WalkForwardStudy(
                study_id=info.study_id,
                strategy_name=strategy_name,
                config=json.dumps(config_dict),
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                train_months=train_months,
                oos_months=oos_months,
                step_months=step_months,
                objective=objective,
                status="running",
                created_at=now,
            )
            thread_db.insert_walk_forward_study(study)

            # Separate engine-level params from strategy params
            engine_param_space: dict = {}
            strategy_param_space: dict = {}
            for pname, pspec in param_space.items():
                if pspec.get("type") == "engine":
                    engine_param_space[pname] = pspec
                else:
                    strategy_param_space[pname] = pspec

            # Build strategy_loader callable
            def strategy_loader(params: dict):
                return ConfigStrategy.from_config_dict(config_dict, params)

            # Progress callback: update StudyInfo
            def progress_callback(windows_completed, windows_total, phase):
                info.windows_completed = windows_completed
                info.windows_total = windows_total
                info.current_phase = phase

            info.current_phase = "walk_forward"

            # Run walk-forward optimization
            results = run_walk_forward(
                strategy_loader=strategy_loader,
                param_space=param_space,
                database=thread_db,
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                benchmark=benchmark,
                train_months=train_months,
                oos_months=oos_months,
                step_months=step_months,
                objective=objective,
                cancel_event=info.cancel_event,
                progress_callback=progress_callback,
            )

            if info.cancel_event.is_set():
                info.status = StudyStatus.CANCELLED
                thread_db.update_walk_forward_study(
                    info.study_id, status="cancelled"
                )
                return

            # Run Monte Carlo on stitched OOS trades
            info.current_phase = "monte_carlo"
            mc_results: dict = {}

            # Collect all OOS trade P&Ls
            all_trade_pnls: list[float] = []
            window_trade_pnls: list[list[float]] = []
            for w in results.get("windows", []):
                window_pnls = [
                    t.get("pnl", 0.0) for t in w.get("oos_trades", [])
                ]
                window_trade_pnls.append(window_pnls)
                all_trade_pnls.extend(window_pnls)

            if all_trade_pnls and monte_carlo_modes:
                if "trade_shuffle" in monte_carlo_modes:
                    mc_result = run_monte_carlo(
                        trade_pnls=all_trade_pnls,
                        initial_capital=initial_capital,
                        num_simulations=simulations,
                    )
                    mc_results["trade_shuffle"] = {
                        "simulations": mc_result.simulations,
                        "median_final_equity": mc_result.median_final_equity,
                        "percentile_5": mc_result.percentile_5,
                        "percentile_95": mc_result.percentile_95,
                        "actual_final_equity": mc_result.actual_final_equity,
                        "probability_of_ruin": mc_result.probability_of_ruin,
                        "max_drawdown_median": mc_result.max_drawdown_median,
                        "max_drawdown_95": mc_result.max_drawdown_95,
                        "is_outlier": mc_result.is_outlier,
                        "percentile_bands": mc_result.percentile_bands,
                        "actual_curve": mc_result.actual_curve,
                        "drawdown_distribution": mc_result.drawdown_distribution,
                    }

                if "window_shuffle" in monte_carlo_modes and window_trade_pnls:
                    ws_result = run_window_shuffle(
                        windows=window_trade_pnls,
                        initial_capital=initial_capital,
                        num_simulations=simulations,
                    )
                    mc_results["window_shuffle"] = ws_result

                if "bootstrap" in monte_carlo_modes and window_trade_pnls:
                    bs_result = run_bootstrap(
                        windows=window_trade_pnls,
                        initial_capital=initial_capital,
                        num_simulations=simulations,
                    )
                    mc_results["bootstrap"] = bs_result

            # Serialise results — convert any non-JSON-serializable types
            results_json = json.dumps(results, default=str)
            mc_json = json.dumps(mc_results, default=str) if mc_results else ""

            # Update study record
            thread_db.update_walk_forward_study(
                info.study_id,
                status="completed",
                results=results_json,
                monte_carlo=mc_json,
            )

            info.status = StudyStatus.COMPLETED
            info.current_phase = "completed"

        except Exception as exc:
            logger.exception("Walk-forward study %s failed", info.study_id)
            info.status = StudyStatus.FAILED
            info.error = f"{type(exc).__name__}: {exc}"
            info.current_phase = "failed"
            if thread_db:
                try:
                    thread_db.update_walk_forward_study(
                        info.study_id, status="failed"
                    )
                except Exception:
                    pass
        finally:
            if thread_db:
                try:
                    thread_db.close()
                except Exception:
                    pass

    @staticmethod
    def _load_strategy_config(strategy_name: str) -> dict:
        """Load a strategy YAML file and return the parsed config dict."""
        yaml_path = _STRATEGIES_DIR / strategy_name
        if not yaml_path.suffix:
            yaml_path = _STRATEGIES_DIR / f"{strategy_name}.yaml"

        if yaml_path.exists():
            with open(yaml_path) as f:
                return yaml.safe_load(f)

        raise ValueError(
            f"Could not find strategy config '{strategy_name}'. "
            f"Provide a YAML filename from config/strategies/."
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_runner: WalkForwardRunner | None = None
_runner_lock = threading.Lock()


def get_walk_forward_runner() -> WalkForwardRunner:
    """Return the singleton WalkForwardRunner instance."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = WalkForwardRunner()
    return _runner
