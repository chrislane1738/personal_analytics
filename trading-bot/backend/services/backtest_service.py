"""Background backtest runner service.

Maintains a registry of active/completed backtest runs and spawns each
run in a background thread so the API can return immediately.
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

from core.engine import BacktestEngine
from data.storage.database import Database
from strategy.config_strategy import ConfigStrategy

logger = logging.getLogger(__name__)

# Resolve strategy directory relative to project root
_STRATEGIES_DIR = Path(__file__).resolve().parents[2] / "config" / "strategies"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunInfo:
    """Track the state of a single backtest run."""

    run_id: str
    status: RunStatus = RunStatus.RUNNING
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    error: str | None = None
    metrics: dict[str, Any] | None = None


class BacktestRunner:
    """Manages background backtest runs."""

    def __init__(self) -> None:
        self._runs: dict[str, RunInfo] = {}
        self._lock = threading.Lock()

    def start_run(
        self,
        *,
        strategy_name: str,
        universe: list[str],
        start_date: date,
        end_date: date,
        initial_capital: float = 100_000.0,
        benchmark: str = "SPY",
        position_size_pct: float = 0.06,
        database: Database,
    ) -> str:
        """Launch a backtest in a background thread. Returns the run_id."""
        run_id = str(uuid.uuid4())[:8]
        cancel_event = threading.Event()

        info = RunInfo(run_id=run_id, cancel_event=cancel_event)

        thread = threading.Thread(
            target=self._run_backtest,
            args=(
                info,
                strategy_name,
                universe,
                start_date,
                end_date,
                initial_capital,
                benchmark,
                position_size_pct,
                database,
            ),
            daemon=True,
            name=f"backtest-{run_id}",
        )
        info.thread = thread

        with self._lock:
            self._runs[run_id] = info

        thread.start()
        return run_id

    def get_status(self, run_id: str) -> RunInfo | None:
        """Return the RunInfo for a given run_id, or None."""
        with self._lock:
            return self._runs.get(run_id)

    def stop_run(self, run_id: str) -> bool:
        """Signal a running backtest to cancel. Returns True if found."""
        with self._lock:
            info = self._runs.get(run_id)
        if info is None:
            return False
        info.cancel_event.set()
        info.status = RunStatus.CANCELLED
        return True

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_backtest(
        self,
        info: RunInfo,
        strategy_name: str,
        universe: list[str],
        start_date: date,
        end_date: date,
        initial_capital: float,
        benchmark: str,
        position_size_pct: float,
        database: Database,
    ) -> None:
        """Execute the backtest (runs in a worker thread)."""
        try:
            # Resolve strategy — either a YAML file or a Python dotted path
            strategy = self._load_strategy(strategy_name)

            engine = BacktestEngine(
                strategy=strategy,
                database=database,
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                benchmark_symbol=benchmark,
                position_size_pct=position_size_pct,
                cancel_event=info.cancel_event,
            )

            metrics = engine.run()
            info.metrics = metrics

            if info.cancel_event.is_set():
                info.status = RunStatus.CANCELLED
            else:
                info.status = RunStatus.COMPLETED

        except Exception as exc:
            logger.exception("Backtest %s failed", info.run_id)
            info.status = RunStatus.FAILED
            info.error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _load_strategy(strategy_name: str):
        """Load a strategy by YAML filename or Python dotted path."""
        # Check if it looks like a YAML filename
        yaml_path = _STRATEGIES_DIR / strategy_name
        if not yaml_path.suffix:
            yaml_path = _STRATEGIES_DIR / f"{strategy_name}.yaml"

        if yaml_path.exists():
            return ConfigStrategy(str(yaml_path))

        # Try as a Python dotted path  (e.g. "strategy.mean_reversion.MeanReversionStrategy")
        try:
            parts = strategy_name.rsplit(".", 1)
            if len(parts) == 2:
                import importlib

                module = importlib.import_module(parts[0])
                cls = getattr(module, parts[1])
                return cls()
        except (ImportError, AttributeError):
            pass

        raise ValueError(
            f"Could not resolve strategy '{strategy_name}'. "
            f"Provide a YAML filename from config/strategies/ or a Python dotted path."
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_runner: BacktestRunner | None = None
_runner_lock = threading.Lock()


def get_backtest_runner() -> BacktestRunner:
    """Return the singleton BacktestRunner instance."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = BacktestRunner()
    return _runner
