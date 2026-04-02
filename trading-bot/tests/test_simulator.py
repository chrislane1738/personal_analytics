"""Tests for TopstepEvalSimulator."""

import tempfile
from datetime import date, timedelta

from data.storage.database import Database
from data.storage.models import DailyBar
from topstep.config import TopstepConfig
from topstep.simulator import TopstepEvalSimulator
from topstep.strategies.orb_strategy import ORBStrategy


def _seed_db(db, symbol="ES", num_bars=60):
    """Seed database with trending bars (close > open) for signal generation."""
    db.create_tables()
    bars = []
    base = 5000.0
    for i in range(num_bars):
        d = date(2024, 1, 2) + timedelta(days=i)
        o = base + i * 2
        h = o + 30
        l = o - 10
        c = o + 20
        bars.append(
            DailyBar(
                symbol=symbol,
                date=d,
                open=o,
                high=h,
                low=l,
                close=c,
                adj_close=c,
                volume=1_000_000,
                vwap=o + 10,
            )
        )
    db.insert_daily_bars(bars)


def test_simulator_runs_to_completion():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 60)
        config = TopstepConfig(max_attempt_days=30)
        sim = TopstepEvalSimulator(
            strategy=ORBStrategy(),
            database=db,
            instrument="ES",
            config=config,
            state_machine_enabled=False,
        )
        result = sim.run_attempt(start_date=date(2024, 1, 2))
        assert result["status"] in ("pass", "fail", "timeout")
        assert result["days_traded"] >= 1
        assert "cumulative_pnl" in result
        assert "daily_pnls" in result
        db.close()


def test_simulator_respects_max_days():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 60)
        config = TopstepConfig(max_attempt_days=5)
        sim = TopstepEvalSimulator(
            strategy=ORBStrategy(),
            database=db,
            instrument="ES",
            config=config,
            state_machine_enabled=False,
        )
        result = sim.run_attempt(start_date=date(2024, 1, 2))
        assert result["days_traded"] <= 5
        db.close()


def test_simulator_with_state_machine():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 60)
        config = TopstepConfig(max_attempt_days=30)
        sim = TopstepEvalSimulator(
            strategy=ORBStrategy(),
            database=db,
            instrument="ES",
            config=config,
            state_machine_enabled=True,
        )
        result = sim.run_attempt(start_date=date(2024, 1, 2))
        assert result["status"] in ("pass", "fail", "timeout")
        assert "state_history" in result
        db.close()


def test_simulator_no_data_graceful():
    """Simulator should handle case where no bars match the date range."""
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 10)
        config = TopstepConfig(max_attempt_days=30)
        sim = TopstepEvalSimulator(
            strategy=ORBStrategy(),
            database=db,
            instrument="ES",
            config=config,
            state_machine_enabled=False,
        )
        # Start date well after seeded data ends — no bars will match
        result = sim.run_attempt(start_date=date(2025, 6, 1))
        # With no data, no days are recorded so status stays active
        assert result["status"] in ("active", "pass", "fail", "timeout")
        assert result["days_traded"] == 0
        assert result["daily_pnls"] == []
        db.close()


def test_simulator_result_has_expected_keys():
    """Verify the result dict contains all expected keys from AttemptTracker."""
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 60)
        config = TopstepConfig(max_attempt_days=30)
        sim = TopstepEvalSimulator(
            strategy=ORBStrategy(),
            database=db,
            instrument="ES",
            config=config,
            state_machine_enabled=False,
        )
        result = sim.run_attempt(start_date=date(2024, 1, 2))
        expected_keys = {
            "cumulative_pnl",
            "days_traded",
            "daily_pnls",
            "best_day_pnl",
            "worst_day_pnl",
            "status",
            "state_history",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )
        db.close()
