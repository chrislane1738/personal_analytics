"""Tests for the CampaignRunner."""
import tempfile
from datetime import date, timedelta

from data.storage.database import Database
from data.storage.models import DailyBar
from topstep.campaign_runner import CampaignRunner, CampaignResult
from topstep.config import TopstepConfig
from topstep.strategies.orb_strategy import ORBStrategy


def _seed_db(db, symbol="ES", num_bars=200):
    db.create_tables()
    bars = []
    base = 5000.0
    for i in range(num_bars):
        d = date(2024, 1, 2) + timedelta(days=i)
        o = base + (i % 50) * 2
        h = o + 30 + (i % 7) * 5
        l = o - 10 - (i % 5) * 3
        c = o + 15 + ((-1) ** i) * 10
        bars.append(DailyBar(symbol=symbol, date=d, open=o, high=h, low=l,
            close=c, adj_close=c, volume=1_000_000, vwap=o + 10))
    db.insert_daily_bars(bars)


def test_campaign_runner_basic():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 200)
        config = TopstepConfig(max_attempt_days=10)
        runner = CampaignRunner(
            strategy_class=ORBStrategy, instrument="ES", config=config,
            database=db, state_machine_enabled=False, num_attempts=5, seed=42,
        )
        result = runner.run()
        assert isinstance(result, CampaignResult)
        assert result.num_attempts == 5
        assert 0.0 <= result.pass_rate <= 1.0
        assert len(result.attempt_outcomes) == 5
        assert result.campaign_id is not None
        db.close()


def test_campaign_runner_reproducible_with_seed():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 200)
        config = TopstepConfig(max_attempt_days=10)
        r1 = CampaignRunner(strategy_class=ORBStrategy, instrument="ES", config=config,
            database=db, state_machine_enabled=False, num_attempts=5, seed=42).run()
        r2 = CampaignRunner(strategy_class=ORBStrategy, instrument="ES", config=config,
            database=db, state_machine_enabled=False, num_attempts=5, seed=42).run()
        assert r1.pass_rate == r2.pass_rate
        assert r1.pnl_distribution == r2.pnl_distribution
        db.close()


def test_campaign_runner_computes_ev():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 200)
        config = TopstepConfig(max_attempt_days=10)
        result = CampaignRunner(
            strategy_class=ORBStrategy, instrument="ES", config=config,
            database=db, state_machine_enabled=False, num_attempts=5, seed=42,
        ).run()
        assert isinstance(result.ev_per_attempt, float)
        assert isinstance(result.cost_to_funded, float)
        assert isinstance(result.annual_ev, float)
        db.close()


def test_campaign_runner_distributions_populated():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        db = Database(f.name)
        _seed_db(db, "ES", 200)
        config = TopstepConfig(max_attempt_days=10)
        result = CampaignRunner(
            strategy_class=ORBStrategy, instrument="ES", config=config,
            database=db, state_machine_enabled=False, num_attempts=5, seed=42,
        ).run()
        assert len(result.pnl_distribution) == 5
        assert len(result.days_distribution) == 5
        db.close()
