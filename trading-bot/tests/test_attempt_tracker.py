"""Tests for topstep.attempt_tracker — per-attempt state tracker."""

from topstep.config import TopstepConfig
from topstep.attempt_tracker import AttemptTracker, AttemptStatus


def test_initial_state():
    tracker = AttemptTracker(TopstepConfig())
    assert tracker.status == AttemptStatus.ACTIVE
    assert tracker.cumulative_pnl == 0.0
    assert tracker.days_traded == 0
    assert tracker.daily_pnls == []


def test_record_day():
    tracker = AttemptTracker(TopstepConfig())
    tracker.record_day(pnl=500.0, eod_balance=50500.0)
    assert tracker.cumulative_pnl == 500.0
    assert tracker.days_traded == 1
    assert tracker.daily_pnls == [500.0]


def test_pass_updates_status():
    # Two equal days of 1500 each: total_profit=3000 >= target, best_day=1500 == 50% → consistent
    tracker = AttemptTracker(TopstepConfig())
    tracker.record_day(pnl=1500.0, eod_balance=51500.0)
    assert tracker.status == AttemptStatus.ACTIVE
    tracker.record_day(pnl=1500.0, eod_balance=53000.0)
    assert tracker.status == AttemptStatus.PASS


def test_fail_updates_status():
    tracker = AttemptTracker(TopstepConfig())
    tracker.record_day(pnl=-2100.0, eod_balance=47900.0)
    assert tracker.status == AttemptStatus.FAIL


def test_state_history_tracked():
    tracker = AttemptTracker(TopstepConfig())
    tracker.record_day(pnl=600.0, eod_balance=50600.0)   # cumulative_pnl=600 → CAREFUL
    tracker.record_day(pnl=-1300.0, eod_balance=49300.0) # cumulative_pnl=-700 → AGGRESSIVE
    assert len(tracker.state_history) == 2


def test_to_dict():
    tracker = AttemptTracker(TopstepConfig())
    tracker.record_day(pnl=500.0, eod_balance=50500.0)
    d = tracker.to_dict()
    assert d["cumulative_pnl"] == 500.0
    assert d["days_traded"] == 1
    assert d["status"] == "active"
    assert "state_history" in d
