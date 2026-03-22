"""Tests for core/events.py — event dataclasses and serialization."""

import uuid
from datetime import date, datetime, timezone

import pytest

from core.events import (
    BarEvent,
    BaseEvent,
    EventType,
    FillEvent,
    OrderEvent,
    PortfolioUpdateEvent,
    RiskEvent,
    SignalEvent,
)


# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------

class TestEventType:
    def test_all_variants_exist(self):
        expected = {"BAR", "SIGNAL", "ORDER", "FILL", "RISK", "PORTFOLIO_UPDATE"}
        actual = {e.name for e in EventType}
        assert expected == actual


# ---------------------------------------------------------------------------
# BaseEvent
# ---------------------------------------------------------------------------

class TestBaseEvent:
    def _make(self, **kwargs):
        # BaseEvent requires event_type; use a concrete subclass or pass directly
        return BaseEvent(event_type=EventType.BAR, **kwargs)

    def test_default_timestamp_is_utc(self):
        evt = self._make()
        assert evt.timestamp.tzinfo is not None
        assert evt.timestamp.tzinfo == timezone.utc

    def test_default_event_id_is_uuid(self):
        evt = self._make()
        # Should not raise
        uuid.UUID(str(evt.event_id))

    def test_two_events_have_different_ids(self):
        a = self._make()
        b = self._make()
        assert a.event_id != b.event_id

    def test_to_dict_contains_required_keys(self):
        evt = self._make()
        d = evt.to_dict()
        assert "event_type" in d
        assert "timestamp" in d
        assert "event_id" in d

    def test_to_dict_event_type_is_string(self):
        evt = self._make()
        d = evt.to_dict()
        assert isinstance(d["event_type"], str)

    def test_to_dict_timestamp_is_iso_string(self):
        evt = self._make()
        d = evt.to_dict()
        # Should parse without error
        datetime.fromisoformat(d["timestamp"])

    def test_to_dict_event_id_is_string(self):
        evt = self._make()
        d = evt.to_dict()
        assert isinstance(d["event_id"], str)


# ---------------------------------------------------------------------------
# BarEvent
# ---------------------------------------------------------------------------

class TestBarEvent:
    def _make(self, **overrides):
        defaults = dict(
            symbol="AAPL",
            date=date(2024, 1, 15),
            open=180.0,
            high=185.0,
            low=179.5,
            close=184.0,
            adj_close=184.0,
            volume=50_000_000,
            vwap=182.5,
        )
        defaults.update(overrides)
        return BarEvent(**defaults)

    def test_creation(self):
        evt = self._make()
        assert evt.symbol == "AAPL"
        assert evt.event_type == EventType.BAR

    def test_to_dict_date_is_iso_string(self):
        evt = self._make()
        d = evt.to_dict()
        # date should become an ISO string
        assert isinstance(d["date"], str)
        # Should parse back
        date.fromisoformat(d["date"])

    def test_to_dict_contains_ohlcv(self):
        evt = self._make()
        d = evt.to_dict()
        for key in ("open", "high", "low", "close", "adj_close", "volume", "vwap"):
            assert key in d

    def test_vwap_optional_none(self):
        evt = self._make(vwap=None)
        assert evt.vwap is None
        d = evt.to_dict()
        assert d["vwap"] is None


# ---------------------------------------------------------------------------
# SignalEvent
# ---------------------------------------------------------------------------

class TestSignalEvent:
    def _make(self, **overrides):
        defaults = dict(
            symbol="AAPL",
            direction="long",
            reason="MA crossover",
            strength=0.75,
        )
        defaults.update(overrides)
        return SignalEvent(**defaults)

    def test_creation_equity(self):
        evt = self._make()
        assert evt.symbol == "AAPL"
        assert evt.direction == "long"
        assert evt.event_type == EventType.SIGNAL

    def test_creation_option(self):
        evt = self._make(
            direction="sell_put",
            option_type="put",
            strike=170.0,
            expiration=date(2024, 2, 16),
        )
        assert evt.option_type == "put"
        assert evt.strike == 170.0

    def test_optional_fields_default_none(self):
        evt = self._make()
        assert evt.option_type is None
        assert evt.strike is None
        assert evt.expiration is None

    def test_strength_stored(self):
        evt = self._make(strength=0.5)
        assert evt.strength == 0.5

    def test_to_dict_expiration_is_iso_when_set(self):
        evt = self._make(expiration=date(2024, 2, 16))
        d = evt.to_dict()
        assert isinstance(d["expiration"], str)
        date.fromisoformat(d["expiration"])

    def test_valid_directions(self):
        for direction in ("long", "short", "sell_put", "sell_call", "buy_to_close", "sell_to_close"):
            evt = self._make(direction=direction)
            assert evt.direction == direction


# ---------------------------------------------------------------------------
# OrderEvent
# ---------------------------------------------------------------------------

class TestOrderEvent:
    def _make(self, **overrides):
        defaults = dict(
            symbol="AAPL",
            action="BUY",
            order_type="market",
            quantity=100,
            price=184.0,
        )
        defaults.update(overrides)
        return OrderEvent(**defaults)

    def test_creation(self):
        evt = self._make()
        assert evt.symbol == "AAPL"
        assert evt.event_type == EventType.ORDER

    def test_optional_fields_default_none(self):
        evt = self._make()
        assert evt.option_type is None
        assert evt.strike is None
        assert evt.expiration is None
        assert evt.signal_ref is None

    def test_signal_ref_stored_as_string(self):
        ref = str(uuid.uuid4())
        evt = self._make(signal_ref=ref)
        assert evt.signal_ref == ref

    def test_valid_actions(self):
        for action in ("BUY", "SELL", "BTO", "STO", "BTC", "STC"):
            evt = self._make(action=action)
            assert evt.action == action

    def test_valid_order_types(self):
        for ot in ("market", "limit", "stop", "stop_limit"):
            evt = self._make(order_type=ot)
            assert evt.order_type == ot

    def test_to_dict(self):
        evt = self._make()
        d = evt.to_dict()
        assert d["action"] == "BUY"
        assert d["quantity"] == 100


# ---------------------------------------------------------------------------
# FillEvent
# ---------------------------------------------------------------------------

class TestFillEvent:
    def _make(self, **overrides):
        defaults = dict(
            symbol="AAPL",
            action="BUY",
            quantity=100,
            fill_price=184.05,
            commission=1.00,
            slippage=0.05,
        )
        defaults.update(overrides)
        return FillEvent(**defaults)

    def test_creation(self):
        evt = self._make()
        assert evt.symbol == "AAPL"
        assert evt.event_type == EventType.FILL

    def test_order_ref_optional(self):
        evt = self._make()
        assert evt.order_ref is None

    def test_order_ref_stored(self):
        ref = str(uuid.uuid4())
        evt = self._make(order_ref=ref)
        assert evt.order_ref == ref

    def test_to_dict(self):
        evt = self._make()
        d = evt.to_dict()
        assert d["fill_price"] == 184.05
        assert d["commission"] == 1.00


# ---------------------------------------------------------------------------
# RiskEvent
# ---------------------------------------------------------------------------

class TestRiskEvent:
    def _make(self, **overrides):
        defaults = dict(
            rule_name="MaxPositionSize",
            signal_blocked_id=str(uuid.uuid4()),
            reason="Position size exceeds 5% limit",
        )
        defaults.update(overrides)
        return RiskEvent(**defaults)

    def test_creation(self):
        evt = self._make()
        assert evt.rule_name == "MaxPositionSize"
        assert evt.event_type == EventType.RISK

    def test_signal_blocked_id_is_string(self):
        sid = str(uuid.uuid4())
        evt = self._make(signal_blocked_id=sid)
        assert isinstance(evt.signal_blocked_id, str)
        assert evt.signal_blocked_id == sid

    def test_to_dict(self):
        evt = self._make()
        d = evt.to_dict()
        assert "rule_name" in d
        assert "signal_blocked_id" in d
        assert "reason" in d


# ---------------------------------------------------------------------------
# PortfolioUpdateEvent
# ---------------------------------------------------------------------------

class TestPortfolioUpdateEvent:
    def _make(self, **overrides):
        defaults = dict(
            equity=105_000.0,
            cash=20_000.0,
            unrealized_pnl=3_000.0,
            realized_pnl=2_000.0,
            drawdown_pct=-0.05,
        )
        defaults.update(overrides)
        return PortfolioUpdateEvent(**defaults)

    def test_creation(self):
        evt = self._make()
        assert evt.equity == 105_000.0
        assert evt.event_type == EventType.PORTFOLIO_UPDATE

    def test_to_dict(self):
        evt = self._make()
        d = evt.to_dict()
        for key in ("equity", "cash", "unrealized_pnl", "realized_pnl", "drawdown_pct"):
            assert key in d

    def test_drawdown_negative(self):
        evt = self._make(drawdown_pct=-0.12)
        assert evt.drawdown_pct == -0.12
