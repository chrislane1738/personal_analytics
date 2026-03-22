"""Tests for the risk management layer.

Covers:
  - MaxPositionSizeRule
  - MaxOpenPositionsRule
  - MaxSectorConcentrationRule
  - DrawdownCircuitBreakerRule
  - MaxOptionsNotionalRule
  - RiskManager full flow (approved + blocked)
  - RiskEvent emission on the event bus
  - Signal-to-action direction mapping
  - Quantity computation via _compute_quantity
"""

from __future__ import annotations

import pytest

from core.event_bus import EventBus
from core.events import EventType, OrderEvent, RiskEvent, SignalEvent
from portfolio.portfolio import Portfolio, Position
from risk.manager import RiskManager
from risk.position import PositionSummary
from risk.rules import (
    DrawdownCircuitBreakerRule,
    MaxOpenPositionsRule,
    MaxOptionsNotionalRule,
    MaxPositionSizeRule,
    MaxSectorConcentrationRule,
)


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def make_signal(
    symbol: str = "AAPL",
    direction: str = "long",
    option_type: str | None = None,
    strike: float | None = None,
) -> SignalEvent:
    return SignalEvent(symbol=symbol, direction=direction, option_type=option_type, strike=strike)


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def portfolio(bus: EventBus) -> Portfolio:
    return Portfolio(initial_capital=100_000.0, event_bus=bus)


def _base_context(equity: float = 100_000.0, price: float = 50.0, qty: int = 0) -> dict:
    return {
        "equity": equity,
        "current_prices": {"AAPL": price, "GOOG": price, "MSFT": price, "JNJ": price},
        "default_quantity": qty,
        "position_size_pct": 0.06,
    }


# ---------------------------------------------------------------------------
# 1. MaxPositionSizeRule
# ---------------------------------------------------------------------------


class TestMaxPositionSizeRule:
    """Portfolio at $100 K, 6 % limit → max $6 000 per position."""

    def _rule(self) -> MaxPositionSizeRule:
        return MaxPositionSizeRule(max_pct=0.06)

    def test_small_order_passes(self, portfolio: Portfolio):
        """120 shares × $50 = $6 000 — exactly at limit → allowed."""
        rule = self._rule()
        ctx = _base_context(price=50.0, qty=120)
        passes, reason = rule.check(make_signal(), portfolio, ctx)
        assert passes
        assert reason == ""

    def test_order_below_limit_passes(self, portfolio: Portfolio):
        """100 shares × $50 = $5 000 < $6 000 — allowed."""
        rule = self._rule()
        ctx = _base_context(price=50.0, qty=100)
        passes, _ = rule.check(make_signal(), portfolio, ctx)
        assert passes

    def test_order_above_limit_blocked(self, portfolio: Portfolio):
        """200 shares × $50 = $10 000 > $6 000 — blocked."""
        rule = self._rule()
        ctx = _base_context(price=50.0, qty=200)
        passes, reason = rule.check(make_signal(), portfolio, ctx)
        assert not passes
        assert "exceeds" in reason

    def test_high_price_stock_small_qty_passes(self, portfolio: Portfolio):
        """12 shares × $500 = $6 000 — exactly at limit → allowed."""
        rule = self._rule()
        ctx = {
            "equity": 100_000.0,
            "current_prices": {"AAPL": 500.0},
            "default_quantity": 12,
            "position_size_pct": 0.06,
        }
        passes, _ = rule.check(make_signal(), portfolio, ctx)
        assert passes

    def test_high_price_stock_excess_qty_blocked(self, portfolio: Portfolio):
        """25 shares × $500 = $12 500 > $6 000 — blocked."""
        rule = self._rule()
        ctx = {
            "equity": 100_000.0,
            "current_prices": {"AAPL": 500.0},
            "default_quantity": 25,
            "position_size_pct": 0.06,
        }
        passes, reason = rule.check(make_signal(), portfolio, ctx)
        assert not passes
        assert "exceeds" in reason

    def test_unknown_price_passes(self, portfolio: Portfolio):
        """When no price is available, the rule cannot estimate — always passes."""
        rule = self._rule()
        ctx = {
            "equity": 100_000.0,
            "current_prices": {},
            "default_quantity": 9999,
            "position_size_pct": 0.06,
        }
        passes, _ = rule.check(make_signal(), portfolio, ctx)
        assert passes

    def test_reason_contains_dollar_amounts(self, portfolio: Portfolio):
        """The reason string should mention both the order value and the limit."""
        rule = self._rule()
        ctx = _base_context(price=50.0, qty=200)
        _, reason = rule.check(make_signal(), portfolio, ctx)
        assert "$" in reason


# ---------------------------------------------------------------------------
# 2. MaxOpenPositionsRule
# ---------------------------------------------------------------------------


class TestMaxOpenPositionsRule:
    def test_below_limit_passes(self, portfolio: Portfolio):
        """19 positions < 20 limit → new signal allowed."""
        for i in range(19):
            portfolio.positions[f"SYM{i}"] = Position(
                symbol=f"SYM{i}", quantity=10, avg_cost=10.0
            )
        rule = MaxOpenPositionsRule(max_positions=20)
        passes, _ = rule.check(make_signal(), portfolio, {})
        assert passes

    def test_at_limit_blocked(self, portfolio: Portfolio):
        """20 positions == 20 limit → blocked."""
        for i in range(20):
            portfolio.positions[f"SYM{i}"] = Position(
                symbol=f"SYM{i}", quantity=10, avg_cost=10.0
            )
        rule = MaxOpenPositionsRule(max_positions=20)
        passes, reason = rule.check(make_signal(), portfolio, {})
        assert not passes
        assert "20" in reason

    def test_above_limit_blocked(self, portfolio: Portfolio):
        """21 positions > 20 limit → also blocked."""
        for i in range(21):
            portfolio.positions[f"SYM{i}"] = Position(
                symbol=f"SYM{i}", quantity=10, avg_cost=10.0
            )
        rule = MaxOpenPositionsRule(max_positions=20)
        passes, _ = rule.check(make_signal(), portfolio, {})
        assert not passes

    def test_empty_portfolio_passes(self, portfolio: Portfolio):
        rule = MaxOpenPositionsRule(max_positions=20)
        passes, _ = rule.check(make_signal(), portfolio, {})
        assert passes

    def test_custom_limit(self, portfolio: Portfolio):
        """Custom limit of 5."""
        for i in range(5):
            portfolio.positions[f"SYM{i}"] = Position(
                symbol=f"SYM{i}", quantity=1, avg_cost=1.0
            )
        rule = MaxOpenPositionsRule(max_positions=5)
        passes, reason = rule.check(make_signal(), portfolio, {})
        assert not passes
        assert "5" in reason


# ---------------------------------------------------------------------------
# 3. MaxSectorConcentrationRule
# ---------------------------------------------------------------------------


SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOG": "Technology",
    "JNJ": "Healthcare",
}


class TestMaxSectorConcentrationRule:
    def _rule(self) -> MaxSectorConcentrationRule:
        return MaxSectorConcentrationRule(max_pct=0.25, sector_map=SECTOR_MAP)

    def _ctx(self, price: float = 50.0, qty: int = 100) -> dict:
        return {
            "equity": 100_000.0,
            "current_prices": {s: price for s in SECTOR_MAP},
            "default_quantity": qty,
        }

    def test_no_existing_positions_small_trade_passes(self, portfolio: Portfolio):
        """No tech positions yet; small trade easily under 25 %."""
        rule = self._rule()
        passes, _ = rule.check(make_signal("AAPL"), portfolio, self._ctx(price=50.0, qty=100))
        assert passes

    def test_existing_tech_under_threshold_passes(self, portfolio: Portfolio):
        """Two tech stocks at 10 % each → 20 % total; adding one below 5 % → 25 % exactly."""
        # 200 shares × $50 each in MSFT + GOOG = $10 000 × 2 = $20 000 (20 % of $100 K)
        portfolio.positions["MSFT"] = Position("MSFT", 200, 50.0)
        portfolio.positions["GOOG"] = Position("GOOG", 200, 50.0)
        rule = self._rule()
        # Adding AAPL: 100 shares × $50 = $5 000 → sector total $25 000 = 25 % — at limit, passes
        passes, _ = rule.check(make_signal("AAPL"), portfolio, self._ctx(price=50.0, qty=100))
        assert passes

    def test_existing_tech_at_threshold_extra_blocked(self, portfolio: Portfolio):
        """3 tech stocks already at 20 %; adding another that pushes to 26 % → blocked."""
        # $20 000 already in tech (20 %)
        portfolio.positions["MSFT"] = Position("MSFT", 200, 50.0)
        portfolio.positions["GOOG"] = Position("GOOG", 200, 50.0)
        rule = self._rule()
        # Adding AAPL at 200 shares × $50 = $10 000 → sector = $30 000 = 30 % → blocked
        passes, reason = rule.check(
            make_signal("AAPL"), portfolio, self._ctx(price=50.0, qty=200)
        )
        assert not passes
        assert "Technology" in reason

    def test_different_sector_always_passes(self, portfolio: Portfolio):
        """Three tech stocks at 20 %; adding a Healthcare stock → always passes."""
        portfolio.positions["MSFT"] = Position("MSFT", 200, 50.0)
        portfolio.positions["GOOG"] = Position("GOOG", 200, 50.0)
        rule = self._rule()
        # JNJ is Healthcare, not Technology
        ctx = {
            "equity": 100_000.0,
            "current_prices": {s: 50.0 for s in SECTOR_MAP},
            "default_quantity": 200,
        }
        passes, _ = rule.check(make_signal("JNJ"), portfolio, ctx)
        assert passes

    def test_unknown_sector_isolated(self, portfolio: Portfolio):
        """Symbols not in sector_map get 'Unknown' sector and don't cross-contaminate."""
        portfolio.positions["MSFT"] = Position("MSFT", 200, 50.0)
        portfolio.positions["GOOG"] = Position("GOOG", 200, 50.0)
        rule = self._rule()
        ctx = {
            "equity": 100_000.0,
            "current_prices": {"NVDA": 50.0},
            "default_quantity": 10,
        }
        # NVDA not in sector_map → "Unknown", no existing "Unknown" positions → passes
        passes, _ = rule.check(make_signal("NVDA"), portfolio, ctx)
        assert passes


# ---------------------------------------------------------------------------
# 4. DrawdownCircuitBreakerRule
# ---------------------------------------------------------------------------


class TestDrawdownCircuitBreakerRule:
    def test_no_drawdown_passes(self, portfolio: Portfolio):
        """Fresh portfolio has 0 % drawdown → allowed."""
        rule = DrawdownCircuitBreakerRule(max_drawdown=-0.15)
        passes, _ = rule.check(make_signal(), portfolio, {})
        assert passes

    def test_moderate_drawdown_passes(self, portfolio: Portfolio):
        """–10 % drawdown < –15 % limit → allowed."""
        portfolio.high_water_mark = 100_000.0
        portfolio.cash = 90_000.0
        rule = DrawdownCircuitBreakerRule(max_drawdown=-0.15)
        passes, _ = rule.check(make_signal(), portfolio, {})
        assert passes

    def test_at_limit_blocked(self, portfolio: Portfolio):
        """Exactly at –15 % drawdown → blocked (<=)."""
        portfolio.high_water_mark = 100_000.0
        portfolio.cash = 85_000.0
        rule = DrawdownCircuitBreakerRule(max_drawdown=-0.15)
        passes, reason = rule.check(make_signal(), portfolio, {})
        assert not passes
        assert "Circuit breaker" in reason

    def test_beyond_limit_blocked(self, portfolio: Portfolio):
        """–16 % drawdown > –15 % limit → blocked."""
        portfolio.high_water_mark = 100_000.0
        portfolio.cash = 84_000.0
        rule = DrawdownCircuitBreakerRule(max_drawdown=-0.15)
        passes, reason = rule.check(make_signal(), portfolio, {})
        assert not passes
        assert "-16.0%" in reason or "16.0" in reason

    def test_reason_contains_both_drawdown_and_limit(self, portfolio: Portfolio):
        portfolio.high_water_mark = 100_000.0
        portfolio.cash = 80_000.0
        rule = DrawdownCircuitBreakerRule(max_drawdown=-0.15)
        _, reason = rule.check(make_signal(), portfolio, {})
        assert "20.0" in reason  # actual drawdown
        assert "15.0" in reason  # limit


# ---------------------------------------------------------------------------
# 5. MaxOptionsNotionalRule
# ---------------------------------------------------------------------------


class TestMaxOptionsNotionalRule:
    def test_equity_signal_always_passes(self, portfolio: Portfolio):
        rule = MaxOptionsNotionalRule(max_pct=0.30)
        passes, _ = rule.check(make_signal(option_type=None), portfolio, {})
        assert passes

    def test_options_signal_passes_placeholder(self, portfolio: Portfolio):
        """Options support is a placeholder — currently always allows."""
        rule = MaxOptionsNotionalRule(max_pct=0.30)
        passes, _ = rule.check(make_signal(option_type="put"), portfolio, {})
        assert passes


# ---------------------------------------------------------------------------
# 6. PositionSummary dataclass
# ---------------------------------------------------------------------------


class TestPositionSummary:
    def test_fields_stored_correctly(self):
        ps = PositionSummary(
            symbol="AAPL",
            quantity=100,
            market_value=5_000.0,
            sector="Technology",
        )
        assert ps.symbol == "AAPL"
        assert ps.quantity == 100
        assert ps.market_value == pytest.approx(5_000.0)
        assert ps.sector == "Technology"
        assert ps.is_option is False
        assert ps.option_notional == pytest.approx(0.0)

    def test_options_fields(self):
        ps = PositionSummary(
            symbol="AAPL",
            quantity=1,
            market_value=500.0,
            sector="Technology",
            is_option=True,
            option_notional=15_000.0,
        )
        assert ps.is_option is True
        assert ps.option_notional == pytest.approx(15_000.0)


# ---------------------------------------------------------------------------
# 7. RiskManager — full flow
# ---------------------------------------------------------------------------


class TestRiskManagerFullFlow:
    def _manager(self, bus: EventBus, rules=None) -> RiskManager:
        if rules is None:
            rules = [
                MaxPositionSizeRule(max_pct=0.06),
                MaxOpenPositionsRule(max_positions=20),
            ]
        return RiskManager(rules=rules, event_bus=bus)

    def test_approved_signal_returns_order_event(self, bus: EventBus, portfolio: Portfolio):
        mgr = self._manager(bus)
        ctx = _base_context(price=50.0, qty=100)
        result = mgr.evaluate_signal(make_signal("AAPL", "long"), portfolio, ctx)
        assert isinstance(result, OrderEvent)

    def test_blocked_signal_returns_none(self, bus: EventBus, portfolio: Portfolio):
        """Fill 20 positions to trigger MaxOpenPositionsRule."""
        for i in range(20):
            portfolio.positions[f"SYM{i}"] = Position(f"SYM{i}", 10, 10.0)
        mgr = self._manager(bus)
        ctx = _base_context(price=50.0, qty=100)
        result = mgr.evaluate_signal(make_signal("AAPL", "long"), portfolio, ctx)
        assert result is None

    def test_approved_order_event_has_correct_symbol(self, bus: EventBus, portfolio: Portfolio):
        mgr = self._manager(bus)
        ctx = _base_context(price=50.0, qty=100)
        order = mgr.evaluate_signal(make_signal("GOOG", "long"), portfolio, ctx)
        assert order is not None
        assert order.symbol == "GOOG"

    def test_approved_order_event_has_market_type(self, bus: EventBus, portfolio: Portfolio):
        mgr = self._manager(bus)
        ctx = _base_context(price=50.0, qty=100)
        order = mgr.evaluate_signal(make_signal(), portfolio, ctx)
        assert order.order_type == "market"

    def test_approved_order_has_signal_ref(self, bus: EventBus, portfolio: Portfolio):
        mgr = self._manager(bus)
        signal = make_signal()
        ctx = _base_context(price=50.0, qty=100)
        order = mgr.evaluate_signal(signal, portfolio, ctx)
        assert order.signal_ref == str(signal.event_id)

    def test_approved_options_signal(self, bus: EventBus, portfolio: Portfolio):
        mgr = self._manager(bus)
        signal = make_signal("AAPL", "sell_put", option_type="put", strike=150.0)
        ctx = _base_context(price=50.0, qty=1)
        order = mgr.evaluate_signal(signal, portfolio, ctx)
        assert order is not None
        assert order.option_type == "put"
        assert order.strike == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# 8. RiskEvent emission
# ---------------------------------------------------------------------------


class TestRiskEventEmission:
    def test_blocked_signal_emits_risk_event(self, bus: EventBus, portfolio: Portfolio):
        """Blocked signal must emit exactly one RiskEvent."""
        received: list[RiskEvent] = []
        bus.subscribe(EventType.RISK, received.append)

        for i in range(20):
            portfolio.positions[f"SYM{i}"] = Position(f"SYM{i}", 10, 10.0)

        mgr = RiskManager(
            rules=[MaxOpenPositionsRule(max_positions=20)],
            event_bus=bus,
        )
        mgr.evaluate_signal(make_signal(), portfolio, {})

        assert len(received) == 1
        assert isinstance(received[0], RiskEvent)

    def test_risk_event_has_correct_rule_name(self, bus: EventBus, portfolio: Portfolio):
        received: list[RiskEvent] = []
        bus.subscribe(EventType.RISK, received.append)

        for i in range(20):
            portfolio.positions[f"SYM{i}"] = Position(f"SYM{i}", 10, 10.0)

        mgr = RiskManager(
            rules=[MaxOpenPositionsRule(max_positions=20)],
            event_bus=bus,
        )
        mgr.evaluate_signal(make_signal(), portfolio, {})

        assert received[0].rule_name == "MaxOpenPositionsRule"

    def test_risk_event_has_signal_blocked_id(self, bus: EventBus, portfolio: Portfolio):
        received: list[RiskEvent] = []
        bus.subscribe(EventType.RISK, received.append)

        for i in range(20):
            portfolio.positions[f"SYM{i}"] = Position(f"SYM{i}", 10, 10.0)

        signal = make_signal()
        mgr = RiskManager(
            rules=[MaxOpenPositionsRule(max_positions=20)],
            event_bus=bus,
        )
        mgr.evaluate_signal(signal, portfolio, {})

        assert received[0].signal_blocked_id == str(signal.event_id)

    def test_risk_event_has_non_empty_reason(self, bus: EventBus, portfolio: Portfolio):
        received: list[RiskEvent] = []
        bus.subscribe(EventType.RISK, received.append)

        for i in range(20):
            portfolio.positions[f"SYM{i}"] = Position(f"SYM{i}", 10, 10.0)

        mgr = RiskManager(
            rules=[MaxOpenPositionsRule(max_positions=20)],
            event_bus=bus,
        )
        mgr.evaluate_signal(make_signal(), portfolio, {})

        assert received[0].reason != ""

    def test_approved_signal_emits_no_risk_event(self, bus: EventBus, portfolio: Portfolio):
        received: list[RiskEvent] = []
        bus.subscribe(EventType.RISK, received.append)

        mgr = RiskManager(
            rules=[MaxOpenPositionsRule(max_positions=20)],
            event_bus=bus,
        )
        ctx = _base_context(price=50.0, qty=100)
        mgr.evaluate_signal(make_signal(), portfolio, ctx)

        assert len(received) == 0

    def test_risk_event_rule_name_for_size_rule(self, bus: EventBus, portfolio: Portfolio):
        """Verify RiskEvent rule_name matches the class name of the failing rule."""
        received: list[RiskEvent] = []
        bus.subscribe(EventType.RISK, received.append)

        mgr = RiskManager(
            rules=[MaxPositionSizeRule(max_pct=0.06)],
            event_bus=bus,
        )
        ctx = _base_context(price=50.0, qty=9999)  # massively oversized
        mgr.evaluate_signal(make_signal(), portfolio, ctx)

        assert len(received) == 1
        assert received[0].rule_name == "MaxPositionSizeRule"

    def test_first_failing_rule_emits_single_risk_event(
        self, bus: EventBus, portfolio: Portfolio
    ):
        """When multiple rules would fail, only the first emits a RiskEvent."""
        received: list[RiskEvent] = []
        bus.subscribe(EventType.RISK, received.append)

        # Violates both rules
        for i in range(20):
            portfolio.positions[f"SYM{i}"] = Position(f"SYM{i}", 10, 10.0)

        mgr = RiskManager(
            rules=[
                MaxOpenPositionsRule(max_positions=20),
                MaxPositionSizeRule(max_pct=0.06),
            ],
            event_bus=bus,
        )
        ctx = _base_context(price=50.0, qty=9999)
        mgr.evaluate_signal(make_signal(), portfolio, ctx)

        assert len(received) == 1  # stops at first failure
        assert received[0].rule_name == "MaxOpenPositionsRule"


# ---------------------------------------------------------------------------
# 9. Signal-to-action mapping
# ---------------------------------------------------------------------------


class TestSignalToAction:
    def _manager(self, bus: EventBus) -> RiskManager:
        return RiskManager(rules=[], event_bus=bus)

    @pytest.mark.parametrize(
        "direction, expected_action",
        [
            ("long", "BUY"),
            ("short", "SELL"),
            ("sell_put", "STO"),
            ("sell_call", "STO"),
            ("buy_to_close", "BTC"),
            ("sell_to_close", "STC"),
        ],
    )
    def test_direction_maps_to_correct_action(
        self, direction: str, expected_action: str, bus: EventBus, portfolio: Portfolio
    ):
        mgr = self._manager(bus)
        signal = make_signal("AAPL", direction)
        ctx = _base_context(price=50.0, qty=10)
        order = mgr.evaluate_signal(signal, portfolio, ctx)
        assert order is not None
        assert order.action == expected_action

    def test_unknown_direction_defaults_to_buy(self, bus: EventBus, portfolio: Portfolio):
        mgr = self._manager(bus)
        signal = make_signal("AAPL", "mystery_direction")
        ctx = _base_context(price=50.0, qty=10)
        order = mgr.evaluate_signal(signal, portfolio, ctx)
        assert order is not None
        assert order.action == "BUY"


# ---------------------------------------------------------------------------
# 10. Quantity computation
# ---------------------------------------------------------------------------


class TestComputeQuantity:
    def _manager(self, bus: EventBus) -> RiskManager:
        return RiskManager(rules=[], event_bus=bus)

    def test_quantity_based_on_pct_of_equity(self, bus: EventBus, portfolio: Portfolio):
        """6 % of $100 K = $6 000; at $50/share → 120 shares."""
        mgr = self._manager(bus)
        ctx = {
            "equity": 100_000.0,
            "current_prices": {"AAPL": 50.0},
            "position_size_pct": 0.06,
            "default_quantity": 0,
        }
        order = mgr.evaluate_signal(make_signal(), portfolio, ctx)
        assert order.quantity == 120

    def test_zero_price_returns_zero_quantity(self, bus: EventBus, portfolio: Portfolio):
        mgr = self._manager(bus)
        ctx = {
            "equity": 100_000.0,
            "current_prices": {"AAPL": 0},
            "position_size_pct": 0.06,
            "default_quantity": 0,
        }
        order = mgr.evaluate_signal(make_signal(), portfolio, ctx)
        assert order.quantity == 0

    def test_quantity_truncated_to_int(self, bus: EventBus, portfolio: Portfolio):
        """$6 000 / $55 = 109.09 → truncated to 109."""
        mgr = self._manager(bus)
        ctx = {
            "equity": 100_000.0,
            "current_prices": {"AAPL": 55.0},
            "position_size_pct": 0.06,
            "default_quantity": 0,
        }
        order = mgr.evaluate_signal(make_signal(), portfolio, ctx)
        assert order.quantity == 109

    def test_quantity_falls_back_to_initial_capital(self, bus: EventBus, portfolio: Portfolio):
        """When equity not in context, initial_capital is used."""
        mgr = self._manager(bus)
        ctx = {
            "current_prices": {"AAPL": 50.0},
            "position_size_pct": 0.06,
            "default_quantity": 0,
        }
        order = mgr.evaluate_signal(make_signal(), portfolio, ctx)
        # 6 % of $100 K = $6 000 / $50 = 120
        assert order.quantity == 120
