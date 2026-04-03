"""Tests for futures contract specs, FuturesSimBroker, and Portfolio multiplier support."""

from __future__ import annotations

from datetime import date

import pytest

from core.event_bus import EventBus
from core.events import EventType, FillEvent, OrderEvent
from execution.futures_sim_broker import FuturesSimBroker
from portfolio.portfolio import Portfolio, Position
from topstep.futures_specs import FUTURES_SPECS, FuturesContractSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_futures_broker(
    spec: FuturesContractSpec | None = None,
    slippage_pct: float = 0.0,
) -> tuple[EventBus, FuturesSimBroker]:
    bus = EventBus()
    spec = spec or FUTURES_SPECS["ES"]
    broker = FuturesSimBroker(bus, contract_spec=spec, slippage_pct=slippage_pct)
    return bus, broker


def collected_fills(bus: EventBus) -> list[FillEvent]:
    return bus.get_history(EventType.FILL)


def market_order(symbol: str, action: str, quantity: int) -> OrderEvent:
    return OrderEvent(symbol=symbol, action=action, order_type="market", quantity=quantity)


def limit_order(symbol: str, action: str, quantity: int, price: float) -> OrderEvent:
    return OrderEvent(symbol=symbol, action=action, order_type="limit", quantity=quantity, price=price)


def make_fill(
    symbol: str = "ES",
    action: str = "BUY",
    quantity: int = 1,
    fill_price: float = 5000.0,
    commission: float = 0.0,
) -> FillEvent:
    return FillEvent(
        symbol=symbol,
        action=action,
        quantity=quantity,
        fill_price=fill_price,
        commission=commission,
    )


# ===========================================================================
# 1. FuturesContractSpec tests
# ===========================================================================

class TestFuturesContractSpec:
    def test_es_spec_values(self):
        spec = FUTURES_SPECS["ES"]
        assert spec.symbol == "ES"
        assert spec.tick_size == 0.25
        assert spec.tick_value == 12.50
        assert spec.multiplier == 50.0
        assert spec.margin == 12_980.0
        assert spec.commission == 1.25

    def test_mes_spec_values(self):
        spec = FUTURES_SPECS["MES"]
        assert spec.symbol == "MES"
        assert spec.multiplier == 5.0
        assert spec.tick_value == 1.25
        assert spec.commission == 0.62

    def test_nq_spec_values(self):
        spec = FUTURES_SPECS["NQ"]
        assert spec.symbol == "NQ"
        assert spec.multiplier == 20.0
        assert spec.tick_value == 5.00

    def test_mnq_spec_values(self):
        spec = FUTURES_SPECS["MNQ"]
        assert spec.symbol == "MNQ"
        assert spec.multiplier == 2.0
        assert spec.tick_value == 0.50

    def test_all_specs_present(self):
        assert set(FUTURES_SPECS.keys()) == {"ES", "MES", "NQ", "MNQ"}

    def test_spec_is_frozen(self):
        spec = FUTURES_SPECS["ES"]
        with pytest.raises(AttributeError):
            spec.multiplier = 999.0  # type: ignore[misc]

    def test_tick_value_equals_tick_times_multiplier(self):
        for spec in FUTURES_SPECS.values():
            assert spec.tick_value == pytest.approx(
                spec.tick_size * spec.multiplier, rel=1e-9
            )


# ===========================================================================
# 2. Tick rounding tests
# ===========================================================================

class TestTickRounding:
    def test_round_exact_tick(self):
        assert FuturesSimBroker.round_to_tick(5000.25, 0.25) == pytest.approx(5000.25)

    def test_round_up(self):
        # 5000.13 is between 5000.00 and 5000.25 — closer to 5000.25
        assert FuturesSimBroker.round_to_tick(5000.13, 0.25) == pytest.approx(5000.25)

    def test_round_down(self):
        # 5000.10 is closer to 5000.00
        assert FuturesSimBroker.round_to_tick(5000.10, 0.25) == pytest.approx(5000.00)

    def test_round_midpoint(self):
        # 5000.125 is exactly between 5000.00 and 5000.25 — banker's rounding → 5000.0
        result = FuturesSimBroker.round_to_tick(5000.125, 0.25)
        assert result in (5000.0, 5000.25)  # Either is acceptable at midpoint

    def test_round_zero_tick_returns_price(self):
        assert FuturesSimBroker.round_to_tick(5000.13, 0.0) == pytest.approx(5000.13)

    def test_round_negative_tick_returns_price(self):
        assert FuturesSimBroker.round_to_tick(5000.13, -1.0) == pytest.approx(5000.13)

    def test_round_nq_tick(self):
        # NQ also has 0.25 ticks
        assert FuturesSimBroker.round_to_tick(18456.37, 0.25) == pytest.approx(18456.25)

    def test_round_preserves_negative_prices(self):
        # Sanity check for hypothetical negative values
        assert FuturesSimBroker.round_to_tick(-5.10, 0.25) == pytest.approx(-5.00)


# ===========================================================================
# 3. FuturesSimBroker fill tests
# ===========================================================================

class TestFuturesSimBrokerFill:
    def test_market_buy_tick_rounded(self):
        """Fill price should be rounded to the nearest tick."""
        bus, broker = make_futures_broker(slippage_pct=0.0001)  # tiny slippage
        order = market_order("ES", "BUY", 1)
        broker.submit_order(order)

        # open=5000.0, after slippage 5000.0 * 1.0001 = 5000.5 → rounded to 5000.5
        broker.process_bar("ES", 5000.0, 5010.0, 4990.0, 5005.0, 100_000)

        fills = collected_fills(bus)
        assert len(fills) == 1
        fill_price = fills[0].fill_price
        # Verify tick alignment: fill_price mod tick_size should be ~0
        assert fill_price % 0.25 == pytest.approx(0.0, abs=1e-9)

    def test_market_buy_zero_slippage(self):
        """With zero slippage, fill price == open (already on tick)."""
        bus, broker = make_futures_broker(slippage_pct=0.0)
        order = market_order("ES", "BUY", 1)
        broker.submit_order(order)

        broker.process_bar("ES", 5000.0, 5010.0, 4990.0, 5005.0, 100_000)

        fills = collected_fills(bus)
        assert len(fills) == 1
        assert fills[0].fill_price == pytest.approx(5000.0)

    def test_per_contract_commission(self):
        """Commission = quantity * spec.commission (one side)."""
        bus, broker = make_futures_broker()
        order = market_order("ES", "BUY", 3)
        broker.submit_order(order)
        broker.process_bar("ES", 5000.0, 5010.0, 4990.0, 5005.0, 100_000)

        fills = collected_fills(bus)
        assert len(fills) == 1
        # 3 contracts * $1.25 per contract
        assert fills[0].commission == pytest.approx(3.75)

    def test_mes_commission(self):
        """MES commission = quantity * 0.62."""
        spec = FUTURES_SPECS["MES"]
        bus, broker = make_futures_broker(spec=spec)
        order = market_order("MES", "BUY", 5)
        broker.submit_order(order)
        broker.process_bar("MES", 5000.0, 5010.0, 4990.0, 5005.0, 100_000)

        fills = collected_fills(bus)
        assert fills[0].commission == pytest.approx(5 * 0.62)

    def test_limit_buy_tick_rounded(self):
        """Limit fill prices should also be tick-rounded."""
        bus, broker = make_futures_broker(slippage_pct=0.0)
        order = limit_order("ES", "BUY", 1, price=4995.0)
        broker.submit_order(order)

        broker.process_bar("ES", 4998.0, 5005.0, 4990.0, 5000.0, 50_000)
        fills = collected_fills(bus)
        assert len(fills) == 1
        assert fills[0].fill_price % 0.25 == pytest.approx(0.0, abs=1e-9)

    def test_sell_commission(self):
        """Sell-side also charges per-contract commission."""
        bus, broker = make_futures_broker()
        order = market_order("ES", "SELL", 2)
        broker.submit_order(order)
        broker.process_bar("ES", 5000.0, 5010.0, 4990.0, 5005.0, 100_000)

        fills = collected_fills(bus)
        assert fills[0].commission == pytest.approx(2 * 1.25)

    def test_pending_orders_cleared_after_fill(self):
        bus, broker = make_futures_broker()
        broker.submit_order(market_order("ES", "BUY", 1))
        broker.process_bar("ES", 5000.0, 5010.0, 4990.0, 5005.0, 100_000)
        assert broker.get_pending_orders() == []


# ===========================================================================
# 4. Portfolio multiplier tests
# ===========================================================================

class TestPortfolioMultiplier:
    def test_get_equity_with_multiplier(self):
        """Futures equity = cash + unrealised_pnl (using multiplier)."""
        bus = EventBus()
        port = Portfolio(
            initial_capital=50_000.0,
            event_bus=bus,
            contract_multipliers={"ES": 50.0},
        )

        # Simulate buying 1 ES at 5000
        fill = make_fill("ES", "BUY", 1, 5000.0, commission=1.25)
        port.update_on_fill(fill, date(2024, 1, 2))

        # Price rises to 5010 → unrealised = (5010 - 5000) * 1 * 50 = 500
        equity = port.get_equity({"ES": 5010.0})
        # Cash should be initial_capital - commission (no notional deduction)
        expected = 50_000.0 - 1.25 + 500.0
        assert equity == pytest.approx(expected)

    def test_get_equity_with_multiplier_loss(self):
        """Futures equity reflects losses correctly."""
        bus = EventBus()
        port = Portfolio(
            initial_capital=50_000.0,
            event_bus=bus,
            contract_multipliers={"ES": 50.0},
        )
        fill = make_fill("ES", "BUY", 2, 5000.0, commission=2.50)
        port.update_on_fill(fill, date(2024, 1, 2))

        # Price drops to 4990 → unrealised = (4990 - 5000) * 2 * 50 = -1000
        equity = port.get_equity({"ES": 4990.0})
        expected = 50_000.0 - 2.50 - 1_000.0
        assert equity == pytest.approx(expected)

    def test_buy_sell_roundtrip_futures(self):
        """Buy then sell should credit realised P&L * multiplier."""
        bus = EventBus()
        port = Portfolio(
            initial_capital=50_000.0,
            event_bus=bus,
            contract_multipliers={"ES": 50.0},
        )
        # Buy 1 at 5000
        port.update_on_fill(
            make_fill("ES", "BUY", 1, 5000.0, commission=1.25),
            date(2024, 1, 2),
        )
        # Sell 1 at 5020
        port.update_on_fill(
            make_fill("ES", "SELL", 1, 5020.0, commission=1.25),
            date(2024, 1, 3),
        )

        # Realised P&L = (5020 - 5000) * 1 * 50 - 1.25 = 998.75
        assert port._realized_pnl == pytest.approx(998.75)
        # Position should be closed
        assert not port.has_position("ES")
        # Cash = 50000 - 1.25 (buy comm) + 998.75 (realised) = 50997.50
        assert port.cash == pytest.approx(50_000.0 - 1.25 + 998.75)

    def test_futures_position_no_cash_deduction(self):
        """Buying futures should not deduct notional value from cash."""
        bus = EventBus()
        port = Portfolio(
            initial_capital=50_000.0,
            event_bus=bus,
            contract_multipliers={"ES": 50.0},
        )
        fill = make_fill("ES", "BUY", 1, 5000.0, commission=1.25)
        port.update_on_fill(fill, date(2024, 1, 2))
        # Cash = initial - commission only (not 5000 * 1)
        assert port.cash == pytest.approx(50_000.0 - 1.25)

    def test_equity_without_multiplier_unchanged(self):
        """Equity symbols behave exactly as before (no multiplier)."""
        bus = EventBus()
        port = Portfolio(initial_capital=100_000.0, event_bus=bus)
        fill = make_fill("AAPL", "BUY", 100, 50.0, commission=0.0)
        port.update_on_fill(fill, date(2024, 1, 2))
        equity = port.get_equity({"AAPL": 55.0})
        # cash=95000, value=100*55=5500 → 100500
        assert equity == pytest.approx(100_500.0)

    def test_mixed_equity_and_futures(self):
        """Portfolio can hold both equity and futures positions."""
        bus = EventBus()
        port = Portfolio(
            initial_capital=100_000.0,
            event_bus=bus,
            contract_multipliers={"ES": 50.0},
        )
        # Buy equity
        port.update_on_fill(
            make_fill("AAPL", "BUY", 100, 50.0, commission=0.0),
            date(2024, 1, 2),
        )
        # Buy futures
        port.update_on_fill(
            make_fill("ES", "BUY", 1, 5000.0, commission=1.25),
            date(2024, 1, 2),
        )

        prices = {"AAPL": 55.0, "ES": 5010.0}
        equity = port.get_equity(prices)
        # Equity: cash = 100000 - 5000 (AAPL notional) - 1.25 (ES comm) = 94998.75
        # AAPL value = 100 * 55 = 5500
        # ES unrealised = (5010 - 5000) * 1 * 50 = 500
        expected = 94_998.75 + 5_500.0 + 500.0
        assert equity == pytest.approx(expected)

    def test_record_equity_futures(self):
        """Equity curve records correct futures-aware values."""
        bus = EventBus()
        port = Portfolio(
            initial_capital=50_000.0,
            event_bus=bus,
            contract_multipliers={"MES": 5.0},
        )
        port.update_on_fill(
            make_fill("MES", "BUY", 5, 5000.0, commission=3.10),
            date(2024, 1, 2),
        )
        port.record_equity(date(2024, 1, 2), {"MES": 5005.0})
        # Unrealised = (5005 - 5000) * 5 * 5 = 125
        expected = 50_000.0 - 3.10 + 125.0
        assert port.equity_curve[0][1] == pytest.approx(expected)


# ===========================================================================
# 5. Integration: FuturesSimBroker + Portfolio roundtrip
# ===========================================================================

class TestFuturesBrokerPortfolioIntegration:
    def test_buy_and_sell_es_contract(self):
        """Full roundtrip: broker fills buy+sell, portfolio tracks P&L."""
        bus = EventBus()
        spec = FUTURES_SPECS["ES"]
        broker = FuturesSimBroker(bus, contract_spec=spec, slippage_pct=0.0)
        port = Portfolio(
            initial_capital=50_000.0,
            event_bus=bus,
            contract_multipliers={"ES": spec.multiplier},
        )

        # Wire fill handler
        def on_fill(fill: FillEvent):
            port.update_on_fill(fill, date(2024, 1, 2))
        bus.subscribe(EventType.FILL, on_fill)

        # Submit buy
        broker.submit_order(market_order("ES", "BUY", 1))
        broker.process_bar("ES", 5000.0, 5010.0, 4990.0, 5005.0, 100_000)
        assert port.has_position("ES")
        assert port.positions["ES"].quantity == 1

        # Price move: equity should reflect unrealised
        equity_before_sell = port.get_equity({"ES": 5020.0})
        # unrealised = (5020 - 5000) * 1 * 50 = 1000
        expected_before = 50_000.0 - spec.commission + 1_000.0
        assert equity_before_sell == pytest.approx(expected_before)

        # Submit sell
        broker.submit_order(market_order("ES", "SELL", 1))
        broker.process_bar("ES", 5020.0, 5025.0, 5015.0, 5022.0, 80_000)
        assert not port.has_position("ES")

        # Final equity = initial + realised - commissions
        # Realised = (5020 - 5000) * 1 * 50 - 1.25 = 998.75
        final_equity = port.get_equity({})
        expected_final = 50_000.0 - spec.commission + 998.75
        assert final_equity == pytest.approx(expected_final)

    def test_mes_micro_contract_pnl(self):
        """MES micro contract has $5/point multiplier."""
        bus = EventBus()
        spec = FUTURES_SPECS["MES"]
        broker = FuturesSimBroker(bus, contract_spec=spec, slippage_pct=0.0)
        port = Portfolio(
            initial_capital=50_000.0,
            event_bus=bus,
            contract_multipliers={"MES": spec.multiplier},
        )
        bus.subscribe(EventType.FILL, lambda f: port.update_on_fill(f, date(2024, 1, 2)))

        # Buy 10 MES at 5000
        broker.submit_order(market_order("MES", "BUY", 10))
        broker.process_bar("MES", 5000.0, 5010.0, 4990.0, 5005.0, 100_000)

        # Price goes to 5010: unrealised = (5010-5000) * 10 * 5 = 500
        equity = port.get_equity({"MES": 5010.0})
        expected = 50_000.0 - (10 * spec.commission) + 500.0
        assert equity == pytest.approx(expected)

    def test_losing_trade(self):
        """Verify negative P&L is tracked correctly."""
        bus = EventBus()
        spec = FUTURES_SPECS["ES"]
        broker = FuturesSimBroker(bus, contract_spec=spec, slippage_pct=0.0)
        port = Portfolio(
            initial_capital=50_000.0,
            event_bus=bus,
            contract_multipliers={"ES": spec.multiplier},
        )
        d = date(2024, 1, 2)
        bus.subscribe(EventType.FILL, lambda f: port.update_on_fill(f, d))

        # Buy at 5000, sell at 4990 → loss of 10 pts * $50 = -$500
        broker.submit_order(market_order("ES", "BUY", 1))
        broker.process_bar("ES", 5000.0, 5010.0, 4990.0, 5005.0, 100_000)

        broker.submit_order(market_order("ES", "SELL", 1))
        broker.process_bar("ES", 4990.0, 4995.0, 4985.0, 4992.0, 100_000)

        # Realised = (4990-5000)*1*50 - 1.25 = -501.25
        assert port._realized_pnl == pytest.approx(-501.25)
        # Cash = 50000 - 1.25 (buy comm) + (-501.25) (realised) = 49497.50
        assert port.cash == pytest.approx(49_497.50)
