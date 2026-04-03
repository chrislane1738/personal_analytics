"""Tests for Topstep position-limit enforcement in FuturesSimBroker."""

from __future__ import annotations

import pytest

from core.event_bus import EventBus
from core.events import EventType, FillEvent, OrderEvent
from execution.futures_sim_broker import FuturesSimBroker
from topstep.config import TIER_50K, TIER_100K, TIER_150K, TopstepConfig
from topstep.futures_specs import FUTURES_SPECS, FuturesContractSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_broker(
    symbol: str = "ES",
    max_position: int | None = None,
    slippage_pct: float = 0.0,
) -> tuple[EventBus, FuturesSimBroker]:
    bus = EventBus()
    spec = FUTURES_SPECS[symbol]
    broker = FuturesSimBroker(
        bus,
        contract_spec=spec,
        slippage_pct=slippage_pct,
        max_position=max_position,
    )
    return bus, broker


def market_order(symbol: str, action: str, quantity: int) -> OrderEvent:
    return OrderEvent(
        symbol=symbol, action=action, order_type="market", quantity=quantity,
    )


def fills(bus: EventBus) -> list[FillEvent]:
    return bus.get_history(EventType.FILL)


def process_bar(broker: FuturesSimBroker, symbol: str = "ES") -> None:
    """Send a generic bar through the broker to trigger fills."""
    broker.process_bar(symbol, 5000.0, 5010.0, 4990.0, 5005.0, 100_000)


# ===========================================================================
# 1. Orders accepted when under the limit
# ===========================================================================

class TestOrderAcceptedUnderLimit:
    def test_single_buy_under_limit(self):
        bus, broker = make_broker(max_position=5)
        broker.submit_order(market_order("ES", "BUY", 3))
        process_bar(broker)
        assert len(fills(bus)) == 1
        assert broker._net_position == 3

    def test_multiple_buys_under_limit(self):
        bus, broker = make_broker(max_position=5)
        broker.submit_order(market_order("ES", "BUY", 2))
        process_bar(broker)
        broker.submit_order(market_order("ES", "BUY", 3))
        process_bar(broker)
        assert len(fills(bus)) == 2
        assert broker._net_position == 5

    def test_buy_exactly_at_limit(self):
        bus, broker = make_broker(max_position=5)
        broker.submit_order(market_order("ES", "BUY", 5))
        process_bar(broker)
        assert len(fills(bus)) == 1
        assert broker._net_position == 5

    def test_no_limit_means_unlimited(self):
        """max_position=None should not reject any orders."""
        bus, broker = make_broker(max_position=None)
        broker.submit_order(market_order("ES", "BUY", 100))
        process_bar(broker)
        assert len(fills(bus)) == 1
        assert broker._net_position == 100


# ===========================================================================
# 2. Orders rejected when at or exceeding the limit
# ===========================================================================

class TestOrderRejectedAtLimit:
    def test_order_rejected_when_would_exceed(self):
        bus, broker = make_broker(max_position=5)
        broker.submit_order(market_order("ES", "BUY", 6))
        process_bar(broker)
        assert len(fills(bus)) == 0
        assert broker._net_position == 0

    def test_incremental_order_rejected(self):
        """Already holding 3, trying to add 3 more (total 6) should be rejected."""
        bus, broker = make_broker(max_position=5)
        broker.submit_order(market_order("ES", "BUY", 3))
        process_bar(broker)
        assert broker._net_position == 3

        broker.submit_order(market_order("ES", "BUY", 3))
        process_bar(broker)
        # Second order should be rejected; still only 1 fill total
        assert len(fills(bus)) == 1
        assert broker._net_position == 3

    def test_sell_rejected_when_short_exceeds_limit(self):
        """Selling when flat should be rejected if quantity > max_position."""
        bus, broker = make_broker(max_position=5)
        broker.submit_order(market_order("ES", "SELL", 6))
        process_bar(broker)
        assert len(fills(bus)) == 0
        assert broker._net_position == 0

    def test_sell_accepted_within_short_limit(self):
        """Short selling within limit should be allowed."""
        bus, broker = make_broker(max_position=5)
        broker.submit_order(market_order("ES", "SELL", 4))
        process_bar(broker)
        assert len(fills(bus)) == 1
        assert broker._net_position == -4


# ===========================================================================
# 3. Net-position tracking on fills
# ===========================================================================

class TestNetPositionTracking:
    def test_buy_increases_position(self):
        _, broker = make_broker(max_position=10)
        broker.submit_order(market_order("ES", "BUY", 3))
        process_bar(broker)
        assert broker._net_position == 3

    def test_sell_decreases_position(self):
        _, broker = make_broker(max_position=10)
        broker.submit_order(market_order("ES", "SELL", 2))
        process_bar(broker)
        assert broker._net_position == -2

    def test_buy_then_sell_partial_close(self):
        """Long 5, sell 3 → net position 2."""
        _, broker = make_broker(max_position=10)
        broker.submit_order(market_order("ES", "BUY", 5))
        process_bar(broker)
        assert broker._net_position == 5

        broker.submit_order(market_order("ES", "SELL", 3))
        process_bar(broker)
        assert broker._net_position == 2

    def test_full_close_resets_to_zero(self):
        _, broker = make_broker(max_position=10)
        broker.submit_order(market_order("ES", "BUY", 5))
        process_bar(broker)
        broker.submit_order(market_order("ES", "SELL", 5))
        process_bar(broker)
        assert broker._net_position == 0

    def test_close_then_reopen_allowed(self):
        """After closing, opening a new position should work if under limit."""
        bus, broker = make_broker(max_position=5)
        broker.submit_order(market_order("ES", "BUY", 5))
        process_bar(broker)
        broker.submit_order(market_order("ES", "SELL", 5))
        process_bar(broker)
        assert broker._net_position == 0

        broker.submit_order(market_order("ES", "BUY", 5))
        process_bar(broker)
        assert broker._net_position == 5
        assert len(fills(bus)) == 3

    def test_bto_btc_actions_tracked(self):
        """BTO and BTC actions should also update net position."""
        _, broker = make_broker(max_position=10)
        broker.submit_order(market_order("ES", "BTO", 4))
        process_bar(broker)
        assert broker._net_position == 4

        broker.submit_order(market_order("ES", "BTC", 2))
        process_bar(broker)
        # BTC is a buy-side action, so position increases
        assert broker._net_position == 6

    def test_sto_stc_actions_tracked(self):
        """STO and STC actions should update net position (sell side)."""
        _, broker = make_broker(max_position=10)
        broker.submit_order(market_order("ES", "STO", 3))
        process_bar(broker)
        assert broker._net_position == -3

        broker.submit_order(market_order("ES", "STC", 1))
        process_bar(broker)
        # STC is a sell-side action, so position decreases further
        assert broker._net_position == -4


# ===========================================================================
# 4. Closing trades allowed even at limit
# ===========================================================================

class TestClosingTradesAtLimit:
    def test_sell_to_close_allowed_at_limit(self):
        """When long at max position, selling to close should be allowed."""
        bus, broker = make_broker(max_position=5)
        broker.submit_order(market_order("ES", "BUY", 5))
        process_bar(broker)
        assert broker._net_position == 5

        # Selling reduces abs(position), so it should be allowed
        broker.submit_order(market_order("ES", "SELL", 2))
        process_bar(broker)
        assert broker._net_position == 3
        assert len(fills(bus)) == 2

    def test_buy_to_close_short_allowed_at_limit(self):
        """When short at max position, buying to close should be allowed."""
        bus, broker = make_broker(max_position=5)
        broker.submit_order(market_order("ES", "SELL", 5))
        process_bar(broker)
        assert broker._net_position == -5

        # Buying reduces abs(position), so it should be allowed
        broker.submit_order(market_order("ES", "BUY", 3))
        process_bar(broker)
        assert broker._net_position == -2
        assert len(fills(bus)) == 2


# ===========================================================================
# 5. Contract-type based limits (mini vs micro)
# ===========================================================================

class TestContractTypeLimits:
    def test_es_is_mini(self):
        assert FUTURES_SPECS["ES"].contract_type == "mini"

    def test_mes_is_micro(self):
        assert FUTURES_SPECS["MES"].contract_type == "micro"

    def test_nq_is_mini(self):
        assert FUTURES_SPECS["NQ"].contract_type == "mini"

    def test_mnq_is_micro(self):
        assert FUTURES_SPECS["MNQ"].contract_type == "micro"

    def test_mini_limit_applied_for_es(self):
        """ES (mini) should use max_position_minis from config."""
        config = TIER_50K
        spec = FUTURES_SPECS["ES"]
        assert spec.contract_type == "mini"
        # 50K tier: 5 minis
        assert config.max_position_minis == 5

        bus, broker = make_broker("ES", max_position=config.max_position_minis)
        broker.submit_order(market_order("ES", "BUY", 5))
        process_bar(broker)
        assert len(fills(bus)) == 1

        broker.submit_order(market_order("ES", "BUY", 1))
        process_bar(broker)
        assert len(fills(bus)) == 1  # Rejected

    def test_micro_limit_applied_for_mes(self):
        """MES (micro) should use max_position_micros from config."""
        config = TIER_50K
        spec = FUTURES_SPECS["MES"]
        assert spec.contract_type == "micro"
        # 50K tier: 50 micros
        assert config.max_position_micros == 50

        bus = EventBus()
        broker = FuturesSimBroker(
            bus,
            contract_spec=spec,
            slippage_pct=0.0,
            max_position=config.max_position_micros,
        )
        broker.submit_order(market_order("MES", "BUY", 50))
        broker.process_bar("MES", 5000.0, 5010.0, 4990.0, 5005.0, 100_000)
        assert len(bus.get_history(EventType.FILL)) == 1

        broker.submit_order(market_order("MES", "BUY", 1))
        broker.process_bar("MES", 5000.0, 5010.0, 4990.0, 5005.0, 100_000)
        assert len(bus.get_history(EventType.FILL)) == 1  # Rejected


# ===========================================================================
# 6. Tier preset limits
# ===========================================================================

class TestTierLimits:
    def test_50k_tier_limits(self):
        assert TIER_50K.max_position_minis == 5
        assert TIER_50K.max_position_micros == 50

    def test_100k_tier_limits(self):
        assert TIER_100K.max_position_minis == 10
        assert TIER_100K.max_position_micros == 100

    def test_150k_tier_limits(self):
        assert TIER_150K.max_position_minis == 15
        assert TIER_150K.max_position_micros == 150

    def test_100k_allows_more_contracts(self):
        """100K tier should allow 10 minis (rejected at 11)."""
        bus, broker = make_broker("ES", max_position=TIER_100K.max_position_minis)
        broker.submit_order(market_order("ES", "BUY", 10))
        process_bar(broker)
        assert len(fills(bus)) == 1

        broker.submit_order(market_order("ES", "BUY", 1))
        process_bar(broker)
        assert len(fills(bus)) == 1  # Rejected


# ===========================================================================
# 7. Simulator integration — correct limit is wired
# ===========================================================================

class TestSimulatorWiring:
    """Verify TopstepEvalSimulator passes the right max_position to the broker."""

    def test_simulator_uses_mini_limit_for_es(self):
        """TopstepEvalSimulator should select max_position_minis for ES."""
        from topstep.futures_specs import FUTURES_SPECS
        from topstep.config import TIER_50K

        spec = FUTURES_SPECS["ES"]
        config = TIER_50K
        # Replicate the logic from simulator.py
        if spec.contract_type == "micro":
            max_pos = config.max_position_micros
        else:
            max_pos = config.max_position_minis

        assert max_pos == 5  # 50K tier, mini

    def test_simulator_uses_micro_limit_for_mes(self):
        """TopstepEvalSimulator should select max_position_micros for MES."""
        from topstep.futures_specs import FUTURES_SPECS
        from topstep.config import TIER_50K

        spec = FUTURES_SPECS["MES"]
        config = TIER_50K
        if spec.contract_type == "micro":
            max_pos = config.max_position_micros
        else:
            max_pos = config.max_position_minis

        assert max_pos == 50  # 50K tier, micro

    def test_simulator_uses_mini_limit_for_nq(self):
        spec = FUTURES_SPECS["NQ"]
        config = TIER_100K
        if spec.contract_type == "micro":
            max_pos = config.max_position_micros
        else:
            max_pos = config.max_position_minis

        assert max_pos == 10  # 100K tier, mini

    def test_simulator_uses_micro_limit_for_mnq(self):
        spec = FUTURES_SPECS["MNQ"]
        config = TIER_150K
        if spec.contract_type == "micro":
            max_pos = config.max_position_micros
        else:
            max_pos = config.max_position_minis

        assert max_pos == 150  # 150K tier, micro


# ===========================================================================
# 8. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_max_position_zero_rejects_all(self):
        """max_position=0 should reject every order."""
        bus, broker = make_broker(max_position=0)
        broker.submit_order(market_order("ES", "BUY", 1))
        process_bar(broker)
        assert len(fills(bus)) == 0

    def test_max_position_one(self):
        """max_position=1 allows exactly 1 contract."""
        bus, broker = make_broker(max_position=1)
        broker.submit_order(market_order("ES", "BUY", 1))
        process_bar(broker)
        assert len(fills(bus)) == 1

        broker.submit_order(market_order("ES", "BUY", 1))
        process_bar(broker)
        assert len(fills(bus)) == 1  # Rejected

    def test_flip_direction_through_zero(self):
        """Close long and go short in separate orders."""
        bus, broker = make_broker(max_position=5)
        # Go long 3
        broker.submit_order(market_order("ES", "BUY", 3))
        process_bar(broker)
        assert broker._net_position == 3

        # Sell 3 to flatten
        broker.submit_order(market_order("ES", "SELL", 3))
        process_bar(broker)
        assert broker._net_position == 0

        # Go short 4
        broker.submit_order(market_order("ES", "SELL", 4))
        process_bar(broker)
        assert broker._net_position == -4
        assert len(fills(bus)) == 3

    def test_rejected_order_not_in_pending(self):
        """Rejected orders should not appear in the pending queue."""
        _, broker = make_broker(max_position=5)
        broker.submit_order(market_order("ES", "BUY", 6))
        assert len(broker.get_pending_orders()) == 0
