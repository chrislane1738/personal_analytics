"""Tests for TopstepConfig dataclass and tier presets."""

from topstep.config import TopstepConfig, TIER_50K, TIER_100K, TIER_150K


def test_default_config_is_50k():
    """Test that default TopstepConfig matches 50K tier."""
    config = TopstepConfig()
    assert config.account_size == 50_000.0
    assert config.profit_target == 3_000.0
    assert config.max_loss == 2_000.0
    assert config.consistency_pct == 0.50
    assert config.subscription_fee == 49.0
    assert config.activation_fee == 149.0
    assert config.max_payout == 5_000.0
    assert config.payout_split == 0.90
    assert config.max_position_micros == 50
    assert config.max_attempt_days == 60


def test_tier_presets():
    """Test that tier presets have correct values."""
    assert TIER_50K.profit_target == 3_000.0
    assert TIER_50K.max_loss == 2_000.0
    assert TIER_50K.max_position_micros == 50
    assert TIER_100K.profit_target == 6_000.0
    assert TIER_100K.max_loss == 3_000.0
    assert TIER_100K.max_position_micros == 100
    assert TIER_150K.profit_target == 9_000.0
    assert TIER_150K.max_loss == 4_500.0
    assert TIER_150K.max_position_micros == 150


def test_config_custom_values():
    """Test that custom values override defaults."""
    config = TopstepConfig(profit_target=5_000.0, max_loss=3_000.0)
    assert config.profit_target == 5_000.0
    assert config.max_loss == 3_000.0
    assert config.account_size == 50_000.0


def test_config_ev_break_even_pass_rate():
    """Test that break-even pass rate is reasonable."""
    config = TopstepConfig()
    break_even = config.subscription_fee / (config.max_payout * config.payout_split - config.activation_fee)
    assert break_even < 0.02
