"""Topstep account configuration and tier presets."""

from dataclasses import dataclass


@dataclass
class TopstepConfig:
    """Configuration for Topstep trading accounts and evaluation rules.

    Attributes:
        account_size: Total account size for the evaluation (default: 50,000)
        profit_target: Target profit to pass the evaluation (default: 3,000)
        max_loss: Maximum allowed loss before failing (default: 2,000)
        consistency_pct: Required minimum consistency percentage (default: 0.50)
        subscription_fee: Monthly subscription fee in dollars (default: 49)
        activation_fee: One-time activation fee in dollars (default: 149)
        max_payout: Maximum payout amount (default: 5,000)
        payout_split: Trader's payout split percentage (default: 0.90)
        max_position_minis: Maximum number of ES/MES mini contracts (default: 5)
        max_position_micros: Maximum number of MES micro contracts (default: 50)
        max_attempt_days: Maximum days to complete evaluation (default: 60)
    """

    account_size: float = 50_000.0
    profit_target: float = 3_000.0
    max_loss: float = 2_000.0
    consistency_pct: float = 0.50
    subscription_fee: float = 49.0
    activation_fee: float = 149.0
    max_payout: float = 5_000.0
    payout_split: float = 0.90
    max_position_minis: int = 5
    max_position_micros: int = 50
    max_attempt_days: int = 60
    flat_by_time: str = "15:10"       # CT (Central Time) — Topstep's deadline
    session_close: str = "16:00"      # ET (Eastern Time) — market close
    force_flat_daily: bool = True     # Force flat at end of each daily bar


# Tier presets for common Topstep account sizes
TIER_50K = TopstepConfig(
    account_size=50_000.0,
    profit_target=3_000.0,
    max_loss=2_000.0,
    max_position_minis=5,
    max_position_micros=50,
    subscription_fee=49.0,
)

TIER_100K = TopstepConfig(
    account_size=100_000.0,
    profit_target=6_000.0,
    max_loss=3_000.0,
    max_position_minis=10,
    max_position_micros=100,
    subscription_fee=99.0,
)

TIER_150K = TopstepConfig(
    account_size=150_000.0,
    profit_target=9_000.0,
    max_loss=4_500.0,
    max_position_minis=15,
    max_position_micros=150,
    subscription_fee=149.0,
)

# Mapping of tier names to configurations
TIERS = {
    "50k": TIER_50K,
    "100k": TIER_100K,
    "150k": TIER_150K,
}
