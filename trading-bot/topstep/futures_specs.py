"""Futures contract specifications for Topstep-supported instruments.

Each spec defines the tick size, tick value, point multiplier, initial
margin requirement, and per-side commission for a single contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FuturesContractSpec:
    """Immutable specification for a single futures contract.

    Attributes
    ----------
    symbol:
        Root symbol (e.g. ``"ES"``, ``"MES"``).
    tick_size:
        Minimum price increment (0.25 for all E-mini/Micro E-mini S&P
        and Nasdaq contracts).
    tick_value:
        Dollar value of one tick move per contract.
    multiplier:
        Dollar value per full index point per contract.
    margin:
        Approximate initial margin requirement per contract.
    commission:
        Per-contract, per-side commission (charged on each fill).
    contract_type:
        ``"mini"`` for E-mini contracts (ES, NQ) or ``"micro"`` for
        Micro E-mini contracts (MES, MNQ).
    """

    symbol: str
    tick_size: float
    tick_value: float
    multiplier: float
    margin: float
    commission: float
    contract_type: str = "mini"


# Pre-built specs for commonly traded Topstep instruments.
FUTURES_SPECS: dict[str, FuturesContractSpec] = {
    "ES": FuturesContractSpec(
        symbol="ES",
        tick_size=0.25,
        tick_value=12.50,
        multiplier=50.0,
        margin=12_980.0,
        commission=1.25,
        contract_type="mini",
    ),
    "MES": FuturesContractSpec(
        symbol="MES",
        tick_size=0.25,
        tick_value=1.25,
        multiplier=5.0,
        margin=1_298.0,
        commission=0.62,
        contract_type="micro",
    ),
    "NQ": FuturesContractSpec(
        symbol="NQ",
        tick_size=0.25,
        tick_value=5.00,
        multiplier=20.0,
        margin=18_700.0,
        commission=1.25,
        contract_type="mini",
    ),
    "MNQ": FuturesContractSpec(
        symbol="MNQ",
        tick_size=0.25,
        tick_value=0.50,
        multiplier=2.0,
        margin=1_870.0,
        commission=0.62,
        contract_type="micro",
    ),
}
