"""Black-Scholes option pricing and Greeks.

All functions assume European-style options.  Inputs / outputs use
standard financial conventions:

- time_to_expiry in *years*  (e.g. 30 calendar days → 30/365 ≈ 0.0822)
- risk_free_rate as a decimal (e.g. 4 % → 0.04)
- volatility     as a decimal (e.g. 20 % → 0.20)
"""

from __future__ import annotations

import math

from scipy.stats import norm


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
) -> tuple[float, float]:
    """Return (d1, d2) for the Black-Scholes formula."""
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + volatility ** 2 / 2.0) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return d1, d2


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
) -> float:
    """Calculate the Black-Scholes option price.

    Parameters
    ----------
    spot:
        Current underlying price.
    strike:
        Option strike price.
    time_to_expiry:
        Time to expiration in years (e.g. 30 days → 30/365).
    risk_free_rate:
        Continuously compounded risk-free rate (e.g. 0.04 for 4 %).
    volatility:
        Implied / historical volatility (e.g. 0.20 for 20 %).
    option_type:
        ``"call"`` or ``"put"`` (case-insensitive).

    Returns
    -------
    float
        Theoretical option price.

    Raises
    ------
    ValueError
        If ``option_type`` is not ``"call"`` or ``"put"``.
    """
    if time_to_expiry <= 0:
        # At or past expiry — intrinsic value only.
        if option_type.lower() == "call":
            return max(spot - strike, 0.0)
        elif option_type.lower() == "put":
            return max(strike - spot, 0.0)
        else:
            raise ValueError(f"Unknown option_type: {option_type!r}")

    d1, d2 = _d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    discount = math.exp(-risk_free_rate * time_to_expiry)

    if option_type.lower() == "call":
        return spot * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    elif option_type.lower() == "put":
        return strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option_type: {option_type!r}")


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
) -> dict[str, float]:
    """Calculate Black-Scholes option Greeks.

    Parameters
    ----------
    spot, strike, time_to_expiry, risk_free_rate, volatility:
        Same conventions as :func:`black_scholes_price`.
    option_type:
        ``"call"`` or ``"put"`` (case-insensitive).

    Returns
    -------
    dict with keys:
        ``delta``   — rate of change of option price w.r.t. spot
        ``gamma``   — rate of change of delta w.r.t. spot
        ``theta``   — daily time decay (option price change per calendar day)
        ``vega``    — price change per 1 percentage-point move in volatility

    Raises
    ------
    ValueError
        If ``option_type`` is not ``"call"`` or ``"put"``.
    """
    if time_to_expiry <= 0:
        # Greeks are degenerate at expiry.
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    d1, d2 = _d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility)
    sqrt_t = math.sqrt(time_to_expiry)
    discount = math.exp(-risk_free_rate * time_to_expiry)

    # Gamma and vega are the same for calls and puts.
    gamma = norm.pdf(d1) / (spot * volatility * sqrt_t)
    # Vega per 1 % move in volatility (divide by 100).
    vega = spot * norm.pdf(d1) * sqrt_t / 100.0

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (
            -spot * norm.pdf(d1) * volatility / (2.0 * sqrt_t)
            - risk_free_rate * strike * discount * norm.cdf(d2)
        ) / 365.0
    elif option_type.lower() == "put":
        delta = norm.cdf(d1) - 1.0
        theta = (
            -spot * norm.pdf(d1) * volatility / (2.0 * sqrt_t)
            + risk_free_rate * strike * discount * norm.cdf(-d2)
        ) / 365.0
    else:
        raise ValueError(f"Unknown option_type: {option_type!r}")

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
    }
