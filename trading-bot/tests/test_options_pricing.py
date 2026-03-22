"""Tests for Black-Scholes option pricing and Greeks."""

from __future__ import annotations

import math

import pytest

from execution.options_pricing import black_scholes_greeks, black_scholes_price


# ---------------------------------------------------------------------------
# Shared reference parameters
# ---------------------------------------------------------------------------

SPOT = 100.0
STRIKE = 100.0
T = 30 / 365        # 30 calendar days
R = 0.05            # 5 % risk-free rate
VOL = 0.20          # 20 % volatility


# ---------------------------------------------------------------------------
# 1. ATM call price ≈ $2.53
# ---------------------------------------------------------------------------

def test_atm_call_price():
    price = black_scholes_price(SPOT, STRIKE, T, R, VOL, option_type="call")
    assert price == pytest.approx(2.53, abs=0.05)


# ---------------------------------------------------------------------------
# 2. ATM put price ≈ $2.12
# ---------------------------------------------------------------------------

def test_atm_put_price():
    price = black_scholes_price(SPOT, STRIKE, T, R, VOL, option_type="put")
    assert price == pytest.approx(2.12, abs=0.05)


# ---------------------------------------------------------------------------
# 3. Put-call parity: C - P = S - K * exp(-r*T)
# ---------------------------------------------------------------------------

def test_put_call_parity():
    call = black_scholes_price(SPOT, STRIKE, T, R, VOL, option_type="call")
    put = black_scholes_price(SPOT, STRIKE, T, R, VOL, option_type="put")
    rhs = SPOT - STRIKE * math.exp(-R * T)
    assert (call - put) == pytest.approx(rhs, abs=1e-8)


# ---------------------------------------------------------------------------
# 4. Greeks for ATM option
# ---------------------------------------------------------------------------

def test_greeks_atm_call():
    greeks = black_scholes_greeks(SPOT, STRIKE, T, R, VOL, option_type="call")

    # Delta of ATM call ≈ 0.53 (slightly above 0.5 due to drift term).
    assert greeks["delta"] == pytest.approx(0.53, abs=0.03)

    # Gamma > 0 for long options.
    assert greeks["gamma"] > 0

    # Theta < 0 — time decay costs the holder money each day.
    assert greeks["theta"] < 0

    # Vega > 0 — higher vol benefits long options.
    assert greeks["vega"] > 0


def test_greeks_atm_put():
    greeks = black_scholes_greeks(SPOT, STRIKE, T, R, VOL, option_type="put")

    # Delta of ATM put ≈ -0.47.
    assert greeks["delta"] == pytest.approx(-0.47, abs=0.03)
    assert greeks["gamma"] > 0
    assert greeks["theta"] < 0
    assert greeks["vega"] > 0


# ---------------------------------------------------------------------------
# 5. Deep ITM call — delta ≈ 1.0
# ---------------------------------------------------------------------------

def test_deep_itm_call_delta():
    greeks = black_scholes_greeks(150.0, 100.0, T, R, VOL, option_type="call")
    assert greeks["delta"] == pytest.approx(1.0, abs=0.01)


def test_deep_itm_call_price_near_intrinsic():
    price = black_scholes_price(150.0, 100.0, T, R, VOL, option_type="call")
    intrinsic = 150.0 - 100.0  # $50
    # Deep ITM call should be very close to intrinsic + slight time value.
    assert price >= intrinsic
    assert price < intrinsic + 2.0


# ---------------------------------------------------------------------------
# 6. Deep OTM put — delta ≈ 0.0
# ---------------------------------------------------------------------------

def test_deep_otm_put_delta():
    greeks = black_scholes_greeks(150.0, 100.0, T, R, VOL, option_type="put")
    assert greeks["delta"] == pytest.approx(0.0, abs=0.01)


def test_deep_otm_put_price_near_zero():
    price = black_scholes_price(150.0, 100.0, T, R, VOL, option_type="put")
    assert price == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# 7. Call and put prices are non-negative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spot,strike", [
    (100, 90),    # ITM call / OTM put
    (100, 100),   # ATM
    (100, 110),   # OTM call / ITM put
])
def test_prices_non_negative(spot, strike):
    call = black_scholes_price(float(spot), float(strike), T, R, VOL, "call")
    put = black_scholes_price(float(spot), float(strike), T, R, VOL, "put")
    assert call >= 0.0
    assert put >= 0.0


# ---------------------------------------------------------------------------
# 8. Invalid option_type raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_option_type_price():
    with pytest.raises(ValueError, match="option_type"):
        black_scholes_price(100.0, 100.0, T, R, VOL, option_type="straddle")


def test_invalid_option_type_greeks():
    with pytest.raises(ValueError, match="option_type"):
        black_scholes_greeks(100.0, 100.0, T, R, VOL, option_type="straddle")


# ---------------------------------------------------------------------------
# 9. Gamma and vega are equal for call and put (same underlying params)
# ---------------------------------------------------------------------------

def test_call_put_gamma_equality():
    call_g = black_scholes_greeks(SPOT, STRIKE, T, R, VOL, "call")
    put_g = black_scholes_greeks(SPOT, STRIKE, T, R, VOL, "put")
    assert call_g["gamma"] == pytest.approx(put_g["gamma"], rel=1e-9)
    assert call_g["vega"] == pytest.approx(put_g["vega"], rel=1e-9)


# ---------------------------------------------------------------------------
# 10. Price monotonically increases with volatility (for long options)
# ---------------------------------------------------------------------------

def test_call_price_increases_with_volatility():
    prices = [
        black_scholes_price(SPOT, STRIKE, T, R, v, "call")
        for v in (0.10, 0.20, 0.30, 0.50)
    ]
    assert all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))
