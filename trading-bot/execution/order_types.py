"""Order action and type enumerations for the execution layer."""

from enum import Enum


class OrderAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    BTO = "BTO"    # Buy to Open (options)
    STO = "STO"    # Sell to Open (options)
    BTC = "BTC"    # Buy to Close (options)
    STC = "STC"    # Sell to Close (options)


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
