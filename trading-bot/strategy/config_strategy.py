"""
Config-driven strategy that loads indicator definitions, entry rules, and exit
rules from a YAML file and uses the expression parser to evaluate conditions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from core.events import BarEvent, SignalEvent
from indicators.base import Indicator
from indicators.technical import SMA, EMA, RSI, MACD, BollingerBands
from strategy.base import Strategy
from strategy.expression_parser import evaluate_ast, parse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indicator factory
# ---------------------------------------------------------------------------

_INDICATOR_MAP: dict[str, type[Indicator]] = {
    "SMA": SMA,
    "EMA": EMA,
    "RSI": RSI,
    "MACD": MACD,
    "BollingerBands": BollingerBands,
}


def create_indicator(type_name: str, spec: dict[str, Any]) -> Indicator:
    """Instantiate an indicator from a type name and parameter dict.

    The ``spec`` dict may contain extra keys like ``type`` which are stripped
    before being forwarded to the indicator constructor.
    """
    cls = _INDICATOR_MAP.get(type_name)
    if cls is None:
        raise ValueError(
            f"Unknown indicator type {type_name!r}. "
            f"Available: {sorted(_INDICATOR_MAP)}"
        )

    # Build constructor kwargs — exclude the ``type`` key itself
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# ConfigStrategy
# ---------------------------------------------------------------------------

class ConfigStrategy(Strategy):
    """A strategy whose rules are defined entirely in a YAML config file."""

    def __init__(self, config_path: str) -> None:
        super().__init__()

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Strategy config not found: {config_path}")

        with open(path) as f:
            self.config: dict = yaml.safe_load(f)

        self.name: str = self.config["name"]
        self.universe: list[str] = self.config.get("universe", [])

        self._build_indicators()
        self._parse_rules()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_indicators(self) -> None:
        """Create indicator instances from the ``indicators`` config section."""
        for name, spec in self.config.get("indicators", {}).items():
            indicator_type = spec["type"]
            self.indicators[name] = create_indicator(indicator_type, spec)

    def _parse_rules(self) -> None:
        """Pre-parse entry/exit rule conditions into ASTs.

        Parsing happens at load time so that syntax errors are caught
        immediately rather than at run time.
        """
        self.entry_rules: list[dict] = []
        for rule in self.config.get("entry_rules", []):
            ast = parse(rule["condition"])
            self.entry_rules.append({
                "ast": ast,
                "direction": rule["direction"],
                "reason": rule["reason"],
            })

        self.exit_rules: list[dict] = []
        for rule in self.config.get("exit_rules", []):
            ast = parse(rule["condition"])
            self.exit_rules.append({
                "ast": ast,
                "reason": rule["reason"],
            })

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def generate_signals(self, bar: BarEvent, portfolio) -> list[SignalEvent]:
        """Evaluate entry/exit rules against current bar and portfolio state."""

        # 1. Update all indicators with the close price
        for ind in self.indicators.values():
            ind.update(bar.close)

        # 2. If not all indicators are ready, return early
        if not all(ind.is_ready for ind in self.indicators.values()):
            return []

        # 3. Build evaluation context
        context = self._build_context(bar, portfolio)

        signals: list[SignalEvent] = []

        # 4. Check entry rules (only if no position)
        has_position = portfolio is not None and portfolio.has_position(bar.symbol)

        if not has_position:
            for rule in self.entry_rules:
                result = evaluate_ast(rule["ast"], context)
                if result is True:
                    signals.append(SignalEvent(
                        symbol=bar.symbol,
                        direction=rule["direction"],
                        reason=rule["reason"],
                        strength=1.0,
                    ))

        # 5. Check exit rules (only if has position)
        if has_position:
            for rule in self.exit_rules:
                result = evaluate_ast(rule["ast"], context)
                if result is True:
                    signals.append(SignalEvent(
                        symbol=bar.symbol,
                        direction="short",  # exit = sell
                        reason=rule["reason"],
                        strength=1.0,
                    ))

        return signals

    def warm_up_period(self) -> int:
        """Return the maximum warm-up period across all indicators."""
        if not self.indicators:
            return 0
        return max(ind.warm_up_period for ind in self.indicators.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, bar: BarEvent, portfolio) -> dict:
        """Build the evaluation context dict from bar data, indicators, and portfolio."""
        context: dict = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "adj_close": bar.adj_close,
            "volume": bar.volume,
            "vwap": bar.vwap if bar.vwap is not None else 0.0,
        }

        # Add indicator objects — the expression evaluator resolves .value
        # automatically for simple comparisons (e.g., ``rsi_14 < 30``) and
        # handles attribute access for dot notation (e.g., ``bb_20.lower``).
        for name, ind in self.indicators.items():
            context[name] = ind

        # Add portfolio context if there is an active position
        if portfolio is not None and portfolio.has_position(bar.symbol):
            pos = portfolio.positions[bar.symbol]
            avg_cost = pos.avg_cost
            if avg_cost != 0:
                context["pnl_pct"] = (bar.close - avg_cost) / avg_cost
            else:
                context["pnl_pct"] = 0.0
            context["position_size"] = abs(pos.quantity) * bar.close
            context["holding_days"] = 0  # simplified for now

        return context
