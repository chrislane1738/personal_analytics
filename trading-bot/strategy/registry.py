"""Strategy registry: resolve strategy references by module path or YAML config."""

from __future__ import annotations

import importlib
from pathlib import Path

from strategy.base import Strategy


class StrategyRegistry:
    """Resolve strategy references by YAML path or Python module path."""

    @staticmethod
    def resolve(strategy_ref: str) -> Strategy:
        """Resolve a strategy reference.

        - If ends with .yaml/.yml → load as config strategy (Task 9, return None for now)
        - If looks like a module path (dots, no file ext) → import and instantiate
        - Raise ValueError if can't resolve
        """
        if strategy_ref.endswith((".yaml", ".yml")):
            # Will be handled by ConfigStrategy in Task 9
            # For now, just verify the file exists
            path = Path(strategy_ref)
            if not path.exists():
                raise FileNotFoundError(f"Strategy config not found: {strategy_ref}")
            return None  # placeholder until Task 9

        # Python module path: e.g., "strategy.library.momentum"
        try:
            parts = strategy_ref.rsplit(".", 1)
            if len(parts) == 2:
                module_path, class_name = parts[0], parts[1]
            else:
                raise ValueError(f"Invalid strategy reference: {strategy_ref}")

            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            if not (isinstance(cls, type) and issubclass(cls, Strategy)):
                raise TypeError(f"{class_name} is not a Strategy subclass")

            return cls()
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot resolve strategy: {strategy_ref}") from e
