from abc import ABC, abstractmethod


class Indicator(ABC):
    """Base class for all technical indicators."""

    @abstractmethod
    def update(self, price: float) -> None:
        """Feed a new price to the indicator."""

    @property
    @abstractmethod
    def value(self) -> float | None:
        """Current indicator value. None if not enough data (warm-up period)."""

    @property
    @abstractmethod
    def warm_up_period(self) -> int:
        """Number of data points needed before the indicator produces values."""

    @property
    def is_ready(self) -> bool:
        return self.value is not None
