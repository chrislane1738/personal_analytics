"""Structured logging configuration for the trading bot CLI."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging to stderr.

    Parameters
    ----------
    level:
        Log level name, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
        Defaults to ``"INFO"``.  Unknown values fall back to INFO.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
