"""Shared dependencies for the FastAPI backend."""

from __future__ import annotations

import os
from functools import lru_cache

from data.storage.database import Database


@lru_cache(maxsize=1)
def get_database() -> Database:
    """Return a singleton Database instance.

    The database path is read from the ``TRADING_BOT_DB`` environment
    variable, falling back to ``db/trading_bot.db`` relative to the
    project root.
    """
    db_path = os.environ.get("TRADING_BOT_DB", "db/trading_bot.db")
    db = Database(db_path)
    db.create_tables()
    return db
