"""Trading session utilities for Topstep time-of-day rules.

Topstep requires all positions to be flat by 3:10 PM CT (4:10 PM ET).
If a trader holds positions past this deadline, Topstep auto-liquidates.

These utilities are infrastructure for when intraday bars arrive.  With
daily bars the engine uses the ``force_flat_daily`` flag in
:class:`BacktestEngine` to close all positions at the end of each day's
bar processing.
"""

from datetime import time

# Topstep requires all positions flat by 3:10 PM CT = 4:10 PM ET
TOPSTEP_FLAT_DEADLINE_ET = time(16, 10)

# Regular trading hours (Eastern Time)
RTH_OPEN_ET = time(9, 30)
RTH_CLOSE_ET = time(16, 0)


def is_past_flat_deadline(current_time: time) -> bool:
    """Check if *current_time* (ET) is at or past Topstep's flat-by deadline.

    Parameters
    ----------
    current_time:
        A :class:`datetime.time` in Eastern Time.

    Returns
    -------
    bool
        ``True`` when ``current_time >= 16:10 ET``.
    """
    return current_time >= TOPSTEP_FLAT_DEADLINE_ET


def minutes_until_flat_deadline(current_time: time) -> int:
    """Minutes remaining until the flat deadline.

    Returns a negative value when *current_time* is already past the
    deadline.

    Parameters
    ----------
    current_time:
        A :class:`datetime.time` in Eastern Time.

    Returns
    -------
    int
        Minutes until 16:10 ET.  Negative if the deadline has passed.
    """
    deadline_minutes = TOPSTEP_FLAT_DEADLINE_ET.hour * 60 + TOPSTEP_FLAT_DEADLINE_ET.minute
    current_minutes = current_time.hour * 60 + current_time.minute
    return deadline_minutes - current_minutes
