"""Token-bucket style rate limiter using a sliding window of timestamps."""
import threading
import time
from collections import deque


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Tracks the timestamps of the last N requests within a 60-second window.
    When the bucket is full, ``acquire()`` blocks until the oldest request
    slides out of the window.

    Parameters
    ----------
    max_requests_per_minute:
        Maximum number of requests allowed in any 60-second rolling window.
        Defaults to 700 (headroom below the FMP hard limit of 750).
    """

    def __init__(self, max_requests_per_minute: int = 700) -> None:
        self._max = max_requests_per_minute
        self._window = 60.0  # seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """Block until a request token is available, then consume it.

        Uses a busy-wait with short sleeps so the thread yields the GIL
        while waiting.  After returning the caller is guaranteed to be
        within the rate limit.
        """
        while True:
            with self._lock:
                now = time.monotonic()
                # Evict timestamps that are outside the rolling window
                cutoff = now - self._window
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                if len(self._timestamps) < self._max:
                    # Slot available — consume it and return immediately
                    self._timestamps.append(now)
                    return

                # Bucket full — compute how long until the oldest slot frees up
                wait_for = self._timestamps[0] - cutoff

            # Sleep outside the lock so other threads can make progress
            time.sleep(min(wait_for, 0.05))

    # ------------------------------------------------------------------
    # Inspection helpers (useful for testing)
    # ------------------------------------------------------------------

    @property
    def current_count(self) -> int:
        """Number of requests recorded in the current window (approximate)."""
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._window
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            return len(self._timestamps)

    def reset(self) -> None:
        """Clear all recorded timestamps.  Intended for testing only."""
        with self._lock:
            self._timestamps.clear()
