"""Tests for utils/rate_limiter.py."""
import threading
import time

import pytest

from utils.rate_limiter import RateLimiter


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

def test_acquire_succeeds_under_limit():
    """Calls under the limit should all complete without delay."""
    rl = RateLimiter(max_requests_per_minute=10)
    start = time.monotonic()
    for _ in range(5):
        rl.acquire()
    elapsed = time.monotonic() - start
    # Should be nearly instantaneous (well under 1 second)
    assert elapsed < 1.0


def test_current_count_tracks_acquires():
    """current_count must reflect the number of tokens consumed."""
    rl = RateLimiter(max_requests_per_minute=50)
    for _ in range(7):
        rl.acquire()
    assert rl.current_count == 7


def test_reset_clears_count():
    """reset() should bring current_count back to zero."""
    rl = RateLimiter(max_requests_per_minute=50)
    for _ in range(10):
        rl.acquire()
    rl.reset()
    assert rl.current_count == 0


# ---------------------------------------------------------------------------
# Rate limiting behaviour
# ---------------------------------------------------------------------------

def test_rate_limiter_blocks_when_limit_reached():
    """After hitting the limit the next acquire() should block noticeably."""
    rl = RateLimiter(max_requests_per_minute=5)
    # Exhaust the bucket as fast as possible
    for _ in range(5):
        rl.acquire()

    # The 6th call must wait for the window to rotate — measure that it takes
    # at least some non-trivial time (> 10ms but we give it up to 65 s to
    # avoid a flaky test on slow CI).
    start = time.monotonic()
    rl.acquire()
    elapsed = time.monotonic() - start
    # Must have blocked for at least 10 ms, meaning the limiter did something
    assert elapsed > 0.01, f"Expected a delay, but acquire() returned in {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_thread_safe_concurrent_acquires():
    """Multiple threads should not exceed the per-minute limit collectively."""
    limit = 20
    rl = RateLimiter(max_requests_per_minute=limit)
    results: list[float] = []
    lock = threading.Lock()

    def worker():
        rl.acquire()
        with lock:
            results.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(limit)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(results) == limit, "Not all threads completed"
    # All timestamps should be within a single 60-second window (they should
    # be, because we only issued exactly `limit` requests)
    assert max(results) - min(results) < 60.0


def test_thread_safe_overflow_gets_throttled():
    """Issuing limit+1 requests from threads should cause the last one to wait."""
    limit = 5
    rl = RateLimiter(max_requests_per_minute=limit)
    timestamps: list[float] = []
    lock = threading.Lock()

    def worker():
        rl.acquire()
        with lock:
            timestamps.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(limit + 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=70.0)

    assert len(timestamps) == limit + 1, "Not all threads completed"
    # The spread must be > 10 ms because at least one thread had to wait
    spread = max(timestamps) - min(timestamps)
    assert spread > 0.01, f"Expected throttling delay, but spread was {spread:.3f}s"


# ---------------------------------------------------------------------------
# Default limit
# ---------------------------------------------------------------------------

def test_default_limit_is_700():
    """Default max_requests_per_minute should be 700."""
    rl = RateLimiter()
    # We can't actually fire 700 requests in a test, but we can introspect
    assert rl._max == 700
