"""A simple thread-safe token-style rate limiter.

Capital.com documents a general limit around 10 requests/second per session
(and tighter limits on specific endpoints). This limiter spaces out calls to
stay under a configured requests-per-second ceiling.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, max_per_second: float = 9.0) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second must be positive")
        self._min_interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval
