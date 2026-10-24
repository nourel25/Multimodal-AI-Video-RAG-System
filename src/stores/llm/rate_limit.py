"""Client-side throttling so free-tier API quotas aren't burned by bursts of calls."""

import threading
import time

_registry: dict[str, "RateLimiter"] = {}
_registry_lock = threading.Lock()


class RateLimiter:
    """Spaces calls evenly to stay under a requests-per-minute budget (thread-safe).

    Every caller reserves the next free slot, so concurrent threads queue up instead of
    firing together. A budget of 0 disables throttling.
    """

    def __init__(self, requests_per_minute: float, clock=time.monotonic, sleep=time.sleep):
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until this call may run; returns the seconds waited."""
        if not self.interval:
            return 0.0
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.interval
        wait = slot - now
        if wait > 0:
            self._sleep(wait)
        return wait


def shared_limiter(key: str, requests_per_minute: float) -> RateLimiter:
    """One limiter per API key/model, shared by every client instance in the process."""
    with _registry_lock:
        limiter = _registry.get(key)
        if limiter is None or limiter.interval != (
                60.0 / requests_per_minute if requests_per_minute > 0 else 0.0):
            limiter = _registry[key] = RateLimiter(requests_per_minute)
        return limiter
