"""
Small, dependency-free helpers for running Reverra on the public internet.

Everything here is plain Python (no FastAPI imports) so it can be unit-tested
on its own — see tests/test_security.py.
"""
import hmac
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple


def env_int(name: str, default: int, minimum: int = 1) -> int:
    """Reads an integer environment variable; falls back to `default` if it is
    unset, blank or not a number. Never returns less than `minimum`."""
    raw = os.environ.get(name)
    try:
        value = int(raw.strip()) if raw is not None and raw.strip() != "" else default
    except ValueError:
        value = default
    return max(minimum, value)


def parse_origins(value: Optional[str], public_mode: bool = False) -> list:
    """Turns the CORS_ORIGINS setting into the list CORSMiddleware wants.

      unset / blank -> ["*"] for local development; [] (same-origin only) in public mode
      "none"        -> []   (same-origin only: the single-server deployment)
      "*"           -> ["*"]
      "a.com,b.com" -> ["a.com", "b.com"]   (trailing slashes removed)
    """
    if value is None or value.strip() == "":
        return [] if public_mode else ["*"]
    text = value.strip()
    if text.lower() == "none":
        return []
    if text == "*":
        return ["*"]
    return [o.strip().rstrip("/") for o in text.split(",") if o.strip()]


def token_ok(provided: Optional[str], expected: str) -> bool:
    """Constant-time token comparison. False if either side is empty."""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def client_ip(forwarded_for: Optional[str], peer: Optional[str], trusted_hops: int = 1) -> str:
    """Best-effort visitor identity for rate limiting.

    Behind a proxy (Render, Cloudflare...) every connection's `peer` is the proxy,
    so we read X-Forwarded-For instead. Each trusted proxy APPENDS the address it
    saw, so the entry `trusted_hops` places from the right is the one our own
    infrastructure vouches for; anything further left can be forged by the client.
    """
    if forwarded_for and trusted_hops >= 1:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if len(parts) >= trusted_hops:
            return parts[-trusted_hops]
    return peer or "unknown"


class SlidingWindowLimiter:
    """At most `limit` events per `window` seconds, tracked per key (thread-safe)."""

    def __init__(self, limit: int, window_seconds: float, max_keys: int = 10_000,
                 clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self.max_keys = max_keys
        self._clock = clock
        self._events: dict = {}
        self._lock = threading.Lock()

    def _live(self, key: str, now: float) -> deque:
        q = self._events.get(key)
        if q is None:
            return deque()
        while q and q[0] <= now - self.window:
            q.popleft()
        if not q:
            del self._events[key]
            return deque()
        return q

    def _retry_after(self, q: deque, now: float) -> float:
        return max(0.0, q[0] + self.window - now) if q else 0.0

    def peek(self, key: str) -> Tuple[bool, float]:
        """Would one more event be allowed? Does not record anything."""
        with self._lock:
            now = self._clock()
            q = self._live(key, now)
            if len(q) >= self.limit:
                return False, self._retry_after(q, now)
            return True, 0.0

    def record(self, key: str) -> None:
        """Counts one event for `key`."""
        with self._lock:
            now = self._clock()
            q = self._live(key, now)
            if not q:
                self._events[key] = q = deque()
                self._evict_if_needed()
            q.append(now)

    def allow(self, key: str) -> Tuple[bool, float]:
        """Checks and, if allowed, records the event in one atomic step.
        Returns (allowed, seconds_until_next_slot)."""
        with self._lock:
            now = self._clock()
            q = self._live(key, now)
            if len(q) >= self.limit:
                return False, self._retry_after(q, now)
            if not q:
                self._events[key] = q = deque()
                self._evict_if_needed()
            q.append(now)
            return True, 0.0

    def _evict_if_needed(self) -> None:
        # Memory stays bounded even if someone floods us with distinct keys.
        if len(self._events) > self.max_keys:
            keep = max(1, int(self.max_keys * 0.9))
            for k in list(self._events)[: len(self._events) - keep]:   # oldest keys first
                del self._events[k]


class DailyCap:
    """A global budget of `limit` events per UTC day (thread-safe)."""

    def __init__(self, limit: int,
                 today: Callable[[], object] = lambda: datetime.now(timezone.utc).date()):
        self.limit = limit
        self._today = today
        self._day = today()
        self._count = 0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            day = self._today()
            if day != self._day:
                self._day, self._count = day, 0
            if self._count >= self.limit:
                return False
            self._count += 1
            return True
