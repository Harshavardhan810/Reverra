"""Tests for the public-deployment helpers (backend/security.py).
Run from the project root:  pytest -q"""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.security import (DailyCap, SlidingWindowLimiter, client_ip,  # noqa: E402
                              env_int, parse_origins, token_ok)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_token_ok():
    assert token_ok("s3cret-token-value", "s3cret-token-value")
    assert not token_ok("wrong", "s3cret-token-value")
    assert not token_ok("", "s3cret-token-value")
    assert not token_ok(None, "s3cret-token-value")
    assert not token_ok("anything", "")           # never match when no token is configured
    assert token_ok("tökén-ünïcode", "tökén-ünïcode")


def test_parse_origins():
    assert parse_origins(None) == ["*"]
    assert parse_origins("") == ["*"]
    assert parse_origins("   ") == ["*"]
    assert parse_origins(None, public_mode=True) == []
    assert parse_origins("none") == [] and parse_origins("NONE") == []
    assert parse_origins("*") == ["*"]
    assert parse_origins("https://a.com/, https://b.com") == ["https://a.com", "https://b.com"]
    assert parse_origins("none", public_mode=False) == []


def test_env_int():
    import os
    os.environ["REVERRA_T"] = "12"
    assert env_int("REVERRA_T", 5) == 12
    os.environ["REVERRA_T"] = "abc"
    assert env_int("REVERRA_T", 5) == 5
    os.environ["REVERRA_T"] = "  "
    assert env_int("REVERRA_T", 5) == 5
    os.environ["REVERRA_T"] = "-3"
    assert env_int("REVERRA_T", 5, minimum=1) == 1
    del os.environ["REVERRA_T"]
    assert env_int("REVERRA_T", 7) == 7


def test_client_ip_trusts_only_the_proxy_appended_entry():
    # a client forging the left side of X-Forwarded-For must not choose its own identity
    assert client_ip("6.6.6.6, 203.0.113.9", "10.0.0.1", 1) == "203.0.113.9"
    assert client_ip("203.0.113.9", "10.0.0.1", 1) == "203.0.113.9"
    assert client_ip("6.6.6.6, 203.0.113.9, 198.51.100.2", "10.0.0.1", 2) == "203.0.113.9"
    assert client_ip(None, "127.0.0.1") == "127.0.0.1"
    assert client_ip("", "127.0.0.1") == "127.0.0.1"
    assert client_ip(None, None) == "unknown"
    assert client_ip("1.1.1.1", "10.0.0.1", 3) == "10.0.0.1"   # fewer entries than trusted hops -> ignore header


def test_sliding_window_blocks_then_recovers():
    clock = FakeClock()
    lim = SlidingWindowLimiter(3, 60, clock=clock)
    for _ in range(3):
        assert lim.allow("a")[0]
    ok, retry = lim.allow("a")
    assert not ok and 0 < retry <= 60
    assert lim.allow("b")[0]                       # other visitors are unaffected
    clock.t += 61
    assert lim.allow("a")[0]                       # window slid past the old events


def test_peek_does_not_record_and_record_does():
    clock = FakeClock()
    lim = SlidingWindowLimiter(2, 60, clock=clock)
    for _ in range(5):
        assert lim.peek("x")[0]                    # peeking never uses up slots
    lim.record("x")
    lim.record("x")
    ok, retry = lim.peek("x")
    assert not ok and retry > 0


def test_blocked_attempts_are_not_recorded():
    clock = FakeClock()
    lim = SlidingWindowLimiter(1, 60, clock=clock)
    assert lim.allow("a")[0]
    for _ in range(10):
        assert not lim.allow("a")[0]
    clock.t += 61
    assert lim.allow("a")[0]                       # hammering while blocked didn't extend the block


def test_limiter_memory_is_bounded():
    lim = SlidingWindowLimiter(1, 3600, max_keys=10)
    for i in range(500):
        lim.allow(f"visitor-{i}")
    assert len(lim._events) <= 10
    assert not lim.allow("visitor-499")[0]         # the newest visitor is still tracked


def test_daily_cap_resets_each_utc_day():
    day = {"d": date(2026, 10, 4)}
    cap = DailyCap(2, today=lambda: day["d"])
    assert cap.allow() and cap.allow()
    assert not cap.allow()
    day["d"] += timedelta(days=1)
    assert cap.allow()
