"""Throttles guessing of access codes and the admin secret.

Codes carry ~40 bits and /api/access/status answered "unknown" vs "valid"
without limit, so codes (which spend the server's Anthropic key) could be
enumerated. Only *failed* lookups count, so people using a real code are never
slowed. Two windows:

  * per client — stops a single guesser. A client is an IPv4 address or an
    IPv6 /64, since one IPv6 host can rotate through a /64 for free;
  * global — when failures from everyone pass the global limit (a distributed
    guess), the per-client allowance drops to a handful. It never refuses a
    client that hasn't failed: an earlier version refused every lookup once
    the global window filled, so one attacker could lock out every user with
    a valid code, and the admin with them.

The check runs *before* the lookup: a throttled request must not learn whether
its code exists. Client IPs are only as real as the proxy config (see
FORWARDED_ALLOW_IPS); a spoofable X-Forwarded-For turns this into the global
brake alone. Admin-secret failures use their own limiter, so a code-guessing
flood can't lock the admin out. Longer codes would be the stronger fix against
distributed guessing (they need a frontend format change). In memory, like the
job store (one worker process only).
"""
from __future__ import annotations

import ipaddress
import time
from collections import deque

from fastapi import HTTPException, Request

from app.config import get_settings

_MAX_TRACKED_KEYS = 10_000


class FailureLimiter:
    def __init__(self, *, per_key: int, global_limit: int, window_seconds: float, tight_per_key: int = 3) -> None:
        self._per_key = per_key
        self._tight_per_key = tight_per_key
        self._global_limit = global_limit
        self._window = window_seconds
        self._by_key: dict[str, deque[float]] = {}
        self._all: deque[float] = deque()

    def _prune(self, hits: deque[float], now: float) -> None:
        while hits and now - hits[0] > self._window:
            hits.popleft()

    def blocked(self, key: str) -> bool:
        now = time.monotonic()
        self._prune(self._all, now)
        hits = self._by_key.get(key)
        if hits is not None:
            self._prune(hits, now)
            if not hits:
                del self._by_key[key]
        failures = len(hits) if hits is not None else 0
        limit = self._tight_per_key if len(self._all) >= self._global_limit else self._per_key
        return failures >= limit

    def record_failure(self, key: str) -> None:
        now = time.monotonic()
        self._all.append(now)
        if key not in self._by_key and len(self._by_key) >= _MAX_TRACKED_KEYS:
            # Bounded memory under an IP-rotating flood; the global window
            # still counts every failure.
            self._by_key.pop(next(iter(self._by_key)))
        self._by_key.setdefault(key, deque()).append(now)


_limiters: dict[str, FailureLimiter] = {}


def get_limiter(scope: str = "code") -> FailureLimiter:
    """One limiter per scope ("code", "admin"), so their global windows are separate."""
    if scope not in _limiters:
        s = get_settings()
        _limiters[scope] = FailureLimiter(
            per_key=s.code_failures_per_ip, global_limit=s.code_failures_global, window_seconds=s.code_failure_window_seconds
        )
    return _limiters[scope]


def client_key(request: Request) -> str:
    host = request.client.host if request.client else "unknown"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped:
            return str(address.ipv4_mapped)
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


def ensure_not_throttled(request: Request, scope: str = "code") -> None:
    if get_limiter(scope).blocked(client_key(request)):
        raise HTTPException(status_code=429, detail="Too many invalid attempts. Try again in a few minutes.")


def record_failure(request: Request, scope: str = "code") -> None:
    get_limiter(scope).record_failure(client_key(request))
