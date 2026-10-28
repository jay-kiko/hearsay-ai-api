"""Throttles guessing of access codes and the admin secret.

Codes carry ~40 bits and /api/access/status answered "unknown" vs "valid"
without limit, so codes (which spend the server's Anthropic key) could be
enumerated. Only *failed* lookups count, so people using a real code are never
slowed. Two windows:

  * per client IP — stops a single guesser;
  * global — the backstop if IPs are spoofed or rotated (X-Forwarded-For is
    only as trustworthy as the proxy config, see FORWARDED_ALLOW_IPS). While it
    trips, unknown-code lookups from everyone are refused, which costs a user
    who mistypes a code a short wait; the default caps an attacker at roughly
    150k guesses a day, decades for a 40-bit space.

The check runs *before* the lookup: a throttled request must not learn whether
its code exists. In memory, like the job store (one worker process only).
"""
from __future__ import annotations

import time
from collections import deque

from fastapi import HTTPException, Request

from app.config import get_settings

_MAX_TRACKED_KEYS = 10_000


class FailureLimiter:
    def __init__(self, *, per_key: int, global_limit: int, window_seconds: float) -> None:
        self._per_key = per_key
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
        return len(self._all) >= self._global_limit or (hits is not None and len(hits) >= self._per_key)

    def record_failure(self, key: str) -> None:
        now = time.monotonic()
        self._all.append(now)
        if key not in self._by_key and len(self._by_key) >= _MAX_TRACKED_KEYS:
            # Bounded memory under an IP-rotating flood; the global window
            # still counts every failure.
            self._by_key.pop(next(iter(self._by_key)))
        self._by_key.setdefault(key, deque()).append(now)


_limiter: FailureLimiter | None = None


def get_limiter() -> FailureLimiter:
    global _limiter
    if _limiter is None:
        s = get_settings()
        _limiter = FailureLimiter(
            per_key=s.code_failures_per_ip, global_limit=s.code_failures_global, window_seconds=s.code_failure_window_seconds
        )
    return _limiter


def client_key(request: Request, scope: str = "code") -> str:
    return f"{scope}:{request.client.host if request.client else 'unknown'}"


def ensure_not_throttled(request: Request, scope: str = "code") -> None:
    if get_limiter().blocked(client_key(request, scope)):
        raise HTTPException(status_code=429, detail="Too many invalid attempts. Try again in a few minutes.")
