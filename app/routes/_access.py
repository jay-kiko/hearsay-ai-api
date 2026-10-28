"""Access-code checks and upstream-error mapping shared by every route that
spends the server's Anthropic key. Previously copy-pasted into each route,
with slightly different messages each time."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from anthropic import AnthropicError, AuthenticationError
from fastapi import HTTPException, Request

from app.code_store import get_code_store
from app.config import get_settings
from app.errors import user_facing_error
from app.rate_limit import ensure_not_throttled, record_failure

logger = logging.getLogger("hearsay.routes")

_STATUS_ERRORS = {
    "unknown": (404, "Access code not found"),
    "revoked": (403, "This access code has been revoked"),
    "exhausted": (403, "This access code has no uses remaining"),
    "rate_limited": (429, "Too many AI generation calls for this access code"),
}


def raise_for_status(status: str, request: Request | None = None) -> None:
    if status == "unknown" and request is not None:
        record_failure(request)
    error = _STATUS_ERRORS.get(status)
    if error:
        raise HTTPException(status_code=error[0], detail=error[1])


async def charge_generation_call(access_code: str, request: Request) -> None:
    """The pre-spend throttle: every billed call made before a job is launched
    (detect, categories, personas, prompts) draws on one shared per-code
    budget, validated but not spending a use."""
    ensure_not_throttled(request)
    raise_for_status(
        await get_code_store().register_prompt_call(access_code, get_settings().max_prompt_calls_per_code), request
    )


async def require_usable_code(access_code: str, request: Request) -> None:
    ensure_not_throttled(request)
    status = await get_code_store().status(access_code)
    raise_for_status("ok" if status == "valid" else status, request)


@contextmanager
def upstream_errors() -> Iterator[None]:
    """Maps Anthropic failures to HTTP errors with a fixed message; the
    exception text (request ids, internals) goes to the log only."""
    try:
        yield
    except AuthenticationError as exc:
        logger.error("Anthropic rejected the server key: %s", exc)
        raise HTTPException(status_code=500, detail=user_facing_error(exc)) from exc
    except AnthropicError as exc:
        logger.warning("Anthropic call failed: %s", exc)
        raise HTTPException(status_code=502, detail=user_facing_error(exc)) from exc
