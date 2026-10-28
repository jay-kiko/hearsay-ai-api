"""Thin wrapper around the Anthropic SDK.

The key is the server's own ANTHROPIC_API_KEY (see app.config) — there are no
accounts, so access is gated by the multi-use codes in app.code_store rather
than by a per-user key.

One client per key and event loop, shared by every call, so connections are
reused instead of a TLS handshake per call. Its own retries are off: retrying
is done once, here, in _with_retry (the SDK's 2 retries used to stack under
ours). An explicit timeout replaces the SDK's 10-minute default, which let one
hung call hold a job for 10+ minutes per attempt. A process-wide semaphore caps
calls in flight across all jobs, since each job alone fans out to dozens.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from anthropic import (
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from app.config import get_settings

logger = logging.getLogger("hearsay.anthropic")

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 6}

# A 429 needs to be treated differently from a generic failure: Anthropic's
# own guidance for "concurrent requests across all models exceeded" is to
# wait for in-flight requests to clear, not retry immediately — confirmed
# live, a job with several personas running concurrently (each doing 2 calls
# at once) hit this org-level ceiling and lost 7 of 8 personas because the
# generic path only gets `anthropic_max_retries` attempts with a ~1s flat
# backoff, nowhere near enough to ride out the window. Rate limits get their
# own, much more patient retry budget instead.
_RATE_LIMIT_MAX_RETRIES = 5
_RATE_LIMIT_BASE_DELAY_SECONDS = 5.0
_RATE_LIMIT_MAX_DELAY_SECONDS = 60.0


# Requests that fail the same way every time: retrying only costs time.
_NOT_RETRYABLE = (AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError, UnprocessableEntityError)

_clients: dict[tuple[int, str], AsyncAnthropic] = {}
_semaphores: dict[int, asyncio.Semaphore] = {}


def _client(api_key: str) -> AsyncAnthropic:
    # Keyed by loop too: the client's connection pool belongs to the loop
    # that opened it (one loop in production; tests open one per test).
    key = (id(asyncio.get_running_loop()), api_key)
    if key not in _clients:
        settings = get_settings()
        _clients[key] = AsyncAnthropic(api_key=api_key, timeout=settings.anthropic_timeout_seconds, max_retries=0)
    return _clients[key]


def _in_flight() -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _semaphores:
        _semaphores[loop_id] = asyncio.Semaphore(get_settings().anthropic_max_concurrency)
    return _semaphores[loop_id]


async def close_clients() -> None:
    for client in list(_clients.values()):
        await client.close()
    _clients.clear()
    _semaphores.clear()


def _cacheable_system(system: str) -> list[dict[str, Any]]:
    """Marks the system prompt as an Anthropic prompt-cache breakpoint.

    Every persona/prompt call within one job reuses the same answer/sentiment
    system prompt (only the brand name varies), so a job's ~dozens of calls
    hit this cache repeatedly instead of paying full input-token price each
    time. Below Anthropic's per-model minimum cacheable length this is simply
    a no-op — cache_control just doesn't engage — so it's safe to apply
    unconditionally rather than sizing every system prompt by hand.
    """
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


async def _with_retry(fn, *, retries: int, label: str):
    attempt = 0
    rate_limit_attempt = 0
    while True:
        try:
            async with _in_flight():
                return await fn()
        except _NOT_RETRYABLE:
            raise
        except RateLimitError as exc:
            rate_limit_attempt += 1
            if rate_limit_attempt > _RATE_LIMIT_MAX_RETRIES:
                logger.warning(
                    "%s rate-limited, giving up after %d attempt(s): %s", label, rate_limit_attempt, exc
                )
                raise
            delay = min(
                _RATE_LIMIT_BASE_DELAY_SECONDS * (2 ** (rate_limit_attempt - 1)), _RATE_LIMIT_MAX_DELAY_SECONDS
            )
            logger.info(
                "%s rate-limited (attempt %d/%d), backing off %.1fs: %s",
                label, rate_limit_attempt, _RATE_LIMIT_MAX_RETRIES, delay, exc,
            )
            await asyncio.sleep(delay)
        except Exception as exc:  # noqa: BLE001 - surfaced to caller after retries exhausted
            attempt += 1
            if attempt > retries:
                logger.warning("%s failed after %d attempt(s): %s", label, attempt, exc)
                raise
            logger.info("%s failed (attempt %d), retrying: %s", label, attempt, exc)
            await asyncio.sleep(1.0 * attempt)


async def call_text(*, api_key: str, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
    settings = get_settings()
    client = _client(api_key)

    async def _do() -> str:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_cacheable_system(system),
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    return await _with_retry(_do, retries=settings.anthropic_max_retries, label="call_text")


# Some models (confirmed live for the quality tier) reject a forced tool call
# with a 400 — "tool_choice: type 'tool' and 'any' are not supported for this
# model". Remember which ones so only the first call per model pays for the
# failed attempt.
_NO_FORCED_TOOL_CHOICE: set[str] = set()
_AUTO_TOOL_NUDGE = (
    "\n\nYou must reply by calling the provided tool exactly once, with every required field filled. "
    "Do not answer in plain text."
)


def _rejects_forced_tool_choice(exc: BadRequestError) -> bool:
    return "tool_choice" in str(exc)


async def call_structured(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
    max_tokens: int = 1536,
) -> dict[str, Any]:
    """Return a tool call's input as guaranteed-shape JSON.

    Forces the tool call where the model allows it; for models that reject a
    forced tool_choice, falls back to tool_choice=auto plus an instruction to
    call the tool, and raises if the model still doesn't.
    """
    settings = get_settings()
    client = _client(api_key)

    async def _request(forced: bool):
        return await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_cacheable_system(system if forced else system + _AUTO_TOOL_NUDGE),
            messages=[{"role": "user", "content": user}],
            tools=[{"name": tool_name, "description": tool_description, "input_schema": input_schema}],
            tool_choice={"type": "tool", "name": tool_name} if forced else {"type": "auto"},
        )

    async def _do() -> dict[str, Any]:
        forced = model not in _NO_FORCED_TOOL_CHOICE
        try:
            response = await _request(forced)
        except BadRequestError as exc:
            if not forced or not _rejects_forced_tool_choice(exc):
                raise
            logger.info("%s rejects forced tool_choice, falling back to auto", model)
            _NO_FORCED_TOOL_CHOICE.add(model)
            response = await _request(False)
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input
        raise ValueError(f"model did not return a {tool_name} tool call")

    return await _with_retry(_do, retries=settings.anthropic_max_retries, label=f"call_structured:{tool_name}")


# Search localisation only supports some countries (confirmed live: "Country
# code SG is not supported" 400s the whole request). A safety net for markets
# grounding._location_for doesn't already handle: unsupported codes are learned
# from the error and searched without localisation.
_UNSUPPORTED_COUNTRIES: set[str] = set()
# A long server-side search turn can come back paused (stop_reason
# "pause_turn"); it is resumed by sending the paused content back.
_MAX_PAUSE_RESUMES = 2


async def call_web_search(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1536,
    max_uses: int | None = None,
    user_location: dict | None = None,
):
    """Runs a Claude call with the native web_search tool enabled and returns the
    response. A paused turn is resumed, and the returned response carries the
    content blocks of every part of the turn."""
    settings = get_settings()
    client = _client(api_key)

    async def _do():
        country = (user_location or {}).get("country")
        use_location = bool(user_location) and country not in _UNSUPPORTED_COUNTRIES
        tool = dict(WEB_SEARCH_TOOL)
        if max_uses is not None:
            tool["max_uses"] = max_uses
        if use_location:
            tool["user_location"] = user_location

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        blocks: list = []
        for _ in range(_MAX_PAUSE_RESUMES + 1):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=_cacheable_system(system),
                    messages=messages,
                    tools=[tool],
                )
            except BadRequestError as exc:
                if "user_location" not in tool or "Country code" not in str(exc):
                    raise
                logger.info("search localisation unsupported for %s; searching without it", user_location)
                if country:
                    _UNSUPPORTED_COUNTRIES.add(country)
                tool = {k: v for k, v in tool.items() if k != "user_location"}
                continue
            blocks.extend(response.content)
            if response.stop_reason != "pause_turn":
                break
            messages = [*messages, {"role": "assistant", "content": response.content}]
        if hasattr(response, "model_copy"):
            return response.model_copy(update={"content": blocks})
        response.content = blocks
        return response

    return await _with_retry(_do, retries=settings.anthropic_max_retries, label="call_web_search")
