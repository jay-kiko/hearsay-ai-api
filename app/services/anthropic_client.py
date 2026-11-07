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
import json
import logging
import weakref
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from anthropic import (
    AsyncAnthropic,
    APITimeoutError,
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

# Keyed by the loop object itself, weakly: the client's connection pool and
# the semaphore belong to the loop that made them. An id(loop) key could be
# reused by a new loop after the old one was collected (tests open one per
# test) and hand back objects bound to a closed loop.
_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, AsyncAnthropic]] = weakref.WeakKeyDictionary()
_semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()


def _client(api_key: str) -> AsyncAnthropic:
    per_loop = _clients.setdefault(asyncio.get_running_loop(), {})
    if api_key not in per_loop:
        settings = get_settings()
        per_loop[api_key] = AsyncAnthropic(api_key=api_key, timeout=settings.anthropic_timeout_seconds, max_retries=0)
    return per_loop[api_key]


def _in_flight() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    if loop not in _semaphores:
        _semaphores[loop] = asyncio.Semaphore(get_settings().anthropic_max_concurrency)
    return _semaphores[loop]


async def close_clients() -> None:
    """At shutdown. The semaphore is left alone: calls still in flight hold
    its slots, and a fresh one would let new calls ignore them."""
    for per_loop in list(_clients.values()):
        for client in per_loop.values():
            await client.close()
    _clients.clear()


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


async def _with_retry(fn, *, retries: int, label: str, retry_timeouts: bool = True):
    attempt = 0
    rate_limit_attempt = 0
    while True:
        try:
            async with _in_flight():
                return await fn()
        except _NOT_RETRYABLE:
            raise
        except APITimeoutError:
            if not retry_timeouts:
                raise
            attempt += 1
            if attempt > retries:
                logger.warning("%s timed out after %d attempt(s)", label, attempt)
                raise
            logger.info("%s timed out (attempt %d), retrying", label, attempt)
            await asyncio.sleep(1.0 * attempt)
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


async def call_text(*, api_key: str, model: str, system: str, user: str, max_tokens: int = 8000) -> str:
    """The measured answer. A roomy, fixed ceiling: the standard tier thinks,
    and thinking counts against max_tokens (answers measured ~370 tokens with
    it). A reply cut off at the ceiling is a failed attempt, never scored: a
    truncated answer would read as "brand not mentioned"."""
    settings = get_settings()
    client = _client(api_key)

    async def _do() -> str:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_cacheable_system(system),
            messages=[{"role": "user", "content": user}],
        )
        if getattr(response, "stop_reason", None) == "max_tokens":
            logger.warning("call_text stopped at the %d-token ceiling; treating it as a failed call", max_tokens)
            raise ValueError(f"answer stopped at max_tokens={max_tokens}")
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


# One ceiling for every structured call, not a per-call guess at the reply's
# size: thinking counts against max_tokens, and tight per-call limits cut
# replies off (6 personas x 3 prompts spent all of 2048 thinking, and every
# persona got the fallback question). Only tokens used are billed. Measured
# live, the largest reply the app can ask for (10 personas x 10 prompts) used
# 7.9-9.1k; 32k leaves ~3.5x, and caps a runaway reply at ~$0.64 on Opus.
_STRUCTURED_MAX_TOKENS = 32_000


def _rejects_forced_tool_choice(exc: BadRequestError) -> bool:
    return "tool_choice" in str(exc)


def _decode_stringified(value: dict[str, Any], input_schema: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """Confirmed live (standard tier, ~1 in 4 source-enrichment calls): the
    model sometimes sends a top-level array or object field as a JSON-encoded
    string. Callers iterate it as a list, find nothing, and silently degrade —
    grounding then showed no sources at all. The string can also hold the
    whole input again ('{"sources": [...]}' inside "sources", ~1 in 10), so
    that wrapper is unwrapped. A string that doesn't decode to the declared
    type is left as is, so each caller's own fallback applies."""
    properties = input_schema.get("properties", {})
    decoded = dict(value)
    for key, raw in value.items():
        expected = properties.get(key, {}).get("type")
        if expected not in ("array", "object"):
            continue
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except ValueError:
                continue
            if expected == "array" and isinstance(parsed, dict) and isinstance(parsed.get(key), list):
                parsed = parsed[key]
            if not isinstance(parsed, list if expected == "array" else dict):
                continue
            logger.info("decoded %s.%s, sent as a JSON string", tool_name, key)
            raw = parsed
        if expected == "array" and properties[key].get("items", {}).get("type") == "object":
            raw = _decode_items(raw, tool_name, key)
        decoded[key] = raw
    return decoded


def _decode_items(items: list, tool_name: str, key: str) -> list:
    """Confirmed live (quality tier, persona generation): the array arrived
    fine but each object in it was a JSON-encoded string, and the caller's
    `entry.get` crashed the request. Items that decode to an object are
    replaced; any other item is left for the caller's own checks."""
    out, fixed = [], 0
    for item in items:
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                item, fixed = parsed, fixed + 1
        out.append(item)
    if fixed:
        logger.info("decoded %d item(s) of %s.%s, sent as JSON strings", fixed, tool_name, key)
    return out


def _check_required_shapes(value: dict[str, Any], input_schema: dict[str, Any], tool_name: str) -> None:
    """Raises (so _with_retry makes a fresh call) when a required array or
    object field is still missing or the wrong type after decoding: a shape
    nobody has seen yet must not reach a caller as an empty result."""
    properties = input_schema.get("properties", {})
    for key in input_schema.get("required", []):
        expected = properties.get(key, {}).get("type")
        if expected in ("array", "object") and not isinstance(value.get(key), list if expected == "array" else dict):
            raise ValueError(f"{tool_name}.{key} is not a JSON {expected}")


async def call_structured(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
    max_tokens: int | None = None,
    retry_malformed: bool = False,
) -> dict[str, Any]:
    """Return a tool call's input as guaranteed-shape JSON.

    Forces the tool call where the model allows it; for models that reject a
    forced tool_choice, falls back to tool_choice=auto plus an instruction to
    call the tool, and raises if the model still doesn't.

    The reply is streamed so its ceiling can sit above the SDK's limit for a
    non-streaming request (~21k tokens); callers get the finished reply as
    before. `max_tokens` defaults to one generous ceiling, not a guess at the
    reply's size (see _STRUCTURED_MAX_TOKENS).

    `retry_malformed` retries a reply whose required array/object fields are
    still the wrong shape after decoding. Opt-in: detect parses comma-joined
    strings for its list fields itself, and would fail where it now recovers.
    """
    settings = get_settings()
    client = _client(api_key)
    budget = max_tokens or _STRUCTURED_MAX_TOKENS

    async def _request(forced: bool):
        async with client.messages.stream(
            model=model,
            max_tokens=budget,
            system=_cacheable_system(system if forced else system + _AUTO_TOOL_NUDGE),
            messages=[{"role": "user", "content": user}],
            tools=[{"name": tool_name, "description": tool_description, "input_schema": input_schema}],
            tool_choice={"type": "tool", "name": tool_name} if forced else {"type": "auto"},
        ) as stream:
            return await stream.get_final_message()

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
        if getattr(response, "stop_reason", None) == "max_tokens":
            # Far past any real reply (the largest measured, 100 prompts with
            # thinking, used ~9k): the model ran away, so this attempt failed
            # and _with_retry makes a fresh one.
            logger.warning("%s stopped at the %d-token ceiling; treating it as a failed call", tool_name, budget)
            raise ValueError(f"{tool_name} reply stopped at max_tokens={budget}")
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                value = _decode_stringified(block.input, input_schema, tool_name)
                if retry_malformed:
                    _check_required_shapes(value, input_schema, tool_name)
                return value
        raise ValueError(f"model did not return a {tool_name} tool call")

    return await _with_retry(_do, retries=settings.anthropic_max_retries, label=f"call_structured:{tool_name}")


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
):
    """Runs a Claude call with the native web_search tool enabled and returns the
    response. A paused turn is resumed, and the returned response carries the
    content blocks of every part of the turn."""
    settings = get_settings()
    client = _client(api_key)

    async def _do():
        tool = dict(WEB_SEARCH_TOOL)
        if max_uses is not None:
            tool["max_uses"] = max_uses

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        blocks: list = []
        for _ in range(_MAX_PAUSE_RESUMES + 1):
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=_cacheable_system(system),
                messages=messages,
                tools=[tool],
            )
            blocks.extend(response.content)
            if response.stop_reason != "pause_turn":
                break
            messages = [*messages, {"role": "assistant", "content": response.content}]
        if hasattr(response, "model_copy"):
            return response.model_copy(update={"content": blocks})
        response.content = blocks
        return response

    # One retry at most, and none after a timeout: every attempt re-runs the
    # whole multi-turn search and bills its searches again, and a search that
    # took the full timeout once will likely do it again (3 x 180s otherwise).
    return await _with_retry(
        _do, retries=min(settings.anthropic_max_retries, 1), label="call_web_search", retry_timeouts=False
    )


# ── Web fetch (reading pages that block our own fetcher) ─────────────

WEB_FETCH_TOOL = {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1}
_WEB_FETCH_BETA = "web-fetch-2025-09-10"
_WEB_FETCH_SYSTEM = "Fetch the given URL with the web_fetch tool exactly once, then reply only: done."


@dataclass
class FetchedPage:
    text: str
    # When Anthropic's fetcher retrieved this copy; it can serve a cached copy
    # (confirmed live: up to four months old), so callers show its age.
    retrieved_at: datetime | None


def _block_field(item, key: str):
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def _parse_fetch(response) -> FetchedPage | None:
    for block in getattr(response, "content", None) or []:
        if _block_field(block, "type") != "web_fetch_tool_result":
            continue
        result = _block_field(block, "content")
        if _block_field(result, "type") != "web_fetch_result":
            return None  # web_fetch_tool_error: url_not_accessible and the like
        source = _block_field(_block_field(result, "content"), "source")
        text = _block_field(source, "data")
        if not isinstance(text, str) or not text.strip():
            return None
        retrieved = _block_field(result, "retrieved_at")
        try:
            retrieved_at = datetime.fromisoformat(retrieved) if isinstance(retrieved, str) else retrieved
        except ValueError:
            retrieved_at = None
        return FetchedPage(text=text, retrieved_at=retrieved_at)
    return None


async def call_web_fetch(*, api_key: str, model: str, url: str, max_content_tokens: int) -> FetchedPage | None:
    """The text of `url` as read by Anthropic's web_fetch tool, or None when it
    couldn't be read. For pages that block our own fetcher (confirmed live:
    6 of 7 bot-blocked source pages read this way, about 1 cent each on the
    fast tier). The model's own reply is ignored; only the fetched text is
    used, and counting the brand in it stays deterministic.

    max_tokens must leave room for the tool call itself: at 64 the model
    replied without calling it (confirmed live)."""
    client = _client(api_key)
    tool = {**WEB_FETCH_TOOL, "max_content_tokens": max_content_tokens}

    async def _do() -> FetchedPage | None:
        response = await client.messages.create(
            model=model,
            max_tokens=300,
            system=_WEB_FETCH_SYSTEM,
            messages=[{"role": "user", "content": url}],
            tools=[tool],
            extra_headers={"anthropic-beta": _WEB_FETCH_BETA},
        )
        return _parse_fetch(response)

    # One attempt plus one retry, none after a timeout: a page that timed out
    # once will likely do so again, and each attempt bills the page's tokens.
    return await _with_retry(_do, retries=1, label="call_web_fetch", retry_timeouts=False)
