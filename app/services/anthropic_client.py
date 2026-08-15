"""Thin wrapper around the Anthropic SDK.

The key is the server's own ANTHROPIC_API_KEY (see app.config) — there are no
accounts, so access is gated by the multi-use codes in app.code_store rather
than by a per-user key. Every call still builds a throwaway `AsyncAnthropic`
client rather than a shared one, which is only relevant now for isolating
retries/timeouts per call, not for key custody.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from anthropic import AsyncAnthropic

from app.config import get_settings

logger = logging.getLogger("hearsay.anthropic")

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 6}


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
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - surfaced to caller after retries exhausted
            attempt += 1
            if attempt > retries:
                logger.warning("%s failed after %d attempt(s): %s", label, attempt, exc)
                raise
            logger.info("%s failed (attempt %d), retrying: %s", label, attempt, exc)
            await asyncio.sleep(1.0 * attempt)


async def call_text(*, api_key: str, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
    settings = get_settings()
    client = AsyncAnthropic(api_key=api_key)

    async def _do() -> str:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_cacheable_system(system),
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    try:
        return await _with_retry(_do, retries=settings.anthropic_max_retries, label="call_text")
    finally:
        await client.close()


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
    """Force a tool call so the model's output is guaranteed-shape JSON."""
    settings = get_settings()
    client = AsyncAnthropic(api_key=api_key)

    async def _do() -> dict[str, Any]:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_cacheable_system(system),
            messages=[{"role": "user", "content": user}],
            tools=[{"name": tool_name, "description": tool_description, "input_schema": input_schema}],
            tool_choice={"type": "tool", "name": tool_name},
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input
        raise ValueError(f"model did not return a {tool_name} tool call")

    try:
        return await _with_retry(_do, retries=settings.anthropic_max_retries, label=f"call_structured:{tool_name}")
    finally:
        await client.close()


async def call_web_search(*, api_key: str, model: str, system: str, user: str, max_tokens: int = 1536):
    """Runs a Claude call with the native web_search tool enabled and returns the raw response."""
    settings = get_settings()
    client = AsyncAnthropic(api_key=api_key)

    async def _do():
        return await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_cacheable_system(system),
            messages=[{"role": "user", "content": user}],
            tools=[WEB_SEARCH_TOOL],
        )

    try:
        return await _with_retry(_do, retries=settings.anthropic_max_retries, label="call_web_search")
    finally:
        await client.close()
