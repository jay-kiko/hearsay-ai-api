from types import SimpleNamespace

import httpx
import pytest
from anthropic import BadRequestError

from app.services import anthropic_client as ac


def _bad_request():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request)
    return BadRequestError(
        'tool_choice: type "tool" and "any" are not supported for this model', response=response, body=None
    )


class _Stream:
    """messages.stream(...) as the SDK gives it: an async context manager whose
    get_final_message() is the finished reply."""

    def __init__(self, reply):
        self._reply = reply

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def get_final_message(self):
        return await self._reply


class _FakeClient:
    """Rejects forced tool_choice like the quality-tier model does."""

    calls: list[dict] = []

    def __init__(self, **_):
        self.messages = SimpleNamespace(create=self._create, stream=lambda **kw: _Stream(self._create(**kw)))

    async def _create(self, **kwargs):
        _FakeClient.calls.append(kwargs)
        if kwargs["tool_choice"]["type"] == "tool":
            raise _bad_request()
        block = SimpleNamespace(type="tool_use", name="t", input={"x": "ok"})
        return SimpleNamespace(content=[block])

    async def close(self):
        pass


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    _FakeClient.calls = []
    ac._NO_FORCED_TOOL_CHOICE.clear()
    monkeypatch.setattr(ac, "AsyncAnthropic", _FakeClient)
    ac._clients.clear()


async def _call(model="fancy-model"):
    return await ac.call_structured(
        api_key="k", model=model, system="sys", user="u", tool_name="t", tool_description="d",
        input_schema={"type": "object", "properties": {}},
    )


async def test_falls_back_to_auto_when_forced_tool_choice_rejected():
    assert await _call() == {"x": "ok"}
    assert [c["tool_choice"]["type"] for c in _FakeClient.calls] == ["tool", "auto"]
    assert "calling the provided tool" in _FakeClient.calls[1]["system"][0]["text"]


async def test_remembers_the_model_so_later_calls_skip_the_failed_attempt():
    await _call()
    _FakeClient.calls.clear()
    await _call()
    assert [c["tool_choice"]["type"] for c in _FakeClient.calls] == ["auto"]


def test_decodes_array_and_object_fields_sent_as_json_strings():
    # Confirmed live: source enrichment's `sources` array arrived as a string,
    # so grounding classified nothing and showed no sources.
    schema = {"type": "object", "properties": {
        "sources": {"type": "array"}, "meta": {"type": "object"}, "note": {"type": "string"},
    }}
    value = {"sources": '[{"domain": "a.com"}]', "meta": '{"k": 1}', "note": "[1]"}
    assert ac._decode_stringified(value, schema, "t") == {
        "sources": [{"domain": "a.com"}], "meta": {"k": 1}, "note": "[1]",
    }


def test_unwraps_the_whole_input_resent_inside_a_string_field():
    # Confirmed live: "sources" held '{"sources": [...]}', not the bare array.
    schema = {"type": "object", "properties": {"sources": {"type": "array"}}}
    value = {"sources": '{"sources": [{"domain": "a.com"}]}'}
    assert ac._decode_stringified(value, schema, "t") == {"sources": [{"domain": "a.com"}]}


def test_leaves_undecodable_or_mistyped_strings_alone():
    schema = {"type": "object", "properties": {"sources": {"type": "array"}, "meta": {"type": "object"}}}
    value = {"sources": "[{truncated", "meta": "[1, 2]"}
    assert ac._decode_stringified(value, schema, "t") == value


async def test_retries_when_a_required_array_is_still_malformed(monkeypatch):
    replies = [{"sources": "not json"}, {"sources": [{"domain": "a.com"}]}]

    class _Malformed(_FakeClient):
        async def _create(self, **kwargs):
            block = SimpleNamespace(type="tool_use", name="t", input=replies.pop(0))
            return SimpleNamespace(content=[block])

    monkeypatch.setattr(ac, "AsyncAnthropic", _Malformed)
    monkeypatch.setattr(ac.asyncio, "sleep", _no_sleep)
    result = await ac.call_structured(
        api_key="k", model="m", system="s", user="u", tool_name="t", tool_description="d",
        input_schema={"type": "object", "properties": {"sources": {"type": "array"}}, "required": ["sources"]},
        retry_malformed=True,
    )
    assert result == {"sources": [{"domain": "a.com"}]}


async def test_returns_a_malformed_field_as_is_unless_asked_to_retry(monkeypatch):
    # detect parses "a, b" for its list fields itself; a retry there could
    # fail a lookup it recovers from today.
    class _CommaJoined(_FakeClient):
        async def _create(self, **kwargs):
            block = SimpleNamespace(type="tool_use", name="t", input={"names": "a, b"})
            return SimpleNamespace(content=[block])

    monkeypatch.setattr(ac, "AsyncAnthropic", _CommaJoined)
    result = await ac.call_structured(
        api_key="k", model="m", system="s", user="u", tool_name="t", tool_description="d",
        input_schema={"type": "object", "properties": {"names": {"type": "array"}}, "required": ["names"]},
    )
    assert result == {"names": "a, b"}


async def _no_sleep(_):
    pass


# ── web search ───────────────────────────────────────────────────────

class _SearchFake:
    seen: list = []
    script: list = []  # responses (or exceptions) returned in order

    def __init__(self, **_):
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kw):
        _SearchFake.seen.append(kw)
        item = _SearchFake.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        pass


def _resp(blocks, stop="end_turn"):
    return SimpleNamespace(content=blocks, stop_reason=stop)


async def test_paused_search_turn_is_resumed_and_blocks_merged(monkeypatch):
    monkeypatch.setattr(ac, "AsyncAnthropic", _SearchFake)
    ac._clients.clear()
    _SearchFake.seen, _SearchFake.script = [], [_resp(["a", "b"], stop="pause_turn"), _resp(["c"])]
    out = await ac.call_web_search(api_key="k", model="m", system="s", user="u")
    assert out.content == ["a", "b", "c"]
    resumed = _SearchFake.seen[1]["messages"]
    assert resumed[-1] == {"role": "assistant", "content": ["a", "b"]}


async def test_requests_that_cannot_succeed_are_not_retried(monkeypatch):
    attempts = []

    async def bad():
        attempts.append(1)
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        raise BadRequestError("invalid", response=httpx.Response(400, request=request), body=None)

    with pytest.raises(BadRequestError):
        await ac._with_retry(bad, retries=3, label="t")
    assert len(attempts) == 1


async def test_one_shared_client_with_timeout_and_no_sdk_retries():
    made = []

    class Recording(_FakeClient):
        def __init__(self, **kw):
            made.append(kw)
            super().__init__(**kw)

    ac.AsyncAnthropic = Recording  # restored by the autouse monkeypatch
    ac._clients.clear()
    await _call()
    await _call()
    assert len(made) == 1 and made[0]["max_retries"] == 0 and made[0]["timeout"] > 0


def test_caches_are_per_loop_and_shutdown_keeps_the_semaphore():
    import asyncio

    async def grab():
        return ac._client("k"), ac._in_flight()

    first_loop, second_loop = asyncio.new_event_loop(), asyncio.new_event_loop()
    try:
        client_a, sem_a = first_loop.run_until_complete(grab())
        client_b, _ = second_loop.run_until_complete(grab())
        assert client_a is not client_b  # never a client bound to another loop
        first_loop.run_until_complete(ac.close_clients())
        assert first_loop.run_until_complete(grab())[1] is sem_a
    finally:
        first_loop.close()
        second_loop.close()


async def test_a_timed_out_web_search_is_not_rerun(monkeypatch):
    from anthropic import APITimeoutError

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    monkeypatch.setattr(ac, "AsyncAnthropic", _SearchFake)
    ac._clients.clear()
    _SearchFake.seen, _SearchFake.script = [], [APITimeoutError(request=request), _resp(["ok"])]
    with pytest.raises(APITimeoutError):
        await ac.call_web_search(api_key="k", model="m", system="s", user="u")
    assert len(_SearchFake.seen) == 1


# ── web fetch ────────────────────────────────────────────────────────

def _fetch_response(result):
    return SimpleNamespace(content=[
        {"type": "server_tool_use", "id": "s1", "name": "web_fetch", "input": {"url": "https://a.example/p"}},
        {"type": "web_fetch_tool_result", "tool_use_id": "s1", "content": result},
        {"type": "text", "text": "done."},
    ], stop_reason="end_turn")


async def test_web_fetch_returns_the_page_text_and_copy_date(monkeypatch):
    page = {"type": "web_fetch_result", "url": "https://a.example/p", "retrieved_at": "2026-06-20T00:00:00+00:00",
            "content": {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "Hello DCL"}}}
    monkeypatch.setattr(ac, "AsyncAnthropic", _SearchFake)
    ac._clients.clear()
    _SearchFake.seen, _SearchFake.script = [], [_fetch_response(page)]
    fetched = await ac.call_web_fetch(api_key="k", model="m", url="https://a.example/p", max_content_tokens=500)
    assert fetched.text == "Hello DCL" and fetched.retrieved_at.date().isoformat() == "2026-06-20"
    sent = _SearchFake.seen[0]
    assert sent["tools"][0]["type"] == "web_fetch_20250910" and sent["tools"][0]["max_content_tokens"] == 500
    assert sent["max_tokens"] >= 300  # room for the tool call (at 64 the model skipped it, confirmed live)
    assert sent["extra_headers"]["anthropic-beta"] == "web-fetch-2025-09-10"


async def test_web_fetch_error_or_empty_page_is_none(monkeypatch):
    monkeypatch.setattr(ac, "AsyncAnthropic", _SearchFake)
    ac._clients.clear()
    error = {"type": "web_fetch_tool_error", "error_code": "url_not_accessible"}
    empty = {"type": "web_fetch_result", "content": {"type": "document", "source": {"data": "  "}}}
    _SearchFake.seen, _SearchFake.script = [], [_fetch_response(error), _fetch_response(empty), _resp([])]
    for _ in range(3):
        assert await ac.call_web_fetch(api_key="k", model="m", url="https://a.example/p", max_content_tokens=500) is None


def test_decodes_list_items_sent_as_json_strings():
    # Confirmed live: persona generation got each persona as a JSON string.
    schema = {"type": "object", "properties": {"personas": {"type": "array", "items": {"type": "object"}}}}
    value = {"personas": ['{"title": "Planner"}', {"title": "Parent"}, "not json"]}
    assert ac._decode_stringified(value, schema, "t") == {"personas": [{"title": "Planner"}, {"title": "Parent"}, "not json"]}



async def test_every_structured_call_streams_under_one_generous_ceiling(monkeypatch):
    # Confirmed live: per-call limits sized for the output alone cut replies
    # off, since thinking counts against max_tokens. A reply that still hits the
    # ceiling ran away: the attempt fails and a fresh one is made.
    seen = []

    class _CutOff(_FakeClient):
        async def _create(self, **kwargs):
            seen.append(kwargs["max_tokens"])
            stop = "max_tokens" if len(seen) == 1 else "tool_use"
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="t", input={"x": "ok"})], stop_reason=stop)

    monkeypatch.setattr(ac, "AsyncAnthropic", _CutOff)
    monkeypatch.setattr(ac.asyncio, "sleep", _no_sleep)
    ac._NO_FORCED_TOOL_CHOICE.add("m")
    out = await ac.call_structured(api_key="k", model="m", system="s", user="u", tool_name="t", tool_description="d",
                                   input_schema={"type": "object", "properties": {}})
    assert out == {"x": "ok"} and seen == [32_000, 32_000]


async def test_an_answer_cut_off_at_its_ceiling_is_a_failure_not_a_scored_answer(monkeypatch):
    replies = [SimpleNamespace(content=[SimpleNamespace(type="text", text="Try Royal")], stop_reason="max_tokens"),
               SimpleNamespace(content=[SimpleNamespace(type="text", text="Try Royal Caribbean.")], stop_reason="end_turn")]

    class _Answers(_FakeClient):
        async def _create(self, **kwargs):
            return replies.pop(0)

    monkeypatch.setattr(ac, "AsyncAnthropic", _Answers)
    monkeypatch.setattr(ac.asyncio, "sleep", _no_sleep)
    assert await ac.call_text(api_key="k", model="m", system="s", user="u") == "Try Royal Caribbean."
