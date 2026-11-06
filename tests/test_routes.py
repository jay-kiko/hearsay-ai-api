"""HTTP edges: access-code gating, upstream error mapping, and the guess
throttle on code lookups."""
from types import SimpleNamespace

import httpx
import pytest
from anthropic import APIStatusError
from fastapi.testclient import TestClient

from app import code_store as code_store_module
from app import rate_limit
from app.code_store import CodeStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(code_store_module, "_store", CodeStore(str(tmp_path / "codes.db")))
    monkeypatch.setattr(rate_limit, "_limiters", {})
    from app.main import app

    with TestClient(app) as c:
        yield c


def _mint(client, uses=1) -> str:
    # On the app's own loop: the store's connection belongs to it.
    return client.portal.call(code_store_module.get_code_store().mint, 1, uses)[0]


def test_unknown_code_is_404_and_upstream_errors_hide_detail(client, monkeypatch):
    assert client.post("/api/detect", json={"query": "x", "accessCode": "NOPE-NOPE"}).status_code == 404

    async def failing_detect(**_):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        raise APIStatusError("secret upstream detail req_123", response=httpx.Response(500, request=request), body=None)

    monkeypatch.setattr("app.routes.detect.detect_brand", failing_detect)
    code = _mint(client)
    response = client.post("/api/detect", json={"query": "x", "accessCode": code})
    assert response.status_code == 502
    assert "req_123" not in response.text


def test_code_guessing_is_throttled_before_the_lookup(client):
    real = _mint(client)
    for i in range(20):
        assert client.get("/api/access/status", params={"code": f"GUES-{i:04d}"}).json()["status"] == "unknown"
    # Blocked now, and a throttled request can't tell a real code from a guess.
    assert client.get("/api/access/status", params={"code": real}).status_code == 429
    assert client.post("/api/detect", json={"query": "x", "accessCode": real}).status_code == 429


def test_failure_limiter_windows():
    from app.rate_limit import FailureLimiter

    limiter = FailureLimiter(per_key=3, global_limit=4, window_seconds=60, tight_per_key=2)
    for _ in range(2):
        limiter.record_failure("a")
    assert not limiter.blocked("a")
    limiter.record_failure("a")
    assert limiter.blocked("a") and not limiter.blocked("b")
    limiter.record_failure("b")
    # Global window full: a client that never failed still gets through,
    # clients with failures get the tight allowance.
    assert not limiter.blocked("c")
    limiter.record_failure("b")
    assert limiter.blocked("b")

    expiring = FailureLimiter(per_key=1, global_limit=99, window_seconds=0)
    expiring.record_failure("a")
    assert not expiring.blocked("a")


def test_a_guessing_flood_never_locks_out_valid_codes_or_the_admin(client):
    real = _mint(client)
    limiter = rate_limit.get_limiter()
    for i in range(5000):  # far past the global limit, from other clients
        limiter.record_failure(f"10.0.{i // 250}.{i % 250}")
    assert client.get("/api/access/status", params={"code": real}).json()["status"] == "valid"
    assert client.post("/api/detect", json={"query": "x", "accessCode": "NOPE-NOPE"}).status_code == 404
    assert not rate_limit.get_limiter("admin").blocked("testclient")  # admin has its own window


def test_ipv6_clients_are_keyed_by_their_64():
    from app.rate_limit import client_key

    def req(host):
        return SimpleNamespace(client=SimpleNamespace(host=host))

    assert client_key(req("2001:db8::1")) == client_key(req("2001:db8::ffff")) == "2001:db8::/64"
    assert client_key(req("::ffff:1.2.3.4")) == "1.2.3.4"
    assert client_key(req("1.2.3.4")) == "1.2.3.4"


def test_several_categories_make_one_prompt_call(client, monkeypatch):
    calls = []

    async def fake_generate(**kw):
        calls.append(kw)
        return {"p1": ["q1", "q2", "q3"]}

    monkeypatch.setattr("app.routes.prompts.generate_prompts", fake_generate)
    code = _mint(client)
    persona = {"id": "p1", "title": "T", "role": "r", "pains": "p", "criteria": "c"}
    response = client.post("/api/prompts", json={
        "brand": "B", "industry": "Cruises", "personas": [persona], "accessCode": code, "promptsPerPersona": 3,
        "categories": [{"name": n, "buyerContext": "x"} for n in ("Family", "Weekend", "Luxury")],
    })
    assert response.status_code == 200 and response.json()["prompts"] == {"p1": ["q1", "q2", "q3"]}
    assert len(calls) == 1
    assert [(c.name, c.count) for c in calls[0]["categories"]] == [("Family", 1), ("Weekend", 1), ("Luxury", 1)]
