"""HTTP edges: access-code gating, upstream error mapping, and the guess
throttle on code lookups."""
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
    monkeypatch.setattr(rate_limit, "_limiter", None)
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

    limiter = FailureLimiter(per_key=2, global_limit=3, window_seconds=60)
    limiter.record_failure("a")
    assert not limiter.blocked("a")
    limiter.record_failure("a")
    assert limiter.blocked("a") and not limiter.blocked("b")
    limiter.record_failure("b")
    assert limiter.blocked("c")  # global backstop

    expiring = FailureLimiter(per_key=1, global_limit=99, window_seconds=0)
    expiring.record_failure("a")
    assert not expiring.blocked("a")
