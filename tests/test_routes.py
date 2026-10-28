"""HTTP edges: access-code gating, upstream error mapping, and the guess
throttle on code lookups."""
import httpx
import pytest
from anthropic import APIStatusError
from fastapi.testclient import TestClient

from app import code_store as code_store_module
from app.code_store import CodeStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(code_store_module, "_store", CodeStore(str(tmp_path / "codes.db")))
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
