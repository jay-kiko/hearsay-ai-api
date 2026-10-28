"""Mint/list multi-use access codes over HTTP, for when shelling into the
container to run the CLI script (app/scripts/mint_codes.py) isn't convenient.
Disabled entirely unless ADMIN_SECRET is set.
"""
import secrets

from fastapi import APIRouter, Header, HTTPException, Query, Request

from app.code_store import get_code_store
from app.config import get_settings
from app.rate_limit import client_key, ensure_not_throttled, get_limiter

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(request: Request, x_admin_secret: str | None) -> None:
    configured = get_settings().admin_secret
    if not configured:
        raise HTTPException(status_code=503, detail="Admin endpoints are disabled (ADMIN_SECRET not set)")
    ensure_not_throttled(request, scope="admin")
    if not x_admin_secret or not secrets.compare_digest(x_admin_secret, configured):
        get_limiter().record_failure(client_key(request, scope="admin"))
        raise HTTPException(status_code=401, detail="Invalid admin secret")


@router.post("/codes")
async def mint_codes(
    request: Request,
    count: int = Query(default=1, ge=1, le=500),
    uses: int = Query(default=1, ge=1, le=10_000),
    x_admin_secret: str | None = Header(default=None),
) -> dict[str, list[str]]:
    _require_admin(request, x_admin_secret)
    codes = await get_code_store().mint(count, uses=uses)
    return {"codes": codes}


@router.get("/codes")
async def list_codes(request: Request, x_admin_secret: str | None = Header(default=None)) -> dict[str, list[dict]]:
    _require_admin(request, x_admin_secret)
    records = await get_code_store().list_all()
    return {"codes": [r.model_dump(by_alias=True) for r in records]}


@router.delete("/codes/{code}")
async def revoke_code(code: str, request: Request, x_admin_secret: str | None = Header(default=None)) -> dict[str, bool]:
    """Soft delete — the row and its redemption history stay for the audit
    trail, but the code is immediately and permanently unusable."""
    _require_admin(request, x_admin_secret)
    if not await get_code_store().revoke(code):
        raise HTTPException(status_code=404, detail="Access code not found")
    return {"revoked": True}
