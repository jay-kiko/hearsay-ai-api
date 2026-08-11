"""Mint/list multi-use access codes over HTTP, for when shelling into the
container to run the CLI script (app/scripts/mint_codes.py) isn't convenient.
Disabled entirely unless ADMIN_SECRET is set.
"""
import secrets

from fastapi import APIRouter, Header, HTTPException, Query

from app.code_store import get_code_store
from app.config import get_settings

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(x_admin_secret: str | None) -> None:
    configured = get_settings().admin_secret
    if not configured:
        raise HTTPException(status_code=503, detail="Admin endpoints are disabled (ADMIN_SECRET not set)")
    if not x_admin_secret or not secrets.compare_digest(x_admin_secret, configured):
        raise HTTPException(status_code=401, detail="Invalid admin secret")


@router.post("/codes")
async def mint_codes(
    count: int = Query(default=1, ge=1, le=500),
    uses: int = Query(default=1, ge=1, le=10_000),
    x_admin_secret: str | None = Header(default=None),
) -> dict[str, list[str]]:
    _require_admin(x_admin_secret)
    codes = await get_code_store().mint(count, uses=uses)
    return {"codes": codes}


@router.get("/codes")
async def list_codes(x_admin_secret: str | None = Header(default=None)) -> dict[str, list[dict]]:
    _require_admin(x_admin_secret)
    records = await get_code_store().list_all()
    return {"codes": [r.model_dump(by_alias=True) for r in records]}
