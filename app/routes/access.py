"""Lets the frontend's access gate check a code before letting someone past
it. Pure DB lookup — no redemption, no Anthropic call — but it answers
"does this code exist", so failed lookups are throttled like everywhere else
(see app/rate_limit.py).
"""
from fastapi import APIRouter, Request

from app.code_store import get_code_store
from app.models import AccessCode, AccessStatusResponse
from app.rate_limit import client_key, ensure_not_throttled, get_limiter

router = APIRouter()


@router.get("/api/access/status", response_model=AccessStatusResponse)
async def access_status(code: AccessCode, request: Request) -> AccessStatusResponse:
    ensure_not_throttled(request)
    status = await get_code_store().status(code)
    if status == "unknown":
        get_limiter().record_failure(client_key(request))
    return AccessStatusResponse(status=status)
