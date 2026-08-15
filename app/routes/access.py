"""Lets the frontend's access gate check a code before letting someone past
it. Pure DB lookup — no throttle, no redemption, no Anthropic call — so
unlike /api/prompts, /api/detect, and /api/analysis this isn't a billing
surface and doesn't need to be gated the same way.
"""
from fastapi import APIRouter

from app.code_store import get_code_store
from app.models import AccessStatusResponse

router = APIRouter()


@router.get("/api/access/status", response_model=AccessStatusResponse)
async def access_status(code: str) -> AccessStatusResponse:
    status = await get_code_store().status(code)
    return AccessStatusResponse(status=status)
