from anthropic import AnthropicError, AuthenticationError
from fastapi import APIRouter, HTTPException

from app.code_store import get_code_store
from app.config import get_settings
from app.models import DetectRequest, DetectResponse
from app.services.detect import detect_brand

router = APIRouter()


@router.post("/api/detect", response_model=DetectResponse)
async def detect(body: DetectRequest) -> DetectResponse:
    settings = get_settings()

    # Same pre-spend throttle as /api/prompts, and deliberately the *same*
    # counter (register_prompt_call) rather than a separate one — both are
    # billed AI calls on a code that hasn't been redeemed yet, so they share
    # one budget instead of each getting their own unlimited allowance.
    call_status = await get_code_store().register_prompt_call(body.access_code, settings.max_prompt_calls_per_code)
    if call_status == "unknown":
        raise HTTPException(status_code=404, detail="Access code not found")
    if call_status == "revoked":
        raise HTTPException(status_code=403, detail="This access code has been revoked")
    if call_status == "exhausted":
        raise HTTPException(status_code=403, detail="This access code has no uses remaining")
    if call_status == "rate_limited":
        raise HTTPException(status_code=429, detail="Too many detection/prompt calls for this access code")

    try:
        return await detect_brand(api_key=settings.anthropic_api_key, query=body.query)
    except AuthenticationError as exc:
        raise HTTPException(status_code=500, detail="Server's Anthropic API key was rejected") from exc
    except AnthropicError as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {exc}") from exc
