from anthropic import AnthropicError, AuthenticationError
from fastapi import APIRouter, HTTPException

from app.code_store import get_code_store
from app.config import get_settings
from app.models import GeneratePersonasRequest, GeneratePersonasResponse
from app.services.persona_gen import generate_personas

router = APIRouter()


@router.post("/api/generate-personas", response_model=GeneratePersonasResponse)
async def create_personas(body: GeneratePersonasRequest) -> GeneratePersonasResponse:
    settings = get_settings()

    # Same pre-spend throttle as /api/detect and /api/prompts, and
    # deliberately the *same* counter — all three are billed AI calls on a
    # code that hasn't been redeemed yet, sharing one budget rather than
    # each getting its own unlimited allowance.
    call_status = await get_code_store().register_prompt_call(body.access_code, settings.max_prompt_calls_per_code)
    if call_status == "unknown":
        raise HTTPException(status_code=404, detail="Access code not found")
    if call_status == "revoked":
        raise HTTPException(status_code=403, detail="This access code has been revoked")
    if call_status == "exhausted":
        raise HTTPException(status_code=403, detail="This access code has no uses remaining")
    if call_status == "rate_limited":
        raise HTTPException(status_code=429, detail="Too many detection/prompt/persona calls for this access code")

    try:
        personas = await generate_personas(
            api_key=settings.anthropic_api_key,
            industry=body.industry,
            competitors=body.competitors,
            persona_count=body.persona_count,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=500, detail="Server's Anthropic API key was rejected") from exc
    except AnthropicError as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {exc}") from exc

    return GeneratePersonasResponse(personas=personas)
