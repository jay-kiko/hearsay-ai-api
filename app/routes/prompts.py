from anthropic import AnthropicError, AuthenticationError
from fastapi import APIRouter, HTTPException

from app.code_store import get_code_store
from app.config import get_settings
from app.models import PromptsRequest, PromptsResponse
from app.services.prompt_gen import generate_prompts

router = APIRouter()


@router.post("/api/prompts", response_model=PromptsResponse)
async def create_prompts(body: PromptsRequest) -> PromptsResponse:
    settings = get_settings()

    # Validated but not consumed — the user can regenerate prompts while
    # reviewing them; a use is only spent when they actually launch a job.
    # Still capped per code so this path can't rack up unlimited billed calls
    # on a code that never gets spent on an analysis.
    call_status = await get_code_store().register_prompt_call(body.access_code, settings.max_prompt_calls_per_code)
    if call_status == "unknown":
        raise HTTPException(status_code=404, detail="Access code not found")
    if call_status == "revoked":
        raise HTTPException(status_code=403, detail="This access code has been revoked")
    if call_status == "exhausted":
        raise HTTPException(status_code=403, detail="This access code has no uses remaining")
    if call_status == "rate_limited":
        raise HTTPException(status_code=429, detail="Too many prompt regenerations for this access code")

    try:
        prompts = await generate_prompts(
            api_key=settings.anthropic_api_key,
            brand=body.brand,
            industry=body.industry,
            buyer_context=body.buyer_context,
            personas=body.personas,
            prompts_per_persona=body.prompts_per_persona,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=500, detail="Server's Anthropic API key was rejected") from exc
    except AnthropicError as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {exc}") from exc

    return PromptsResponse(prompts=prompts)
