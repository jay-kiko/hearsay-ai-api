import asyncio

from anthropic import AnthropicError, AuthenticationError
from fastapi import APIRouter, HTTPException

from app.code_store import get_code_store
from app.config import get_settings
from app.models import AdaptSeedPromptRequest, AdaptSeedPromptResponse, PromptsRequest, PromptsResponse
from app.services.prompt_gen import adapt_seed_prompt, generate_prompts

router = APIRouter()

_CALL_STATUS_ERRORS = {
    "unknown": (404, "Access code not found"),
    "revoked": (403, "This access code has been revoked"),
    "exhausted": (403, "This access code has no uses remaining"),
    "rate_limited": (429, "Too many prompt regenerations for this access code"),
}


def _raise_for_call_status(status: str) -> None:
    error = _CALL_STATUS_ERRORS.get(status)
    if error:
        code, detail = error
        raise HTTPException(status_code=code, detail=detail)


@router.post("/api/prompts", response_model=PromptsResponse)
async def create_prompts(body: PromptsRequest) -> PromptsResponse:
    settings = get_settings()
    code_store = get_code_store()

    if body.categories:
        # One category picked used to mean "one generation call" — selecting
        # several now means several real, billed generation calls (one per
        # category, run across every persona), so the pre-spend throttle has
        # to charge proportionally: N categories costs N calls, not 1, or a
        # multi-select run would get unlimited free generation past the cap.
        for _ in body.categories:
            call_status = await code_store.register_prompt_call(body.access_code, settings.max_prompt_calls_per_code)
            _raise_for_call_status(call_status)

        try:
            per_category = await asyncio.gather(
                *[
                    generate_prompts(
                        api_key=settings.anthropic_api_key,
                        brand=body.brand,
                        industry=f"{body.industry} — {category.name}",
                        buyer_context=category.buyer_context,
                        brand_summary=body.brand_summary,
                        market=body.market,
                        personas=body.personas,
                        prompts_per_persona=body.prompts_per_persona,
                    )
                    for category in body.categories
                ]
            )
        except AuthenticationError as exc:
            raise HTTPException(status_code=500, detail="Server's Anthropic API key was rejected") from exc
        except AnthropicError as exc:
            raise HTTPException(status_code=502, detail=f"Anthropic API error: {exc}") from exc

        merged: dict[str, list[str]] = {persona.id: [] for persona in body.personas}
        for category_prompts in per_category:
            for persona_id, prompts in category_prompts.items():
                merged.setdefault(persona_id, []).extend(prompts)
        return PromptsResponse(prompts=merged)

    # Validated but not consumed — the user can regenerate prompts while
    # reviewing them; a use is only spent when they actually launch a job.
    # Still capped per code so this path can't rack up unlimited billed calls
    # on a code that never gets spent on an analysis.
    call_status = await code_store.register_prompt_call(body.access_code, settings.max_prompt_calls_per_code)
    _raise_for_call_status(call_status)

    try:
        prompts = await generate_prompts(
            api_key=settings.anthropic_api_key,
            brand=body.brand,
            industry=body.industry,
            buyer_context=body.buyer_context,
            brand_summary=body.brand_summary,
            market=body.market,
            personas=body.personas,
            prompts_per_persona=body.prompts_per_persona,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=500, detail="Server's Anthropic API key was rejected") from exc
    except AnthropicError as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {exc}") from exc

    return PromptsResponse(prompts=prompts)


@router.post("/api/prompts/seed", response_model=AdaptSeedPromptResponse)
async def create_seed_prompts(body: AdaptSeedPromptRequest) -> AdaptSeedPromptResponse:
    settings = get_settings()

    # Same shared pre-spend throttle, not consumed — one adaptation call per
    # request regardless of persona count, same as /api/prompts.
    call_status = await get_code_store().register_prompt_call(body.access_code, settings.max_prompt_calls_per_code)
    _raise_for_call_status(call_status)

    try:
        prompts = await adapt_seed_prompt(
            api_key=settings.anthropic_api_key,
            brand=body.brand,
            industry=body.industry,
            buyer_context=body.buyer_context,
            brand_summary=body.brand_summary,
            market=body.market,
            personas=body.personas,
            seed_prompt=body.seed_prompt,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=500, detail="Server's Anthropic API key was rejected") from exc
    except AnthropicError as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {exc}") from exc

    return AdaptSeedPromptResponse(prompts=prompts)
