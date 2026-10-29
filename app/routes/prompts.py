import asyncio

from fastapi import APIRouter, Request

from app.config import get_settings
from app.models import AdaptSeedPromptRequest, AdaptSeedPromptResponse, PromptsRequest, PromptsResponse
from app.routes._access import charge_generation_call, upstream_errors
from app.services.prompt_gen import adapt_seed_prompt, generate_prompts

router = APIRouter()


def _screen_names(body: PromptsRequest | AdaptSeedPromptRequest) -> list[str]:
    """Names a generated prompt must not contain. Context names ("Disney")
    count here unconditionally: a prompt that says "Disney" leaks the brand
    whatever the sentence is about."""
    names = [*body.brand_match_names, *body.brand_context_names]
    return [body.brand, *names] if names else []


@router.post("/api/prompts", response_model=PromptsResponse)
async def create_prompts(body: PromptsRequest, request: Request) -> PromptsResponse:
    settings = get_settings()

    if body.categories:
        # prompts_per_persona is a fixed total per persona, not a per-category
        # rate — selecting more categories must split that same budget across
        # them, not multiply it (5 categories at 3 each used to mean 15
        # prompts per persona instead of 3). Distributed as evenly as
        # possible; a category whose share rounds to 0 is skipped entirely —
        # no call, no throttle charge — rather than still costing one for
        # zero prompts.
        total = body.prompts_per_persona or settings.prompts_per_persona
        n = len(body.categories)
        base, extra = divmod(total, n)
        active = [
            (category, base + (1 if i < extra else 0)) for i, category in enumerate(body.categories)
        ]
        active = [(category, count) for category, count in active if count > 0]

        # One category picked used to mean "one generation call" — selecting
        # several now means several real, billed generation calls (one per
        # active category, run across every persona), so the pre-spend
        # throttle has to charge proportionally: N active categories costs N
        # calls, not 1, or a multi-select run would get unlimited free
        # generation past the cap.
        for _ in active:
            await charge_generation_call(body.access_code, request)

        with upstream_errors():
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
                        prompts_per_persona=count,
                        competitors=body.competitors,
                        brand_match_names=_screen_names(body),
                    )
                    for category, count in active
                ]
            )

        merged: dict[str, list[str]] = {persona.id: [] for persona in body.personas}
        for category_prompts in per_category:
            for persona_id, prompts in category_prompts.items():
                merged.setdefault(persona_id, []).extend(prompts)
        return PromptsResponse(prompts=merged)

    # Validated but not consumed — the user can regenerate prompts while
    # reviewing them; a use is only spent when they actually launch a job.
    # Still capped per code so this path can't rack up unlimited billed calls
    # on a code that never gets spent on an analysis.
    await charge_generation_call(body.access_code, request)

    with upstream_errors():
        prompts = await generate_prompts(
            api_key=settings.anthropic_api_key,
            brand=body.brand,
            industry=body.industry,
            buyer_context=body.buyer_context,
            brand_summary=body.brand_summary,
            market=body.market,
            personas=body.personas,
            prompts_per_persona=body.prompts_per_persona,
            competitors=body.competitors,
            brand_match_names=_screen_names(body),
        )

    return PromptsResponse(prompts=prompts)


@router.post("/api/prompts/seed", response_model=AdaptSeedPromptResponse)
async def create_seed_prompts(body: AdaptSeedPromptRequest, request: Request) -> AdaptSeedPromptResponse:
    settings = get_settings()

    # Same shared pre-spend throttle, not consumed — one adaptation call per
    # request regardless of persona count, same as /api/prompts.
    await charge_generation_call(body.access_code, request)

    with upstream_errors():
        prompts = await adapt_seed_prompt(
            api_key=settings.anthropic_api_key,
            brand=body.brand,
            industry=body.industry,
            buyer_context=body.buyer_context,
            brand_summary=body.brand_summary,
            market=body.market,
            personas=body.personas,
            seed_prompt=body.seed_prompt,
            competitors=body.competitors,
            brand_match_names=_screen_names(body),
        )

    return AdaptSeedPromptResponse(prompts=prompts)
