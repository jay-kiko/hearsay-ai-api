from fastapi import APIRouter, Request

from app.config import get_settings
from app.models import AdaptSeedPromptRequest, AdaptSeedPromptResponse, PromptsRequest, PromptsResponse
from app.routes._access import charge_generation_call, upstream_errors
from app.services.prompt_gen import CategoryPlan, adapt_seed_prompt, generate_prompts

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
        # possible; a category whose share rounds to 0 is left out.
        total = body.prompts_per_persona or settings.prompts_per_persona
        n = len(body.categories)
        base, extra = divmod(total, n)
        plan = [
            CategoryPlan(name=category.name, buyer_context=category.buyer_context, count=base + (1 if i < extra else 0))
            for i, category in enumerate(body.categories)
        ]
        plan = [c for c in plan if c.count > 0]

        # One call for every category, so each persona's prompts are written
        # together and don't reword one another (see generate_prompts).
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
                prompts_per_persona=total,
                competitors=body.competitors,
                brand_match_names=_screen_names(body),
                categories=plan,
            )
        return PromptsResponse(prompts=prompts)

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
