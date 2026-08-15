from anthropic import AnthropicError, AuthenticationError
from fastapi import APIRouter, HTTPException

from app.code_store import get_code_store
from app.config import get_settings
from app.models import CategoriesRequest, CategoriesResponse
from app.services.category_gen import generate_categories

router = APIRouter()


@router.post("/api/categories", response_model=CategoriesResponse)
async def create_categories(body: CategoriesRequest) -> CategoriesResponse:
    settings = get_settings()

    # Same shared pre-spend throttle as /api/detect, /api/prompts, and
    # /api/generate-personas — all billed AI calls on a code that hasn't
    # been redeemed yet, sharing one budget rather than each getting its
    # own unlimited allowance.
    call_status = await get_code_store().register_prompt_call(body.access_code, settings.max_prompt_calls_per_code)
    if call_status == "unknown":
        raise HTTPException(status_code=404, detail="Access code not found")
    if call_status == "revoked":
        raise HTTPException(status_code=403, detail="This access code has been revoked")
    if call_status == "exhausted":
        raise HTTPException(status_code=403, detail="This access code has no uses remaining")
    if call_status == "rate_limited":
        raise HTTPException(status_code=429, detail="Too many detection/prompt/persona/category calls for this access code")

    try:
        categories = await generate_categories(
            api_key=settings.anthropic_api_key,
            brand=body.brand,
            industry=body.industry,
            competitors=body.competitors,
            buyer_context=body.buyer_context,
            brand_summary=body.brand_summary,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=500, detail="Server's Anthropic API key was rejected") from exc
    except AnthropicError as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {exc}") from exc

    return CategoriesResponse(categories=categories)
