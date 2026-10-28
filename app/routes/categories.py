from fastapi import APIRouter

from app.config import get_settings
from app.models import CategoriesRequest, CategoriesResponse
from app.routes._access import charge_generation_call, upstream_errors
from app.services.category_gen import generate_categories

router = APIRouter()


@router.post("/api/categories", response_model=CategoriesResponse)
async def create_categories(body: CategoriesRequest) -> CategoriesResponse:
    await charge_generation_call(body.access_code)
    with upstream_errors():
        categories = await generate_categories(
            api_key=get_settings().anthropic_api_key,
            brand=body.brand,
            industry=body.industry,
            competitors=body.competitors,
            buyer_context=body.buyer_context,
            brand_summary=body.brand_summary,
        )
    return CategoriesResponse(categories=categories)
