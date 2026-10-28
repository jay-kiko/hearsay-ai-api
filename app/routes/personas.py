from fastapi import APIRouter

from app.config import get_settings
from app.models import (
    ExpandPersonaRequest,
    ExpandPersonaResponse,
    GeneratePersonasRequest,
    GeneratePersonasResponse,
)
from app.routes._access import charge_generation_call, upstream_errors
from app.services.persona_gen import expand_custom_persona, generate_personas

router = APIRouter()


@router.post("/api/generate-personas", response_model=GeneratePersonasResponse)
async def create_personas(body: GeneratePersonasRequest) -> GeneratePersonasResponse:
    await charge_generation_call(body.access_code)
    with upstream_errors():
        personas = await generate_personas(
            api_key=get_settings().anthropic_api_key,
            industry=body.industry,
            competitors=body.competitors,
            buyer_context=body.buyer_context,
            brand_summary=body.brand_summary,
            market=body.market,
            persona_count=body.persona_count,
        )
    return GeneratePersonasResponse(personas=personas)


@router.post("/api/personas/expand", response_model=ExpandPersonaResponse)
async def expand_persona(body: ExpandPersonaRequest) -> ExpandPersonaResponse:
    await charge_generation_call(body.access_code)
    with upstream_errors():
        persona = await expand_custom_persona(
            api_key=get_settings().anthropic_api_key,
            description=body.description,
            industry=body.industry,
            competitors=body.competitors,
            buyer_context=body.buyer_context,
            brand_summary=body.brand_summary,
            market=body.market,
        )
    return ExpandPersonaResponse(persona=persona)
