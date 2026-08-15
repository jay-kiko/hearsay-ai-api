"""Home-screen query detection — turns the free-text brand description into
a structured brand/industry/competitor set, the input the Wizard needs to
generate personas and prompts against. One forced-tool call, same pattern as
prompt_gen.py.
"""
from __future__ import annotations

from app.config import get_settings
from app.models import DetectResponse
from app.services.anthropic_client import call_structured

_TOOL_NAME = "detect_brand"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "brand": {"type": "string", "description": "The brand or product name."},
        "industry": {
            "type": "string",
            "description": "A specific category name, e.g. 'Project Management Software', not a vague label.",
        },
        "competitors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-5 real, specific, named competitors in that same category — no generic placeholders.",
        },
    },
    "required": ["brand", "industry", "competitors"],
}

_SYSTEM = (
    "You extract a brand, its industry category, and its real competitors from a short "
    "free-text description a user typed about their own product. Identify the brand name "
    "and a specific industry/category label, then name 3-5 real, specific companies that "
    "actually compete in that category — never generic placeholders like 'Competitor A' "
    "and never the brand itself."
)


async def detect_brand(*, api_key: str, query: str) -> DetectResponse:
    settings = get_settings()

    result = await call_structured(
        api_key=api_key,
        model=settings.anthropic_fast_model,
        system=_SYSTEM,
        user=query,
        tool_name=_TOOL_NAME,
        tool_description="Return the detected brand, industry, and competitors.",
        input_schema=_INPUT_SCHEMA,
        max_tokens=512,
    )

    return DetectResponse(
        brand=result["brand"],
        industry=result["industry"],
        competitors=result.get("competitors", []),
    )
