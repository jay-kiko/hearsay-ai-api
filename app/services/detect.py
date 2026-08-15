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
        "buyerContext": {
            "type": "string",
            "description": (
                "One sentence describing what real-world choice buyers in this category actually "
                "make — grounded in the true nature of this specific brand's industry, not assumed "
                "to be software. Examples of the range this must cover: 'Businesses and teams "
                "evaluating and adopting project management software' (B2B SaaS), 'Travelers and "
                "event planners choosing where to stay for leisure, business, or celebrations' "
                "(hospitality), 'Shoppers picking a snack brand off the shelf or in a delivery app' "
                "(consumer packaged goods), 'Marketing teams choosing an agency or platform to run "
                "campaigns' (services/agencies), 'Everyday consumers choosing a payment app for "
                "daily transactions and remittances' (consumer fintech). Never default to a "
                "software/vendor framing unless the brand genuinely is software."
            ),
        },
    },
    "required": ["brand", "industry", "competitors", "buyerContext"],
}

_SYSTEM = (
    "You extract a brand, its industry category, its real competitors, and its buyer context "
    "from a short free-text description a user typed about their own product. Identify the "
    "brand name and a specific industry/category label, then name 3-5 real, specific companies "
    "that actually compete in that category — never generic placeholders like 'Competitor A' "
    "and never the brand itself. Critically, also identify what kind of real-world choice this "
    "actually is: software/vendor procurement, a consumer product purchase, a hospitality/travel "
    "booking, a service or agency hire, or something else entirely — ground this in the true "
    "nature of the brand's industry rather than defaulting to a generic 'evaluating tools' frame."
)


async def detect_brand(*, api_key: str, query: str) -> DetectResponse:
    settings = get_settings()

    result = await call_structured(
        api_key=api_key,
        model=settings.anthropic_fast_model,
        system=_SYSTEM,
        user=query,
        tool_name=_TOOL_NAME,
        tool_description="Return the detected brand, industry, competitors, and buyer context.",
        input_schema=_INPUT_SCHEMA,
        max_tokens=512,
    )

    return DetectResponse(
        brand=result["brand"],
        industry=result["industry"],
        competitors=result.get("competitors", []),
        buyer_context=result["buyerContext"],
    )
