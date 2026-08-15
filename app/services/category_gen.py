"""Candidate product-category facets within a brand's broader industry —
the Wizard's "Select a product category" step. Confirmed live that a single
broad industry label (e.g. "Budget and Value Hospitality") can be too coarse
to consistently ground personas/prompts in the right niche; letting the user
pick a specific facet (or skip, or supply a custom one) fixes that at the
source instead of hoping detection lands on the precise framing every time.
One cheap structured call — mechanical brainstorming over already-confirmed
brand context, not open-ended research, so the fast model is fine here.
"""
from __future__ import annotations

from app.config import get_settings
from app.services.anthropic_client import call_structured

_TOOL_NAME = "suggest_product_categories"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "categories": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "4-6 short (2-4 word), specific, genuinely distinct facets within this brand's "
                "industry — a mix of literal sub-categories (e.g. 'Hair Care' within 'Beauty'), "
                "positioning angles (e.g. 'Affordable Beauty', 'Natural Ingredients'), and "
                "audience focuses (e.g. 'Sensitive Skin') as appropriate to this specific brand. "
                "Not generic filler — each one should meaningfully change what a persona/prompt "
                "set focused on it would look like."
            ),
        }
    },
    "required": ["categories"],
}

_SYSTEM = (
    "You identify specific, distinct product-category facets within a brand's broader industry "
    "— the more specific angles a marketing team might want to focus their AI-visibility research "
    "on, rather than testing the whole broad industry at once. Given a brand's industry, "
    "competitors, and description, propose facets genuinely specific to this brand — not a "
    "generic list that could apply to any company in the industry."
)


def _user_prompt(brand: str, industry: str, competitors: list[str], buyer_context: str | None, brand_summary: str | None) -> str:
    competitor_list = ", ".join(competitors) if competitors else "unspecified competitors"
    lines = [
        f"Brand: {brand}",
        f"Industry: {industry}",
        f"Competitors: {competitor_list}",
    ]
    if buyer_context:
        lines.append(f"Buyer context: {buyer_context}")
    if brand_summary:
        lines.append(f"Brand summary: {brand_summary}")
    lines.append("Suggest specific product-category facets within this industry for this brand.")
    return "\n".join(lines)


async def generate_categories(
    *,
    api_key: str,
    brand: str,
    industry: str,
    competitors: list[str],
    buyer_context: str | None,
    brand_summary: str | None,
) -> list[str]:
    settings = get_settings()

    result = await call_structured(
        api_key=api_key,
        model=settings.anthropic_fast_model,
        system=_SYSTEM,
        user=_user_prompt(brand, industry, competitors, buyer_context, brand_summary),
        tool_name=_TOOL_NAME,
        tool_description="Return the candidate product-category facets.",
        input_schema=_INPUT_SCHEMA,
        max_tokens=512,
    )

    categories = result.get("categories", [])
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]
    return [str(c).strip() for c in categories if str(c).strip()]
