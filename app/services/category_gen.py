"""Candidate product-category facets within a brand's broader industry —
the Wizard's "Select a product category" step. Confirmed live that a single
broad industry label (e.g. "Budget and Value Hospitality") can be too coarse
to consistently ground personas/prompts in the right niche; letting the user
pick a specific facet (or skip, or supply a custom one) fixes that at the
source instead of hoping detection lands on the precise framing every time.

Each facet carries its own scoped buyer_context, not a copy of the brand's
overall one — confirmed live that a brand can genuinely have more than one
distinct buyer type (Hotel101 sells to both real-estate investors AND
short-stay guests), and reusing one shared buyer_context across every facet
lets the wrong audience leak into a facet that should have excluded it (an
"investor" persona showing up under a guest-stay-focused category). Scoping
it here, once, is more reliable than hoping persona/prompt generation
correctly filters a mixed buyer_context down to the right half every time.

One cheap structured call — mechanical brainstorming over already-confirmed
brand context, not open-ended research, so the fast model is fine here.
"""
from __future__ import annotations

from app.config import get_settings
from app.models import CategorySuggestion, Competitor
from app.services.anthropic_client import call_structured

_TOOL_NAME = "suggest_product_categories"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "A short (2-4 word), specific, genuinely distinct facet within this "
                            "brand's industry — a literal sub-category (e.g. 'Hair Care' within "
                            "'Beauty'), a positioning angle (e.g. 'Affordable Beauty'), or an "
                            "audience focus (e.g. 'Sensitive Skin') as appropriate to this brand."
                        ),
                    },
                    "buyerContext": {
                        "type": "string",
                        "description": (
                            "One sentence describing specifically who chooses within THIS facet "
                            "and why — scoped to this facet alone. If the brand's overall buyer "
                            "context describes more than one distinct kind of buyer (e.g. both "
                            "investors and end customers), this facet's buyer context must cover "
                            "only the buyer type actually relevant to this specific facet, never "
                            "a blend of both."
                        ),
                    },
                },
                "required": ["name", "buyerContext"],
            },
            "description": "4-6 genuinely distinct facets, each meaningfully different from the others.",
        }
    },
    "required": ["categories"],
}

_SYSTEM = (
    "You identify specific, distinct product-category facets within a brand's broader industry "
    "— the more specific angles a marketing team might want to focus their AI-visibility research "
    "on, rather than testing the whole broad industry at once. Given a brand's industry, "
    "competitors, and description, propose facets genuinely specific to this brand — not a "
    "generic list that could apply to any company in the industry. Critically: if the brand's "
    "overall buyer context describes more than one distinct kind of buyer (e.g. investors versus "
    "end customers, or businesses versus consumers), each facet must belong clearly to ONE of "
    "those buyer types and its own buyerContext must reflect only that one — never produce a "
    "facet whose buyerContext blends multiple distinct buyer types together."
)


def _user_prompt(brand: str, industry: str, competitors: list[Competitor], buyer_context: str | None, brand_summary: str | None) -> str:
    competitor_list = ", ".join(c.name for c in competitors) if competitors else "unspecified competitors"
    lines = [
        f"Brand: {brand}",
        f"Industry: {industry}",
        f"Competitors: {competitor_list}",
    ]
    if buyer_context:
        lines.append(f"Overall buyer context (may describe more than one distinct buyer type): {buyer_context}")
    if brand_summary:
        lines.append(f"Brand summary: {brand_summary}")
    lines.append("Suggest specific product-category facets within this industry for this brand.")
    return "\n".join(lines)


def _normalize(value: object) -> list[dict]:
    """Same defensive posture as detect.py's competitor parsing — forced tool
    output isn't a hard type guarantee, recover a usable list either way."""
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return []


async def generate_categories(
    *,
    api_key: str,
    brand: str,
    industry: str,
    competitors: list[Competitor],
    buyer_context: str | None,
    brand_summary: str | None,
) -> list[CategorySuggestion]:
    settings = get_settings()

    result = await call_structured(
        api_key=api_key,
        model=settings.anthropic_fast_model,
        system=_SYSTEM,
        user=_user_prompt(brand, industry, competitors, buyer_context, brand_summary),
        tool_name=_TOOL_NAME,
        tool_description="Return the candidate product-category facets.",
        input_schema=_INPUT_SCHEMA,
        max_tokens=1024,
    )

    suggestions: list[CategorySuggestion] = []
    for entry in _normalize(result.get("categories", [])):
        name = str(entry.get("name", "")).strip()
        context = str(entry.get("buyerContext", "")).strip()
        if name and context:
            suggestions.append(CategorySuggestion(name=name, buyer_context=context))
    return suggestions
