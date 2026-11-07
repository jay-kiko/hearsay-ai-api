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

import logging

from app.config import get_settings
from app.models import NAME_MAX, PARAGRAPH_MAX, CategorySuggestion, Competitor, fit
from app.services.anthropic_client import call_structured
from app.services.scoring import find_mentions

logger = logging.getLogger("hearsay.category_gen")

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
                            "audience focus (e.g. 'Sensitive Skin') as appropriate to this brand. "
                            "Never a brand, product or company name."
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
                    "competitorsHere": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Which of the given competitors buyers also weigh within this facet, by "
                            "their names as given. Every facet needs at least one: a facet none of them "
                            "competes in is built around this brand alone."
                        ),
                    },
                },
                "required": ["name", "buyerContext", "competitorsHere"],
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
    "competitors, and description, propose facets of the market this brand sells into: specific "
    "enough to be useful (not a generic list that fits any industry), and each one a choice where "
    "buyers weigh several real options. Never build a facet around something only this brand "
    "offers (its own characters, licensed content, signature products, programmes or features), "
    "and never name the brand, its products or any company: every buyer question written for a "
    "facet only one brand fits points at that brand, and the research then measures nothing. "
    "Critically: if the brand's "
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
    brand_names: list[str] | None = None,
) -> list[CategorySuggestion]:
    """Facets of the market, each one competitors also serve. Confirmed live:
    asked for facets "specific to this brand", it proposed ones only the brand
    fills ("Premium Themed Entertainment Cruises" for Disney), and all 8 of
    Disney's mentions came from that facet's prompts. A facet no given
    competitor serves, or whose text names the brand or a competitor, is
    dropped: its prompts would point at one brand."""
    settings = get_settings()

    result = await call_structured(
        api_key=api_key,
        model=settings.anthropic_quality_model,
        system=_SYSTEM,
        user=_user_prompt(brand, industry, competitors, buyer_context, brand_summary),
        tool_name=_TOOL_NAME,
        tool_description="Return the candidate product-category facets.",
        input_schema=_INPUT_SCHEMA,
        # Room for the quality tier's thinking, which counts against max_tokens.
        max_tokens=4096,
    )

    known = {n.strip().lower() for c in competitors for n in (c.name, *c.match_names) if n.strip()}
    names = [brand, *(brand_names or [])]
    suggestions: list[CategorySuggestion] = []
    for entry in _normalize(result.get("categories", [])):
        # Fitted to the request limits: a suggestion is sent back in the
        # prompts request, and an over-long one failed validation with a 500.
        name = fit(entry.get("name"), NAME_MAX)
        context = fit(entry.get("buyerContext"), PARAGRAPH_MAX)
        if not (name and context):
            continue
        rivals = entry.get("competitorsHere") if isinstance(entry.get("competitorsHere"), list) else []
        if known and not any(isinstance(r, str) and r.strip().lower() in known for r in rivals):
            logger.info("dropping category %r: no given competitor serves it", name)
            continue
        if find_mentions(f"{name}. {context}", brand, competitors, names, caseless=True):
            logger.info("dropping category %r: it names the brand or a competitor", name)
            continue
        suggestions.append(CategorySuggestion(name=name, buyer_context=context))
    return suggestions
