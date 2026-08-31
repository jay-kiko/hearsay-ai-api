"""§07 Competitive insights — one AI reasoning pass over a completed analysis.

Competitor diagnosis (why rivals win / where the brand wins / gaps), the
Opportunities list, and the brand-strength radar all need judgment over the
whole run (every persona's mentions/sentiment/rank, share of voice, cited
sources) — not something deterministic math (app.services.aggregation) or
search grounding alone produces. Bundled into one structured call rather than
three since they all reason over identical evidence — same principle as
category_gen.py's one call covering several facets.

This is synthesis over already-collected data, not the organic-mention
signal being measured (that's pipeline.py's job) — runs on the fast model,
once per job, and fails open (empty results) rather than sinking the whole
job on a bad call, matching grounding.py's posture.
"""
from __future__ import annotations

import json
import logging

from app.config import get_settings
from app.models import (
    Competitor,
    CompetitorDiagnosis,
    Opportunity,
    PersonaIn,
    PersonaResult,
    Product,
    RadarCategory,
    Sources,
)
from app.services.anthropic_client import call_structured

logger = logging.getLogger("hearsay.insights")

_IMPACT_RANK = {"High": 3, "Medium": 2, "Low": 1}

_TOOL_NAME = "generate_competitive_insights"
_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "rivalWins": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short themes where competitors consistently beat the brand across the persona results.",
        },
        "brandWins": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short themes where the brand itself comes out ahead.",
        },
        "gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Buyer scenarios/query types the brand is essentially invisible in.",
        },
        "opportunities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["Critical gap", "Source gap", "Competitive", "Keyword gap", "Community"],
                    },
                    "impact": {"type": "string", "enum": ["High", "Medium", "Low"]},
                    "effort": {"type": "string", "enum": ["High", "Medium", "Low"]},
                    "detail": {"type": "string", "description": "1-2 sentences on what the data actually shows."},
                    "action": {"type": "string", "description": "A concrete, specific next step."},
                },
                "required": ["title", "type", "impact", "effort", "detail", "action"],
            },
            "description": "4-6 concrete, specific opportunities grounded in this run's actual data — never generic advice.",
        },
        "radarCategories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "integer"},
                },
                "required": ["name", "score"],
            },
            "description": (
                "5-6 dimensions that genuinely matter to how buyers in THIS category judge their choice "
                "(e.g. 'Ingredient Transparency' for a skincare brand, 'Room Comfort' for a hotel) — never "
                "default to generic software axes like 'Technical' or 'Enterprise' unless the brand actually "
                "is enterprise software. Score each 0-100 for how strongly this brand comes across on that "
                "dimension, based on the evidence given."
            ),
        },
    },
    "required": ["rivalWins", "brandWins", "gaps", "opportunities", "radarCategories"],
}

_SYSTEM = (
    "You are a competitive-intelligence analyst reviewing the completed results of an AI-visibility study: "
    "real AI-generated answers to realistic buyer questions, scored for whether/how/where a brand got "
    "mentioned versus its named competitors. Reason only from the evidence given — every claim should trace "
    "back to a specific persona result, product share, or cited source in the data, not generic industry "
    "knowledge. Be concrete and specific, never generic filler advice. The radar dimensions must be chosen "
    "to fit this specific brand's actual category and buyer context, not assumed to be software-buyer axes."
)


def _user_prompt(
    *,
    brand: str,
    industry: str,
    buyer_context: str | None,
    brand_summary: str | None,
    market: str | None,
    competitors: list[Competitor],
    personas: list[PersonaIn],
    results: dict[str, PersonaResult],
    products: list[Product],
    sources: Sources,
) -> str:
    persona_summaries = [
        {
            "title": persona.title,
            "mentioned": result.mentioned,
            "sentiment": result.sentiment.value,
            "rank": result.rank,
            "vis": result.vis,
            "opportunity": result.opportunity,
            "quote": result.quote,
        }
        for persona in personas
        if (result := results.get(persona.id)) is not None
    ]

    lines = [
        f"Brand: {brand}",
        f"Industry: {industry}",
        f"Competitors: {', '.join(c.name for c in competitors) if competitors else 'none specified'}",
    ]
    if buyer_context:
        lines.append(f"Buyer context: {buyer_context}")
    if brand_summary:
        lines.append(f"Brand summary: {brand_summary}")
    if market:
        lines.append(f"Market: {market}")
    lines.append(f"\nPersona results:\n{json.dumps(persona_summaries, indent=2)}")
    lines.append(
        f"\nShare of voice (product mention counts across all persona queries):\n"
        f"{json.dumps([p.model_dump(by_alias=True) for p in products], indent=2)}"
    )
    top_domains = sorted(sources.citations, key=lambda c: -c.count)[:8]
    lines.append(
        f"\nTop cited sources found while researching this brand's category:\n"
        f"{json.dumps([c.model_dump(by_alias=True) for c in top_domains], indent=2)}"
    )
    lines.append(
        "\nProduce the competitor diagnosis, a concrete opportunities list, and the brand-strength radar."
    )
    return "\n".join(lines)


async def generate_insights(
    *,
    api_key: str,
    brand: str,
    industry: str,
    buyer_context: str | None,
    brand_summary: str | None,
    market: str | None,
    competitors: list[Competitor],
    personas: list[PersonaIn],
    results: dict[str, PersonaResult],
    products: list[Product],
    sources: Sources,
) -> tuple[CompetitorDiagnosis, list[Opportunity], list[RadarCategory]]:
    settings = get_settings()
    empty = (CompetitorDiagnosis(rival_wins=[], brand_wins=[], gaps=[]), [], [])

    if not results:
        return empty

    try:
        result = await call_structured(
            api_key=api_key,
            model=settings.anthropic_fast_model,
            system=_SYSTEM,
            user=_user_prompt(
                brand=brand,
                industry=industry,
                buyer_context=buyer_context,
                brand_summary=brand_summary,
                market=market,
                competitors=competitors,
                personas=personas,
                results=results,
                products=products,
                sources=sources,
            ),
            tool_name=_TOOL_NAME,
            tool_description="Return the competitor diagnosis, opportunities, and radar categories.",
            input_schema=_INPUT_SCHEMA,
            max_tokens=2048,
        )
        # Forced tool-call output isn't a hard type guarantee (same defensive
        # posture as detect.py/category_gen.py — an array field has come back
        # as a comma-joined string before, which iterating naively would
        # silently shred into one entry per character) — a malformed
        # opportunity or radar entry must not crash the whole job, so
        # parsing failures fall through to the same fail-open empty result
        # as a call failure.
        def _clean_list(raw: object) -> list[str]:
            if not isinstance(raw, list):
                return []
            return [v for v in raw if isinstance(v, str)]

        diagnosis = CompetitorDiagnosis(
            rival_wins=_clean_list(result.get("rivalWins")),
            brand_wins=_clean_list(result.get("brandWins")),
            gaps=_clean_list(result.get("gaps")),
        )
        opportunities_raw = result.get("opportunities", [])
        opportunities = [
            Opportunity(**o) for o in (opportunities_raw if isinstance(opportunities_raw, list) else []) if isinstance(o, dict)
        ]
        # The Opportunities view promises "ranked by impact" — the model's
        # own generation order doesn't reliably match that, so enforce it
        # deterministically rather than trust output order.
        opportunities.sort(key=lambda o: _IMPACT_RANK.get(o.impact, 0), reverse=True)
        radar_raw = result.get("radarCategories", [])
        radar = [
            RadarCategory(**r) for r in (radar_raw if isinstance(radar_raw, list) else []) if isinstance(r, dict)
        ]
        return diagnosis, opportunities, radar
    except Exception:
        logger.exception("insights generation failed, returning empty insights")
        return empty
