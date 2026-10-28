"""§07 Competitive insights — one AI reasoning pass over a completed analysis.

Competitor diagnosis (why rivals win / where the brand wins / gaps), the
Opportunities list, and the brand-strength radar all need judgment over the
whole run (every persona's mentions/sentiment/rank, share of voice, cited
sources) — not something deterministic math (app.services.aggregation) or
search grounding alone produces. Bundled into one structured call rather than
three since they all reason over identical evidence — same principle as
category_gen.py's one call covering several facets.

This is synthesis over already-collected data, not the organic-mention
signal being measured (that's pipeline.py's job). It runs once per job on the
quality model (its text is what a client reads), is fed the same prompt-level
counts and measured source data the UI shows so its claims can't contradict
them, and fails open (empty results) rather than sinking the job.
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
            "description": "Themes where competitors consistently beat the brand across the persona results. Each at most 12 words.",
        },
        "brandWins": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Themes where the brand itself comes out ahead. Each at most 12 words.",
        },
        "gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Buyer scenarios/query types the brand is essentially invisible in. Each at most 12 words.",
        },
        "opportunities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "At most 8 words."},
                    "type": {
                        "type": "string",
                        "enum": ["Critical gap", "Source gap", "Competitive", "Keyword gap", "Community"],
                    },
                    "impact": {"type": "string", "enum": ["High", "Medium", "Low"]},
                    "effort": {"type": "string", "enum": ["High", "Medium", "Low"]},
                    "detail": {"type": "string", "description": "One sentence, at most 25 words, on what the data actually shows."},
                    "action": {"type": "string", "description": "One concrete, specific next step, at most 20 words."},
                },
                "required": ["title", "type", "impact", "effort", "detail", "action"],
            },
            "description": "4-6 concrete, specific opportunities grounded in this run's actual data — never generic advice.",
        },
        "radarCategories": {
            "type": "array",
            "minItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "integer"},
                },
                "required": ["name", "score"],
            },
            "description": (
                "ALWAYS return at least 3-6 dimensions, even for a brand with thin or zero organic mentions — "
                "never an empty array. Pick dimensions that genuinely matter to how buyers in THIS category "
                "judge their choice (e.g. 'Ingredient Transparency' for a skincare brand, 'Room Comfort' for a "
                "hotel) — never default to generic software axes like 'Technical' or 'Enterprise' unless the "
                "brand actually is enterprise software. Score each 0-100 for how strongly this brand comes "
                "across on that dimension. When the persona evidence is rich, ground scores in it directly; "
                "when it's thin (few or no mentions), still choose sensible dimensions from the brand/industry "
                "context and score them conservatively low rather than omitting the dimension — a brand's "
                "near-invisibility is itself a real (low) score on every dimension, not a reason to skip radar."
            ),
        },
    },
    "required": ["rivalWins", "brandWins", "gaps", "opportunities", "radarCategories"],
}

_SYSTEM = (
    "You are a competitive-intelligence analyst reviewing the completed results of an AI-visibility study: "
    "real AI-generated answers to realistic buyer questions, scored for whether/how/where a brand got "
    "mentioned versus its named competitors. For rivalWins/brandWins/gaps/opportunities, reason only from the "
    "evidence given — every claim should trace back to a specific persona result, product share, or cited "
    "source in the data, not generic industry knowledge. Be concrete and specific, never generic filler "
    "advice. The one exception is the radar: even when a brand has thin or zero organic mentions to reason "
    "from directly, still choose dimensions from the brand/industry/buyer context and score them (low, if "
    "that's what the near-invisibility warrants) — never leave radar empty just because the rest of the "
    "evidence is thin. Radar dimensions must be chosen to fit this specific brand's actual category and buyer "
    "context, not assumed to be software-buyer axes. A 'Source gap' or 'Community' opportunity must name a "
    "specific source from the list given and cite its measured gap (e.g. rivals named there, brand absent); "
    "if no source list is given, don't produce those types. Write like a sharp colleague, not a consultant "
    "deck: short, plain sentences with the specific number or example up front. No filler openers "
    "('It is important to', 'Consider'), no jargon ('leverage', 'ensure', 'robust', 'holistic', "
    "'synergy'), no em dashes."
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
    # Prompt-level, like every number the UI shows ("mentioned in 4 of 12
    # queries") — a persona-level boolean let the text say "2 of 4 personas".
    persona_summaries = []
    for persona in personas:
        result = results.get(persona.id)
        if result is None:
            continue
        exchanges = result.exchanges or []
        persona_summaries.append({
            "title": persona.title,
            "promptsMentioningBrand": f"{sum(1 for ex in exchanges if ex.mentioned)} of {len(exchanges)}",
            "exchanges": [
                {"prompt": ex.prompt, "mentioned": ex.mentioned, "sentiment": ex.sentiment.value,
                 "rank": ex.rank, "quote": ex.quote}
                for ex in exchanges
            ],
        })
    all_exchanges = [ex for r in results.values() for ex in (r.exchanges or [])]

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
    lines.append(
        f"\nThe brand was mentioned in {sum(1 for ex in all_exchanges if ex.mentioned)} of "
        f"{len(all_exchanges)} buyer queries. Use prompt-level counts like this, never persona counts."
    )
    lines.append(f"\nPersona results:\n{json.dumps(persona_summaries, indent=2)}")
    lines.append(
        f"\nShare of voice (product mention counts across all persona queries):\n"
        f"{json.dumps([p.model_dump(by_alias=True) for p in products], indent=2)}"
    )
    # Measured per source: who surfaced it, whether the brand and each rival
    # are named on it, and the gap. Source/community opportunities must come
    # from these, not be guessed.
    source_rows = [
        {"domain": i.domain, "type": i.category, "reach": i.influence, "surfacedBy": i.personas,
         "brandVisibility": i.visibility, "brandMentions": i.mentions,
         "competitorMentions": i.competitor_mentions, "gap": i.gap}
        for i in sources.source_intel[:12]
    ]
    lines.append(
        "\nIndependent sources that buyers' own questions surfaced (searched without the brand; brand and "
        f"competitor mentions measured on each page):\n{json.dumps(source_rows, indent=2) if source_rows else 'none found'}"
    )
    lines.append(
        "\nProduce the competitor diagnosis, a concrete opportunities list, and the brand-strength radar."
    )
    return "\n".join(lines)


def _parse_each(raw: object, model: type) -> list:
    out = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(model(**entry))
        except Exception:  # noqa: BLE001 - skip one bad entry, keep the rest
            logger.warning("skipping malformed %s entry: %r", model.__name__, entry)
    return out


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
            model=settings.anthropic_quality_model,
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
            # Same failure mode confirmed live in grounding.py's source
            # enrichment call: rivalWins/brandWins/gaps/opportunities/
            # radarCategories in one JSON tool call can legitimately exceed
            # 2048 tokens for a verbose run (several detailed opportunities
            # alone can approach it), and radarCategories — last in the
            # schema — is exactly what gets cut when that happens.
            max_tokens=4096,
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
        # Validated entry by entry: one malformed opportunity (a bad enum
        # value) used to raise and empty the whole diagnosis/radar with it.
        opportunities = _parse_each(result.get("opportunities"), Opportunity)
        # The Opportunities view promises "ranked by impact" — the model's
        # own generation order doesn't reliably match that, so enforce it
        # deterministically rather than trust output order.
        opportunities.sort(key=lambda o: _IMPACT_RANK.get(o.impact, 0), reverse=True)
        radar = _parse_each(result.get("radarCategories"), RadarCategory)
        for r in radar:
            r.score = max(0, min(100, r.score))
        return diagnosis, opportunities, radar
    except Exception:
        logger.exception("insights generation failed, returning empty insights")
        return empty
