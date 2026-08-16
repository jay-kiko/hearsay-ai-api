"""Turns a detected brand/industry/competitor set into a real, industry-
specific buyer persona set — replacing the frontend's static hardcoded 8.
One batched Claude call, same pattern as prompt_gen.py / detect.py.

Deliberately not hardcoded to B2B software: the "evaluating vendors" framing
that used to be baked in here only fits SaaS-style brands. What kind of real
choice this is (buying software, booking a hotel, picking a snack off a
shelf, hiring an agency...) comes from the caller's buyer_context — see
app.services.detect — so the same code produces sensible personas for tech
and non-tech brands alike.
"""
from __future__ import annotations

from app.config import get_settings
from app.models import GeneratedPersona
from app.services.anthropic_client import call_structured

_TOOL_NAME = "write_buyer_personas"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "personas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "A realistic persona name, e.g. 'Enterprise CTO' or 'Weekend Leisure Traveler'."},
                    "initials": {"type": "string", "description": "2-3 letter initials derived from the title, e.g. 'EC'."},
                    "desc": {
                        "type": "string",
                        "description": "One sentence summarizing what this persona is trying to decide or accomplish.",
                    },
                    "role": {"type": "string", "description": "One sentence on their position, scope, or life context."},
                    "pains": {"type": "string", "description": "Comma-separated realistic frustrations."},
                    "criteria": {
                        "type": "string",
                        "description": "Comma-separated things they'd weigh when making this choice.",
                    },
                },
                "required": ["title", "initials", "desc", "role", "pains", "criteria"],
            },
        }
    },
    "required": ["personas"],
}

_SYSTEM = (
    "You design realistic buyer personas for AI-visibility research. Given an industry, its "
    "named competitors, and the specific real-world choice buyers in this category are making "
    "(the buyer context), invent personas genuinely specific to that category and competitive "
    "landscape — different industries and competitor sets must produce visibly different "
    "personas, never generic reusable archetypes like 'Budget-Conscious Buyer' that could "
    "belong to any product. Ground every persona in the buyer context you're given: if it "
    "describes consumers choosing a hotel, product, or restaurant, write personas as real "
    "people making that kind of personal or experiential choice — not procurement officers "
    "'evaluating vendors'. Only frame personas as software/vendor evaluators when the buyer "
    "context actually describes a software or B2B procurement decision. When a brand summary "
    "and/or market is given, use them too — the brand summary refines what's actually being "
    "chosen between, and the market anchors personas in that specific region's real buyers, "
    "places, and vernacular rather than a generic global default. If the buyer context given "
    "actually describes more than one genuinely distinct kind of buyer (e.g. investors AND end "
    "customers, or businesses AND consumers) and the industry given is specifically about only "
    "one of them, write personas for that one industry only — do not blend the other buyer "
    "type's motivations into these personas just because the buyer context mentions it."
)

_DEFAULT_BUYER_CONTEXT = "People and organizations choosing what to use, buy, or work with in this industry."


def _user_prompt(
    industry: str,
    competitors: list[str],
    buyer_context: str,
    brand_summary: str | None,
    market: str | None,
    count: int,
) -> str:
    competitor_list = ", ".join(competitors) if competitors else "unspecified competitors"
    lines = [
        f"Industry: {industry}",
        f"Competitors in this category: {competitor_list}",
        f"Buyer context: {buyer_context}",
    ]
    if brand_summary:
        lines.append(f"Brand summary: {brand_summary}")
    if market:
        lines.append(f"Market/geography to ground personas in: {market}")
    lines.append(f"Write exactly {count} distinct personas for this category, true to all of the above.")
    return "\n".join(lines)


async def generate_personas(
    *,
    api_key: str,
    industry: str,
    competitors: list[str],
    buyer_context: str | None,
    brand_summary: str | None = None,
    market: str | None = None,
    persona_count: int | None,
) -> list[GeneratedPersona]:
    settings = get_settings()
    count = persona_count or settings.persona_count

    result = await call_structured(
        api_key=api_key,
        model=settings.anthropic_fast_model,
        system=_SYSTEM,
        user=_user_prompt(
            industry, competitors, buyer_context or _DEFAULT_BUYER_CONTEXT, brand_summary, market, count
        ),
        tool_name=_TOOL_NAME,
        tool_description="Return the written buyer personas.",
        input_schema=_INPUT_SCHEMA,
        max_tokens=3072,
    )

    personas: list[GeneratedPersona] = []
    for i, entry in enumerate(result.get("personas", []), start=1):
        title = entry.get("title") or f"Persona {i}"
        personas.append(
            GeneratedPersona(
                id=f"p{i}",
                title=title,
                initials=(entry.get("initials") or "".join(w[0] for w in title.split()[:2])).upper()[:3],
                desc=entry.get("desc", ""),
                role=entry.get("role", ""),
                pains=entry.get("pains", ""),
                criteria=entry.get("criteria", ""),
            )
        )
    return personas
