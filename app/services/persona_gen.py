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

from uuid import uuid4

from app.config import get_settings
from app.models import Competitor, GeneratedPersona
from app.services.anthropic_client import call_structured
from app.services.common import DEFAULT_BUYER_CONTEXT
from app.services.scoring import without_names

_TOOL_NAME = "write_buyer_personas"

# One persona's fields, shared by the batch and the custom-expansion schemas.
_PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "A short, plain persona name of 2-4 words, e.g. 'Enterprise CTO' or 'Weekend Leisure Traveler'."},
        "initials": {"type": "string", "description": "2-3 letter initials derived from the title, e.g. 'EC'."},
        "desc": {
            "type": "string",
            "description": "One plain sentence, at most 15 words: what this persona is trying to decide.",
        },
        "role": {"type": "string", "description": "At most 10 words on their position or life context."},
        "pains": {"type": "string", "description": "2-3 comma-separated frustrations, 12 words total at most. Short phrases, no sentences."},
        "criteria": {
            "type": "string",
            "description": "2-3 comma-separated things they weigh, 10 words total at most. Short phrases.",
        },
    },
    "required": ["title", "initials", "desc", "role", "pains", "criteria"],
}

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"personas": {"type": "array", "items": _PERSONA_SCHEMA}},
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

_BREVITY_RULES = (
    " Keep every field short and plain: write the way a person would jot a note about a friend, "
    "not a marketing persona deck. Short phrases, concrete details, no corporate or AI-sounding "
    "words ('navigate', 'leverage', 'seamless', 'holistic', 'ensure'), no em dashes."
)

# Generated (not user-written) personas feed prompt generation; a brand or
# competitor named here leaks into the prompts and skews what we measure.
_NO_BRANDS_RULE = (
    " Never name any specific brand, product, ship or company in any field — not the brand being "
    "researched (a summary of it may be given as background) and not its competitors. Describe the "
    "situation and what the person cares about, not which options they are choosing between. Each "
    "persona is a buyer weighing the market's options, never a fan or existing customer of one brand, "
    "and never built around something only one brand offers (its characters, licensed content or "
    "signature features): their questions would then point at that brand."
)

def _user_prompt(
    industry: str,
    competitors: list[Competitor],
    buyer_context: str,
    brand_summary: str | None,
    market: str | None,
    count: int,
) -> str:
    competitor_list = ", ".join(c.name for c in competitors) if competitors else "unspecified competitors"
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
    competitors: list[Competitor],
    buyer_context: str | None,
    brand_summary: str | None = None,
    market: str | None = None,
    persona_count: int | None,
    brand: str | None = None,
    brand_names: list[str] | None = None,
) -> list[GeneratedPersona]:
    """`brand` and `brand_names` (aliases and parent names such as "Disney")
    are only screened out: persona text feeds prompt generation, and a name
    left in it leaks into the prompts (confirmed live: "Adult Disney
    Superfan")."""
    settings = get_settings()
    names = [brand, *(brand_names or [])] if brand else []

    def neutral(text: str) -> str:
        return without_names(text, brand or "", competitors, names)
    count = persona_count or settings.persona_count

    result = await call_structured(
        api_key=api_key,
        model=settings.anthropic_quality_model,
        system=_SYSTEM + _BREVITY_RULES + _NO_BRANDS_RULE,
        user=_user_prompt(
            industry, competitors, buyer_context or DEFAULT_BUYER_CONTEXT, brand_summary, market, count
        ),
        tool_name=_TOOL_NAME,
        tool_description="Return the written buyer personas.",
        input_schema=_INPUT_SCHEMA,
    )

    personas: list[GeneratedPersona] = []
    # Tool output isn't a hard type guarantee: an entry that isn't an object
    # is skipped (confirmed live: one crashed persona generation outright).
    entries = result.get("personas") if isinstance(result.get("personas"), list) else []
    for i, entry in enumerate((e for e in entries if isinstance(e, dict)), start=1):
        raw_title = entry.get("title") or ""
        title = neutral(raw_title) or f"Persona {i}"
        # Initials follow the title when a name was taken out of it.
        initials = entry.get("initials") if title == raw_title else None
        personas.append(
            GeneratedPersona(
                id=f"p{i}",
                title=title,
                initials=(initials or "".join(w[0] for w in title.split()[:2])).upper()[:3],
                desc=neutral(entry.get("desc", "")),
                role=neutral(entry.get("role", "")),
                pains=neutral(entry.get("pains", "")),
                criteria=neutral(entry.get("criteria", "")),
            )
        )
    return personas


# ── Custom persona expansion ─────────────────────────────────────────
# Lets a user type one free-text description instead of filling in
# title/role/pains/criteria by hand. The output is a GeneratedPersona
# just like the batch above, so once expanded, a custom persona is
# structurally identical to an AI-generated one — prompt_gen.py and
# everything downstream never need to know which path a persona came
# from.

_EXPAND_TOOL_NAME = "write_persona_from_description"

_EXPAND_INPUT_SCHEMA = _PERSONA_SCHEMA

_EXPAND_SYSTEM = (
    "You turn a short user-written description of a buyer persona into one complete, "
    "structured persona for AI-visibility research. The user's description is the source of "
    "truth for who this persona is — never contradict or genericize it, only fill in what it "
    "leaves unstated. When the description is sparse, keep the fill-in modest and plausible "
    "for the category rather than inventing distinctive biographical specifics the user never "
    "gave — the more the user specified, the more the persona should read exactly as described. "
    "Ground the fill-in details in the given industry, competitors, and buyer "
    "context: if the buyer context describes consumers choosing a hotel, product, or "
    "restaurant, write the persona as a real person making that kind of personal or "
    "experiential choice — not a procurement officer 'evaluating vendors'. Only frame the "
    "persona as a software/vendor evaluator when the buyer context actually describes a "
    "software or B2B procurement decision. When a brand summary and/or market is given, use "
    "them too — the brand summary refines what's actually being chosen between, and the "
    "market anchors the persona in that specific region's real buyers, places, and vernacular "
    "rather than a generic global default."
)


def _expand_user_prompt(
    description: str,
    industry: str,
    competitors: list[Competitor],
    buyer_context: str,
    brand_summary: str | None,
    market: str | None,
) -> str:
    competitor_list = ", ".join(c.name for c in competitors) if competitors else "unspecified competitors"
    lines = [
        f"User's persona description: {description}",
        f"Industry: {industry}",
        f"Competitors in this category: {competitor_list}",
        f"Buyer context: {buyer_context}",
    ]
    if brand_summary:
        lines.append(f"Brand summary: {brand_summary}")
    if market:
        lines.append(f"Market/geography to ground the persona in: {market}")
    lines.append("Write one persona true to the description and all of the above.")
    return "\n".join(lines)


async def expand_custom_persona(
    *,
    api_key: str,
    description: str,
    industry: str,
    competitors: list[Competitor],
    buyer_context: str | None,
    brand_summary: str | None = None,
    market: str | None = None,
) -> GeneratedPersona:
    settings = get_settings()

    result = await call_structured(
        api_key=api_key,
        model=settings.anthropic_quality_model,
        # The no-brands rule too: expanded fields feed prompt generation, and the
        # competitor list given as context leaked rival names into them.
        system=_EXPAND_SYSTEM + _BREVITY_RULES + _NO_BRANDS_RULE,
        user=_expand_user_prompt(
            description, industry, competitors, buyer_context or DEFAULT_BUYER_CONTEXT, brand_summary, market
        ),
        tool_name=_EXPAND_TOOL_NAME,
        tool_description="Return the written persona.",
        input_schema=_EXPAND_INPUT_SCHEMA,
    )

    title = result.get("title") or "Custom Persona"
    return GeneratedPersona(
        id=f"custom-{uuid4().hex[:8]}",
        title=title,
        initials=(result.get("initials") or "".join(w[0] for w in title.split()[:2])).upper()[:3],
        desc=result.get("desc", ""),
        role=result.get("role", ""),
        pains=result.get("pains", ""),
        criteria=result.get("criteria", ""),
    )
