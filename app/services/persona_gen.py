"""Turns a detected brand/industry/competitor set into a real, industry-
specific buyer persona set — replacing the frontend's static hardcoded 8.
One batched Claude call, same pattern as prompt_gen.py / detect.py.
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
                    "title": {"type": "string", "description": "A job-title-like persona name, e.g. 'Enterprise CTO'."},
                    "initials": {"type": "string", "description": "2-3 letter initials derived from the title, e.g. 'EC'."},
                    "desc": {
                        "type": "string",
                        "description": "One sentence summarizing what this persona evaluates tools for.",
                    },
                    "role": {"type": "string", "description": "One sentence on their position and scope."},
                    "pains": {"type": "string", "description": "Comma-separated realistic frustrations."},
                    "criteria": {
                        "type": "string",
                        "description": "Comma-separated decision criteria they'd weigh when choosing a vendor.",
                    },
                },
                "required": ["title", "initials", "desc", "role", "pains", "criteria"],
            },
        }
    },
    "required": ["personas"],
}

_SYSTEM = (
    "You design realistic buyer personas for AI-visibility research. Given an industry "
    "and its named competitors, invent personas genuinely specific to that category and "
    "competitive landscape — different industries and competitor sets must produce "
    "visibly different personas, never generic reusable archetypes like 'Budget-Conscious "
    "Buyer' that could belong to any product. Ground pains and criteria in how buyers in "
    "this specific category actually evaluate and compare vendors."
)


def _user_prompt(industry: str, competitors: list[str], count: int) -> str:
    competitor_list = ", ".join(competitors) if competitors else "unspecified competitors"
    return (
        f"Industry: {industry}\n"
        f"Competitors in this category: {competitor_list}\n"
        f"Write exactly {count} distinct buyer personas for this category."
    )


async def generate_personas(
    *,
    api_key: str,
    industry: str,
    competitors: list[str],
    persona_count: int | None,
) -> list[GeneratedPersona]:
    settings = get_settings()
    count = persona_count or settings.persona_count

    result = await call_structured(
        api_key=api_key,
        model=settings.anthropic_fast_model,
        system=_SYSTEM,
        user=_user_prompt(industry, competitors, count),
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
