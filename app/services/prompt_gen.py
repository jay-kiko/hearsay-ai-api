"""§03 Prompt generation — one batched Claude call writes every persona's
prompts in that persona's own voice, grounded in their stated pains and
decision criteria. Not a fill-in-the-blank template.
"""
from __future__ import annotations

import json

from app.config import get_settings
from app.models import PersonaIn
from app.services.anthropic_client import call_structured

_TOOL_NAME = "write_persona_prompts"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "personas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "personaId": {"type": "string"},
                    "prompts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["personaId", "prompts"],
            },
        }
    },
    "required": ["personas"],
}

_SYSTEM = (
    "You write realistic buyer-research questions. For each persona you are given, "
    "write the exact number of prompts requested, each one a first-person question "
    "that persona would plausibly type into an AI assistant while evaluating tools "
    "in the given industry. Ground every prompt in that persona's stated pains and "
    "decision criteria so different personas produce visibly different questions. "
    "Never mention the brand name — these are neutral buyer questions, not brand "
    "lookups. Never reveal that this is a test or evaluation."
)


def _user_prompt(brand: str, industry: str, personas: list[PersonaIn], count: int) -> str:
    persona_payload = [
        {
            "personaId": p.id,
            "title": p.title,
            "role": p.role,
            "pains": p.pains,
            "criteria": p.criteria,
        }
        for p in personas
    ]
    return (
        f"Industry: {industry}\n"
        f"(Do not mention the brand '{brand}' in any prompt.)\n"
        f"Write exactly {count} prompts per persona.\n\n"
        f"Personas:\n{json.dumps(persona_payload, indent=2)}"
    )


async def generate_prompts(
    *,
    api_key: str,
    brand: str,
    industry: str,
    personas: list[PersonaIn],
    prompts_per_persona: int | None,
) -> dict[str, list[str]]:
    settings = get_settings()
    count = prompts_per_persona or settings.prompts_per_persona

    if not personas:
        return {}

    result = await call_structured(
        api_key=api_key,
        model=settings.anthropic_model,
        system=_SYSTEM,
        user=_user_prompt(brand, industry, personas, count),
        tool_name=_TOOL_NAME,
        tool_description="Return the written prompts for every persona.",
        input_schema=_INPUT_SCHEMA,
        max_tokens=2048,
    )

    by_persona = {entry["personaId"]: entry.get("prompts", []) for entry in result.get("personas", [])}
    # Guarantee every requested persona id comes back with something, even if the
    # model skipped one — fall back to a single templated question rather than 500.
    output: dict[str, list[str]] = {}
    for persona in personas:
        prompts = by_persona.get(persona.id) or [
            f"As a {persona.title}, what should I look for in {industry.lower()} tools?"
        ]
        output[persona.id] = prompts
    return output
