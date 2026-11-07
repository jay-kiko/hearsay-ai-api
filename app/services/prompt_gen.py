"""§03 Prompt generation — one batched Claude call writes every persona's
prompts in that persona's own voice, grounded in their stated pains and
decision criteria. Not a fill-in-the-blank template.

Deliberately not hardcoded to B2B software: framing every prompt as
"evaluating tools" only fits SaaS-style brands. The caller's buyer_context
(see app.services.detect) says what kind of real choice this actually is —
buying software, booking a hotel, picking a snack off a shelf, hiring an
agency — so the same code writes sensible questions for tech and non-tech
brands alike.
"""
from __future__ import annotations

import itertools
import json
import logging
import re
from dataclasses import dataclass

from app.config import get_settings
from app.models import PROMPT_MAX, Competitor, PersonaIn
from app.services.anthropic_client import call_structured
from app.services.common import DEFAULT_BUYER_CONTEXT, persona_payload
from app.services.scoring import find_mentions

logger = logging.getLogger("hearsay.prompt_gen")

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
    "You write realistic buyer-research questions. For each persona you are given, write "
    "the exact number of prompts requested, each one something that persona would plausibly "
    "type into an AI assistant while making the real-world choice described by the buyer "
    "context — not assumed to be software evaluation unless the buyer context actually says "
    "so. Ground every prompt in that persona's own situation (their description, role, pains "
    "and decision criteria) so different personas produce visibly different questions. Never "
    "mention the brand name — these are neutral buyer questions, not brand lookups. Never "
    "reveal that this is a test or evaluation.\n\n"
    "Sound like a real person, not a survey, and keep it short: most prompts 8-20 words, none over "
    "28. Real users type short, specific, slightly messy questions. Raise ONE concern or situation "
    "per prompt — never recite the persona's whole pain or criteria list, and never work through it "
    "in order. Work in concrete details the way a person would (a budget, a party size, a deadline, "
    "a place, what went wrong last time) instead of abstract qualities. Vary length and register: "
    "some prompts a terse fragment, some a short bit of context then a question; some casual, some "
    "more careful. Avoid stock AI phrasing: do not use 'I'm looking for', 'I want to find', 'what are "
    "the best options for', 'that prioritize', 'that offer', 'ideal', 'seamless', 'top-rated', "
    "'comprehensive', 'tailored', 'navigate', 'ensure', em dashes, or lists of three adjectives.\n\n"
    "Variety is required. The prompts for one persona must each lead with a different pain, "
    "criterion, or situation and use a different shape — for example a direct 'which X' question, a "
    "comparison of kinds of options (with vs without something, big vs small), a scenario ('we're "
    "doing Y, what do people usually use'), a constraint-led question ('under $X, no Y'), or a short "
    "keyword-style query. No two prompts anywhere in the output may start with the same three words, "
    "and no shape should be reused within a persona.\n\n"
    "Stay neutral about brands. Do not name any specific brand, product, ship, or company in a prompt "
    "— not the one being researched and not its competitors — and do not point at the researched brand "
    "indirectly (its flagship product, signature character, or a description only it fits). The "
    "assistant's answer is what should bring names up; a prompt that names some options but not "
    "others skews the measurement.\n\n"
    "Critical requirement: every prompt must be phrased so a genuinely helpful answer would "
    "naturally name specific real brands or products. Do not write purely conceptual or "
    "educational questions that a full answer could satisfy without naming a single product "
    "— e.g. avoid 'what does clean beauty even mean' or 'how is X percentage measured' with no "
    "recommendation angle. If a persona's criteria are abstract (ingredient transparency, "
    "certification standards, technical specs), still frame the question so the answer has to "
    "name options that meet that bar. This does not require a stock opener or a closing 'recommend "
    "one' — a scenario or a constraint that clearly calls for named options is enough.\n\n"
    "When a brand summary and/or market is given, use them too: the brand summary sharpens what "
    "category of recommendation is actually being sought, and the market means every prompt "
    "should read like it's coming from a real buyer in that specific region — natural local "
    "phrasing, places, and context, not a generic global-default question. If the buyer context "
    "given actually describes more than one genuinely distinct kind of buyer (e.g. investors AND "
    "end customers) and the industry given is specifically about only one of them, write prompts "
    "for that one industry only — a persona's own pains/criteria are the real guide here, don't "
    "pull in a different buyer type's motivation just because the buyer context mentions it."
)

@dataclass(frozen=True)
class CategoryPlan:
    """A product category the person picked, and how many of each persona's
    prompts are about it."""

    name: str
    buyer_context: str
    count: int


def _user_prompt(
    brand: str,
    industry: str,
    buyer_context: str,
    brand_summary: str | None,
    market: str | None,
    personas: list[PersonaIn],
    count: int,
    categories: list[CategoryPlan] | None = None,
    repeated: list[str] | None = None,
) -> str:
    lines = [f"Industry: {industry}"]
    if categories:
        plan, first = [], 1
        for c in categories:
            last = first + c.count - 1
            where = f"prompt {first}" if c.count == 1 else f"prompts {first}-{last}"
            plan.append(f"- {where}: {c.name}. Who is choosing: {c.buyer_context}")
            first = last + 1
        lines.append(
            "Categories, in this order for every persona (each prompt is about its category, and every "
            "prompt raises a different concern from the persona's others):\n" + "\n".join(plan)
        )
    else:
        lines.append(f"Buyer context: {buyer_context}")
    if brand_summary:
        lines.append(f"Brand summary: {brand_summary}")
    if market:
        lines.append(f"Market/geography to ground prompts in: {market}")
    lines.append(f"(Do not mention the brand '{brand}' in any prompt.)")
    lines.append(
        f"Write exactly {count} prompts per persona, true to all of the above. Each of a persona's "
        f"prompts takes a different angle and a different shape; no two prompts in the whole output "
        f"open the same way."
    )
    if repeated:
        lines.append(
            "A previous attempt repeated one question in different words for: " + ", ".join(repeated)
            + ". Give each of those personas' prompts a different concern, situation and detail, not a "
            "rewording of one scenario."
        )
    lines.append(f"\nPersonas:\n{json.dumps(persona_payload(personas), indent=2)}")
    return "\n".join(lines)


_STOPWORDS = frozenset(
    "a an and the of for to in on at with from is are be our my we i you your it its this that which what any "
    "some do does can has have had not but or so if as by just get got still really actually best good proper".split()
)
# Share of meaningful words two prompts have in common. Calibrated live: one
# scenario reworded scored 0.23-1.0 (categories each generated in their own
# call all asked about "13 and 16 yo think it's babyish" or "Ah Ma in a
# wheelchair plus 6 grandkids"); 144 distinct prompts from single calls
# scored at most 0.17.
_REPEAT_OVERLAP = 0.2


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


def near_duplicate_pairs(prompts: list[str]) -> list[tuple[int, int]]:
    """Index pairs of prompts that ask the same question in other words. One
    definition, used by generation and by evals.metrics."""
    out = []
    for (i, a), (j, b) in itertools.combinations(enumerate(prompts), 2):
        wa, wb = _content_words(a), _content_words(b)
        if wa | wb and len(wa & wb) / len(wa | wb) >= _REPEAT_OVERLAP:
            out.append((i, j))
    return out


def names_a_brand(
    text: str, brand: str, competitors: list[Competitor], brand_match_names: list[str] | None = None
) -> bool:
    """Whether a generated prompt names the brand or a known competitor. The
    instructions forbid it; this enforces it, because a prompt that names the
    brand measures nothing (invariant 5)."""
    # Caseless: a prompt typed "best app like xero" still names the brand.
    return bool(find_mentions(text, brand, competitors, brand_match_names or [brand], caseless=True))


def _fallback_prompt(industry: str, market: str | None) -> str:
    """Used only when the model returns nothing usable for a persona. Neutral,
    product-seeking, and it never reveals a persona or a test."""
    topic = industry.split(" — ")[-1].strip().lower() or "options"
    where = f" in {market}" if market else ""
    return f"Which {topic} do people{where} actually recommend, and why?"


def _clean_prompts(raw: object) -> list[str]:
    # Over-long prompts are dropped, not cut mid-question: they'd fail the
    # analysis request's limit, and the instructions ask for 28 words at most.
    if not isinstance(raw, list):
        return []
    return [p.strip() for p in raw if isinstance(p, str) and p.strip() and len(p.strip()) <= PROMPT_MAX]


async def generate_prompts(
    *,
    api_key: str,
    brand: str,
    industry: str,
    buyer_context: str | None,
    brand_summary: str | None = None,
    market: str | None = None,
    personas: list[PersonaIn],
    prompts_per_persona: int | None,
    competitors: list[Competitor] | None = None,
    brand_match_names: list[str] | None = None,
    categories: list[CategoryPlan] | None = None,
) -> dict[str, list[str]]:
    """Every persona's prompts in one call, categories included: a call sees
    only its own output, so one call per category wrote the persona's most
    obvious scenario once per category (confirmed live: three rewordings of
    "first cruise with a 4 and 7 year old"). A persona whose prompts still
    repeat one question is asked for once more and keeps the less repetitive
    set."""
    settings = get_settings()
    count = sum(c.count for c in categories) if categories else (prompts_per_persona or settings.prompts_per_persona)

    if not personas:
        return {}

    async def attempt(repeated: list[str] | None) -> dict[str, list[str]]:
        result = await call_structured(
            api_key=api_key,
            model=settings.anthropic_quality_model,
            system=_SYSTEM,
            user=_user_prompt(
                brand, industry, buyer_context or DEFAULT_BUYER_CONTEXT, brand_summary, market, personas, count,
                categories, repeated,
            ),
            tool_name=_TOOL_NAME,
            tool_description="Return the written prompts for every persona.",
            input_schema=_INPUT_SCHEMA,
        )

        # Tool output isn't a hard type guarantee: skip malformed entries rather
        # than KeyError into a 500.
        by_persona: dict[str, list[str]] = {}
        for entry in result.get("personas", []) if isinstance(result.get("personas"), list) else []:
            if isinstance(entry, dict) and isinstance(entry.get("personaId"), str):
                # Capped at the requested count: extras would push a persona past
                # the analysis request's per-persona limit.
                by_persona[entry["personaId"]] = _clean_prompts(entry.get("prompts"))[:count]

        out: dict[str, list[str]] = {}
        for persona in personas:
            prompts = by_persona.get(persona.id, [])
            clean = [p for p in prompts if not names_a_brand(p, brand, competitors or [], brand_match_names)]
            if len(clean) < len(prompts):
                logger.warning("dropped %d generated prompt(s) that named the brand or a competitor", len(prompts) - len(clean))
            out[persona.id] = clean
        return out

    output = await attempt(None)
    if not any(output.values()):
        # Nothing usable for anyone is a failed call, not a set of prompts: the
        # fallback question for every persona read as a real result.
        raise ValueError("prompt generation returned no usable prompts")
    repeated = [p for p in personas if near_duplicate_pairs(output[p.id])]
    if repeated:
        logger.warning("prompts repeated one question for %d persona(s); asking once more", len(repeated))
        try:
            retry = await attempt([p.title for p in repeated])
        except Exception:  # noqa: BLE001 - the first set still stands
            logger.exception("prompt retry failed; keeping the first set")
            retry = {}
        for p in repeated:
            again = retry.get(p.id) or []
            if len(again) >= len(output[p.id]) and len(near_duplicate_pairs(again)) < len(near_duplicate_pairs(output[p.id])):
                output[p.id] = again
    return {pid: prompts or [_fallback_prompt(industry, market)] for pid, prompts in output.items()}


_SEED_TOOL_NAME = "adapt_seed_prompt"
_SEED_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "personas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "personaId": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["personaId", "prompt"],
            },
        }
    },
    "required": ["personas"],
}

_SEED_SYSTEM = (
    "You are given one buyer-research question a user wrote themselves, plus a list of personas. For each "
    "persona, rewrite that exact question — same core ask, same recommendation-seeking intent — as that "
    "persona would plausibly phrase it, informed by their stated pains and decision criteria. Do not write "
    "a new, different question; adapt the given one. Keep the requirement that a genuinely helpful answer "
    "would naturally name specific real brands or products. Write each version the way that persona "
    "would actually type it — short, specific, one concern, no stock AI phrasing ('I'm looking for', "
    "'what are the best options for') — and make the versions read differently from one another, not "
    "as one template with swapped nouns. Never mention the brand this is being tested "
    "for, and never reveal that this is a test or evaluation."
)


def _seed_user_prompt(
    brand: str,
    industry: str,
    buyer_context: str,
    brand_summary: str | None,
    market: str | None,
    personas: list[PersonaIn],
    seed_prompt: str,
) -> str:
    lines = [
        f"Industry: {industry}",
        f"Buyer context: {buyer_context}",
    ]
    if brand_summary:
        lines.append(f"Brand summary: {brand_summary}")
    if market:
        lines.append(f"Market/geography to ground prompts in: {market}")
    lines.append(f"(Do not mention the brand '{brand}' in any prompt.)")
    lines.append(f"Seed question to adapt: {seed_prompt}")
    lines.append(f"\nPersonas:\n{json.dumps(persona_payload(personas), indent=2)}")
    return "\n".join(lines)


async def adapt_seed_prompt(
    *,
    api_key: str,
    brand: str,
    industry: str,
    buyer_context: str | None = None,
    brand_summary: str | None = None,
    market: str | None = None,
    personas: list[PersonaIn],
    seed_prompt: str,
    competitors: list[Competitor] | None = None,
    brand_match_names: list[str] | None = None,
) -> dict[str, str]:
    settings = get_settings()

    if not personas or not seed_prompt.strip():
        return {}

    result = await call_structured(
        api_key=api_key,
        model=settings.anthropic_quality_model,
        system=_SEED_SYSTEM,
        user=_seed_user_prompt(
            brand, industry, buyer_context or DEFAULT_BUYER_CONTEXT, brand_summary, market, personas, seed_prompt
        ),
        tool_name=_SEED_TOOL_NAME,
        tool_description="Return the adapted prompt for every persona.",
        input_schema=_SEED_INPUT_SCHEMA,
    )

    by_persona = {
        entry["personaId"]: entry["prompt"].strip()
        for entry in (result.get("personas") if isinstance(result.get("personas"), list) else [])
        if isinstance(entry, dict) and isinstance(entry.get("personaId"), str) and isinstance(entry.get("prompt"), str)
    }
    # An adaptation that adds a brand or competitor the user's own seed didn't
    # have falls back to the seed as written.
    seed_names = names_a_brand(seed_prompt, brand, competitors or [], brand_match_names)
    out = {}
    for p in personas:
        adapted = by_persona.get(p.id, "")
        if adapted and not seed_names and names_a_brand(adapted, brand, competitors or [], brand_match_names):
            adapted = ""
        if len(adapted) > PROMPT_MAX:
            adapted = ""
        out[p.id] = adapted or seed_prompt
    return out
