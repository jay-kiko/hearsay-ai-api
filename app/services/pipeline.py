"""§04 Per-persona pipeline.

For every prompt belonging to a persona: one Claude call for the buyer-style
answer, one structured Claude call classifying sentiment toward the brand and
pulling a quote, then deterministic parsing/scoring (app.services.scoring).
A persona can carry several prompts (see §03) — those per-prompt analyses are
run concurrently. PersonaResult's top-level fields (prompt/quote/parts/vis/
rank/sentiment) stay a single representative/aggregated view for backward
compatibility, but every individual per-prompt analysis is also exposed via
`exchanges` — the score was always the average across all of them; before
this, only the winning one was ever visible.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from dataclasses import dataclass

from app.config import get_settings
from app.models import Competitor, PersonaExchange, PersonaIn, PersonaResult, Sentiment
from app.services.anthropic_client import call_structured, call_text
from app.services.common import DEFAULT_BUYER_CONTEXT
from app.services.scoring import analyze_answer, name_matches

logger = logging.getLogger("hearsay.pipeline")

def _answer_system(buyer_context: str | None, market: str | None) -> str:
    # The prompt text is the *only* thing this call sees — no brand, no
    # industry, no geography. Without buyer_context threaded in, a
    # regionally-specific question with no explicit country in its own
    # wording (e.g. "affordable weekly rooms near the hospital district")
    # silently defaults to US-centric answers, confirmed live for a
    # Philippines-market brand. buyer_context/market are brand-agnostic by
    # construction, so surfacing them here never leaks the brand identity or
    # reveals this is a test — brand_summary is NOT threaded in here on
    # purpose, since it names the brand directly and this is the one call
    # that must measure whether the brand comes up organically.
    parts = [f"Real-world context for who's asking and what they're choosing between: {buyer_context or DEFAULT_BUYER_CONTEXT}"]
    if market:
        parts.append(f"Answer as if for someone in this market: {market} — use real local context where relevant.")
    return (
        "You are a knowledgeable analyst helping someone research real options. "
        + " ".join(parts)
        + " Answer with specific, real product/brand names wherever relevant — "
        "concrete recommendations, not generic advice. Keep the answer to 2-4 sentences."
    )

_SENTIMENT_TOOL_NAME = "classify_brand_sentiment"
_SENTIMENT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["Positive", "Neutral", "Negative"]},
        "quote": {
            "type": "string",
            "description": "The single sentence from the answer that best represents its stance on the brand, verbatim. Empty string if the brand isn't mentioned.",
        },
        "mentionText": {
            "type": "string",
            "description": (
                "The exact substring, copied verbatim from the answer, that identifies the brand — "
                "this can be the brand name itself, or a specific product/model name you recognize "
                "as belonging to it even if it wasn't in the known-names list (e.g. a phone model "
                "number for a phone manufacturer). Must be a specific name, never a generic word "
                "like 'phone' or 'brand'. Empty string if the brand isn't mentioned under any name."
            ),
        },
        "competitorMentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "competitor": {
                        "type": "string",
                        "description": "Must exactly match one of the known competitor names given.",
                    },
                    "mentionText": {
                        "type": "string",
                        "description": (
                            "The exact substring, verbatim from the answer, that identifies this "
                            "competitor under a specific product/model name NOT already in its "
                            "known name list. Never a generic word."
                        ),
                    },
                },
                "required": ["competitor", "mentionText"],
            },
            "description": (
                "Any known competitor mentioned under a product/model name you recognize as "
                "belonging to it but that isn't already in its known name list. Empty array if none."
            ),
        },
    },
    "required": ["sentiment", "quote", "mentionText", "competitorMentions"],
}


def _sentiment_system(brand: str, brand_match_names: list[str], competitors: list[Competitor]) -> str:
    # Must recognize the same aliases (sub-brands, product lines, short forms)
    # as the deterministic mention detection in scoring.py, or this call
    # disagrees with it — e.g. an answer naming "Redmi Note 13" gets marked
    # mentioned=True downstream while sentiment stays stuck at Neutral
    # because this prompt only knew to look for the literal brand name.
    #
    # The known-names list can never be exhaustive for a category with many
    # product lines (phone models, etc.) — mentionText lets this same call
    # catch an unlisted product/model it recognizes from its own knowledge,
    # instead of silently missing anything not pre-guessed at detect time.
    names = ", ".join(f"'{n}'" for n in (brand_match_names or [brand]))
    competitor_lines = (
        "; ".join(f"{c.name} (known names: {', '.join(c.match_names)})" for c in competitors)
        if competitors
        else "none"
    )
    return (
        f"Classify the sentiment of the following AI-generated answer specifically "
        f"toward the brand '{brand}' — not the general tone of the answer. The brand may be "
        f"referred to by any of these known names, all of which count as the brand: {names}. It "
        f"may also be referred to by a specific product or model name not in that list that you "
        f"recognize as belonging to this brand — treat that as a mention too. If the brand isn't "
        f"mentioned under any name, sentiment must be 'Neutral', quote must be '', and mentionText "
        f"must be ''.\n\n"
        f"Separately, here are the known competitors and their known name variants: "
        f"{competitor_lines}. The known-name lists can't be exhaustive for a category with many "
        f"product lines — if the answer names a specific product or model you recognize as "
        f"belonging to one of these competitors, but under a name not already in its known list, "
        f"report it in competitorMentions using that competitor's exact name as given above."
    )


_MIN_ALIAS_LEN = 3
_ARTICLES = {"the", "a", "an"}
_SENTENCE_BREAKS = ".!?:;\n-*•#>|"


def _head_word(alias: str) -> str:
    """The distinctive first word of a name: 'disney' for 'Disney Cruise
    Line', 'ordinary' for 'The Ordinary'."""
    words = [w for w in re.findall(r"[a-z0-9]+", alias.lower()) if w not in _ARTICLES]
    return words[0] if words else ""


def _starts_sentence(text: str, position: int) -> bool:
    before = text[:position].rstrip(" \t")
    return not before or before[-1] in _SENTENCE_BREAKS


def _written_like_a_name(answer: str, position: int, occurrence: str) -> bool:
    """A digit, a capital past the first letter ('iPhone', 'Note 13'), or a
    leading capital where grammar doesn't force one. 'Cruise' opening a
    sentence or a bullet is just a word."""
    if any(ch.isdigit() for ch in occurrence) or any(ch.isupper() for ch in occurrence[1:]):
        return True
    return occurrence[:1].isupper() and not _starts_sentence(answer, position)


def _overlaps(a: str, b: str) -> bool:
    return bool(name_matches(a, b) or name_matches(b, a))


def verified_alias(candidate: object, answer: str, own_names: list[str], other_entities: list[str]) -> str | None:
    """A model-suggested product/model name for an entity, accepted only if
    the answer really names it and it plausibly belongs to that entity.
    Without this, a generic word ("cruise") or a rival's unlisted product
    ("Icon of the Seas") became a brand alias and turned a Royal Caribbean
    answer into mentioned=True, rank 1 (invariant 2: a model never fills a
    measured field).

    Accepted when the candidate:
      * appears in the answer under the same matching rule as every mention,
        at least once written like a name (see _written_like_a_name);
      * shares the head word of one of the entity's known names ('Disney
        Wish' for 'Disney Cruise Line', 'Redmi Note 13' for alias 'Redmi') —
        a product sold under an unrelated name is left to the detect-time
        alias list rather than taken on the model's word;
      * doesn't overlap another entity's name as a whole word (which would
        steal that entity's mention)."""
    if not isinstance(candidate, str):
        return None
    name = candidate.strip()
    if len(name) < _MIN_ALIAS_LEN:
        return None
    if not any(_written_like_a_name(answer, pos, occ) for pos, occ in name_matches(name, answer)):
        return None
    heads = {_head_word(n) for n in own_names} - {""}
    if not heads & set(re.findall(r"[a-z0-9]+", name.lower())):
        return None
    if any(other.strip() and _overlaps(name, other) for other in other_entities):
        return None
    return name


def _apply_competitor_mentions(
    competitors: list[Competitor], raw_mentions: object, answer: str, brand_names: list[str]
) -> list[Competitor]:
    """Builds a per-answer copy of `competitors` with any LLM-recognized,
    previously-unlisted product/model names appended to the matching
    competitor's match_names — each one checked by verified_alias first. Never
    mutates the input list — it's the same object shared across every
    concurrent prompt/persona task in this run."""
    if not isinstance(raw_mentions, list):
        return competitors

    extra_by_name: dict[str, list[str]] = {}
    for entry in raw_mentions:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("competitor", "")).strip()
        claimed = next((c for c in competitors if c.name.lower() == name.lower()), None)
        if claimed is None:
            continue
        others = brand_names + [
            alias for c in competitors if c is not claimed for alias in (c.name, *c.match_names)
        ]
        text = verified_alias(entry.get("mentionText"), answer, [claimed.name, *claimed.match_names], others)
        if name and text:
            extra_by_name.setdefault(name.lower(), []).append(text)

    if not extra_by_name:
        return competitors

    effective: list[Competitor] = []
    for c in competitors:
        extra = extra_by_name.get(c.name.lower(), [])
        if not extra:
            effective.append(c)
            continue
        match_names = list(c.match_names)
        for text in extra:
            if text.lower() not in (n.lower() for n in match_names):
                match_names.append(text)
        # model_copy, not a new Competitor: per-answer names may run past the
        # request-size cap on match_names, which only bounds what callers send.
        effective.append(c.model_copy(update={"match_names": match_names}))
    return effective


@dataclass
class _PromptAnalysis:
    prompt: str
    mentioned: bool
    sentiment: Sentiment
    rank: int | None
    vis: int
    quote: str
    parts: list


async def _analyze_one_prompt(
    *,
    prompt: str,
    brand: str,
    brand_match_names: list[str],
    competitors: list[Competitor],
    buyer_context: str | None,
    market: str | None,
    api_key: str,
    answer_model: str,
    fast_model: str,
) -> _PromptAnalysis:
    answer_task = call_text(
        api_key=api_key, model=answer_model, system=_answer_system(buyer_context, market), user=prompt
    )
    answer_text = await answer_task

    # Sentiment classification is mechanical extraction, not the signal being
    # measured — runs on the cheaper fast_model, not the answer model.
    sentiment_result = await call_structured(
        api_key=api_key,
        model=fast_model,
        system=_sentiment_system(brand, brand_match_names, competitors),
        user=answer_text or "(empty response)",
        tool_name=_SENTIMENT_TOOL_NAME,
        tool_description="Classify sentiment toward the brand and extract a supporting quote.",
        input_schema=_SENTIMENT_INPUT_SCHEMA,
        max_tokens=512,
    )
    # An off-enum value ("positive") would raise and fail the prompt; read it
    # as Neutral instead.
    raw_sentiment = sentiment_result.get("sentiment")
    sentiment = Sentiment(raw_sentiment) if raw_sentiment in Sentiment._value2member_map_ else Sentiment.neutral
    quote = sentiment_result.get("quote", "") or ""
    # mentionText is per-answer and LLM-supplied (may recognize an unlisted
    # product/model name) — appended only once verified against the answer,
    # and on top of the static detect-time aliases, never instead of them.
    effective_match_names = list(brand_match_names or [brand])
    competitor_names = [alias for c in competitors for alias in (c.name, *c.match_names)]
    mention_text = verified_alias(
        sentiment_result.get("mentionText"), answer_text, [brand, *effective_match_names], competitor_names
    )
    if mention_text and mention_text.lower() not in (n.lower() for n in effective_match_names):
        effective_match_names.append(mention_text)

    effective_competitors = _apply_competitor_mentions(
        competitors, sentiment_result.get("competitorMentions", []), answer_text, effective_match_names
    )

    mentioned, rank, vis, parts = analyze_answer(
        text=answer_text,
        brand=brand,
        competitors=effective_competitors,
        sentiment=sentiment,
        brand_match_names=effective_match_names,
    )
    # `mentioned` is the deterministic regex ground truth (scoring.py); the
    # sentiment call's own quote can disagree with it — it was told to return
    # '' when the brand isn't mentioned, but doesn't always comply, and can
    # hand back a quote about a *competitor* instead. Trust the deterministic
    # check: no mention means no quote, full stop, regardless of what the
    # model returned.
    if mentioned:
        if not quote:
            quote = answer_text[:180]
    else:
        quote = ""

    return _PromptAnalysis(
        prompt=prompt,
        mentioned=mentioned,
        sentiment=sentiment,
        rank=rank,
        vis=vis,
        quote=quote,
        parts=parts,
    )


def _classify_opportunity(mentioned: bool, vis: int) -> str:
    # Deterministic read on competitive position, off the same vis score
    # scoring.py already computes — no new inputs, no extra call. Not
    # mentioned at all always wins out over the vis-based tiers below, since
    # a vis of 0 can also mean "mentioned but ranked/sentimented terribly"
    # (a different, less severe situation than never coming up at all).
    if not mentioned:
        return "Critical Gap"
    if vis >= 80:
        return "Defend"
    if vis >= 55:
        return "Grow"
    return "High"


def _aggregate(analyses: list[_PromptAnalysis]) -> PersonaResult:
    mentioned = any(a.mentioned for a in analyses)
    ranked = [a for a in analyses if a.rank is not None]
    representative = min(ranked, key=lambda a: (a.rank, -a.vis)) if ranked else max(
        analyses, key=lambda a: a.vis
    )

    avg_vis = round(sum(a.vis for a in analyses) / len(analyses))
    best_rank = min((a.rank for a in ranked), default=None)
    # Only prompts that named the brand have a sentiment toward it; the rest
    # are forced Neutral and would outvote a real Positive/Negative.
    sentiment_counts = Counter(a.sentiment for a in analyses if a.mentioned)
    top_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else Sentiment.neutral

    return PersonaResult(
        prompt=representative.prompt,
        mentioned=mentioned,
        sentiment=top_sentiment,
        vis=avg_vis,
        rank=best_rank,
        quote=representative.quote,
        parts=representative.parts,
        opportunity=_classify_opportunity(mentioned, avg_vis),
        exchanges=[
            PersonaExchange(
                prompt=a.prompt,
                mentioned=a.mentioned,
                sentiment=a.sentiment,
                rank=a.rank,
                vis=a.vis,
                quote=a.quote,
                parts=a.parts,
            )
            for a in analyses
        ],
    )


async def run_persona(
    *,
    persona: PersonaIn,
    prompts: list[str],
    brand: str,
    brand_match_names: list[str],
    competitors: list[Competitor],
    buyer_context: str | None,
    market: str | None,
    api_key: str,
) -> PersonaResult:
    settings = get_settings()

    tasks = [
        _analyze_one_prompt(
            prompt=p,
            brand=brand,
            brand_match_names=brand_match_names,
            competitors=competitors,
            buyer_context=buyer_context,
            market=market,
            api_key=api_key,
            answer_model=settings.anthropic_model,
            fast_model=settings.anthropic_fast_model,
        )
        for p in prompts
    ]
    # One failed prompt (after its own retries) used to fail the whole persona
    # and throw away its successful answers, which the client had paid for.
    # Keep what succeeded; fail only when nothing did.
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    analyses = [o for o in outcomes if isinstance(o, _PromptAnalysis)]
    errors = [o for o in outcomes if isinstance(o, BaseException)]
    for error in errors:
        if not isinstance(error, Exception):
            raise error  # cancellation and the like are not prompt failures
    if not analyses:
        raise errors[0]
    for error in errors:
        logger.warning("persona %s: one prompt failed, keeping the rest: %r", persona.id, error)
    result = _aggregate(analyses)
    result.failed_prompts = len(errors)
    return result
