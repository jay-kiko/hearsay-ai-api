"""§04 Per-persona pipeline.

For every prompt belonging to a persona: one Claude call for the buyer-style
answer, one structured Claude call classifying sentiment toward the brand and
pulling a quote, then deterministic parsing/scoring (app.services.scoring).
A persona can carry several prompts (see §03) — those per-prompt analyses are
run concurrently and folded into the single PersonaResult the frontend
expects, picking the best-ranked prompt as the representative response.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass

from app.config import get_settings
from app.models import PersonaIn, PersonaResult, Sentiment
from app.services.anthropic_client import call_structured, call_text
from app.services.scoring import analyze_answer

_ANSWER_SYSTEM = (
    "You are a knowledgeable industry analyst helping a software buyer research "
    "options. Answer with specific, real product/vendor names wherever relevant — "
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
    },
    "required": ["sentiment", "quote"],
}


def _sentiment_system(brand: str) -> str:
    return (
        f"Classify the sentiment of the following AI-generated answer specifically "
        f"toward the brand '{brand}' — not the general tone of the answer. If '{brand}' "
        f"is not mentioned at all, sentiment must be 'Neutral' and quote must be ''."
    )


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
    *, prompt: str, brand: str, competitors: list[str], api_key: str, answer_model: str, fast_model: str
) -> _PromptAnalysis:
    answer_task = call_text(api_key=api_key, model=answer_model, system=_ANSWER_SYSTEM, user=prompt)
    answer_text = await answer_task

    # Sentiment classification is mechanical extraction, not the signal being
    # measured — runs on the cheaper fast_model, not the answer model.
    sentiment_result = await call_structured(
        api_key=api_key,
        model=fast_model,
        system=_sentiment_system(brand),
        user=answer_text or "(empty response)",
        tool_name=_SENTIMENT_TOOL_NAME,
        tool_description="Classify sentiment toward the brand and extract a supporting quote.",
        input_schema=_SENTIMENT_INPUT_SCHEMA,
        max_tokens=512,
    )
    sentiment = Sentiment(sentiment_result.get("sentiment", "Neutral"))
    quote = sentiment_result.get("quote", "") or ""

    mentioned, rank, vis, parts = analyze_answer(
        text=answer_text, brand=brand, competitors=competitors, sentiment=sentiment
    )
    if not quote and mentioned:
        quote = answer_text[:180]

    return _PromptAnalysis(
        prompt=prompt,
        mentioned=mentioned,
        sentiment=sentiment,
        rank=rank,
        vis=vis,
        quote=quote,
        parts=parts,
    )


def _aggregate(analyses: list[_PromptAnalysis]) -> PersonaResult:
    mentioned = any(a.mentioned for a in analyses)
    ranked = [a for a in analyses if a.rank is not None]
    representative = min(ranked, key=lambda a: (a.rank, -a.vis)) if ranked else max(
        analyses, key=lambda a: a.vis
    )

    avg_vis = round(sum(a.vis for a in analyses) / len(analyses))
    best_rank = min((a.rank for a in ranked), default=None)
    sentiment_counts = Counter(a.sentiment for a in analyses)
    top_sentiment = sentiment_counts.most_common(1)[0][0]

    return PersonaResult(
        prompt=representative.prompt,
        mentioned=mentioned,
        sentiment=top_sentiment,
        vis=avg_vis,
        rank=best_rank,
        quote=representative.quote,
        parts=representative.parts,
    )


async def run_persona(
    *,
    persona: PersonaIn,
    prompts: list[str],
    brand: str,
    competitors: list[str],
    api_key: str,
) -> PersonaResult:
    settings = get_settings()

    tasks = [
        _analyze_one_prompt(
            prompt=p,
            brand=brand,
            competitors=competitors,
            api_key=api_key,
            answer_model=settings.anthropic_model,
            fast_model=settings.anthropic_fast_model,
        )
        for p in prompts
    ]
    analyses = await asyncio.gather(*tasks)
    return _aggregate(list(analyses))
