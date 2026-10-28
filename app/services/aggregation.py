"""§06 Aggregation.

One pass over every persona result — no model call, everything needed is
already sitting in the per-persona results. Produces the Results screen's
overview cards (mention rate, average sentiment, top competitor) and the
share-of-voice product list, by running the same mention-detection logic
across every persona's response rather than just the brand's own.
"""
from __future__ import annotations

from collections import Counter

from app.models import Competitor, Overview, PersonaExchange, PersonaResult, Product, ScoreComponent, Sentiment, Sources
from app.services.scoring import SENTIMENT_MULTIPLIER, rank_score



def _all_exchanges(results: dict[str, PersonaResult]) -> list[PersonaExchange]:
    return [ex for r in results.values() for ex in r.exchanges]


def _exchange_mentions(exchange: PersonaExchange, canonical_name: str) -> bool:
    # ResponsePart.name is set once, in scoring.py, from whichever alias
    # matched. Re-matching text here against a local alias list misses aliases
    # recognised per answer (confirmed live: 100% mention rate beside 0% Share
    # of Voice).
    return any(part.name == canonical_name for part in exchange.parts)


def _prompt_mention_counts(results: dict[str, PersonaResult]) -> tuple[int, int]:
    """Mentioned/total across every prompt exchange. Not one OR'd boolean per
    persona: that lets a persona with more prompts count as "mentioned" just
    from having more tries. Every mention-based number in the API counts at
    this prompt level so they agree with each other (invariant 8)."""
    all_exchanges = _all_exchanges(results)
    mentioned = sum(1 for ex in all_exchanges if ex.mentioned)
    return mentioned, len(all_exchanges)


def _failed_prompts(results: dict[str, PersonaResult]) -> int:
    return sum(r.failed_prompts for r in results.values())


def build_overview(results: dict[str, PersonaResult], visibility_score: int) -> Overview:
    mentioned, total = _prompt_mention_counts(results)
    mention_rate = (mentioned / total) if total else 0.0

    # Same basis as the breakdown's Sentiment: only prompts that mentioned the
    # brand. Persona-level sentiment also counts non-mentions (forced Neutral),
    # which put "Neutral" next to "4 positive · 0 neutral · 0 negative".
    mentioned_exchanges = [ex for r in results.values() for ex in (r.exchanges or []) if ex.mentioned]
    sentiment_counts = Counter(ex.sentiment for ex in mentioned_exchanges)
    avg_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else Sentiment.neutral

    return Overview(
        visibility_score=visibility_score,
        mentioned=mentioned,
        total=total,
        mention_rate=round(mention_rate, 3),
        avg_sentiment=avg_sentiment,
        top_competitor=None,  # filled in by build_products, which knows the competitor set
        failed_prompts=_failed_prompts(results),
    )


def build_products(results: dict[str, PersonaResult], brand: str, competitors: list[Competitor]) -> list[Product]:
    # (display name, is_brand) — no aliases needed here; _exchange_mentions
    # reads the canonical name scoring.py already attached to each part.
    entries: list[tuple[str, bool]] = [(brand, True)] + [(c.name, False) for c in competitors]

    # Prompt-level, like Overview, so the two counts agree.
    all_exchanges = _all_exchanges(results)
    counts = [
        (name, sum(1 for ex in all_exchanges if _exchange_mentions(ex, name)), is_brand)
        for name, is_brand in entries
    ]
    # Each product's slice of *all* product mentions, so shares sum to 100%
    # (dividing by persona count summed to 126% when answers named several).
    total_mentions = sum(count for _, count, _ in counts) or 1

    products = [
        Product(name=name, count=count, share=round(count / total_mentions, 3), is_brand=is_brand)
        for name, count, is_brand in counts
    ]
    products.sort(key=lambda p: (-p.count, p.name))
    return products


def build_score_breakdown(
    results: dict[str, PersonaResult], products: list[Product], brand: str, sources: Sources
) -> list[ScoreComponent]:
    """Six deterministic sub-scores behind the single visibilityScore — no
    extra AI call, everything here is math over data the pipeline/grounding
    already produced."""
    all_exchanges = _all_exchanges(results)
    mentioned_exchanges = [ex for ex in all_exchanges if ex.mentioned]
    mentioned_prompts = len(mentioned_exchanges)
    total_prompts = len(all_exchanges)

    presence = round(mentioned_prompts / total_prompts * 100) if total_prompts else 0

    brand_product = next((p for p in products if p.is_brand), None)
    sov = round((brand_product.share if brand_product else 0.0) * 100)

    # Recommendation Strength / Ranking / Sentiment average over the same
    # mentioned exchanges Presence counts, so the card shares one denominator.
    rec_strength = round(sum(ex.vis for ex in mentioned_exchanges) / mentioned_prompts) if mentioned_prompts else 0

    ranked = [ex.rank for ex in mentioned_exchanges if ex.rank is not None]
    avg_rank = round(sum(ranked) / len(ranked)) if ranked else None
    ranking_score = rank_score(avg_rank) if avg_rank else 0

    sentiment_counts = Counter(ex.sentiment for ex in mentioned_exchanges)
    pos, neu, neg = (
        sentiment_counts.get(Sentiment.positive, 0),
        sentiment_counts.get(Sentiment.neutral, 0),
        sentiment_counts.get(Sentiment.negative, 0),
    )
    sentiment_score = (
        round(sum(SENTIMENT_MULTIPLIER[s] * c for s, c in sentiment_counts.items()) / mentioned_prompts * 100)
        if mentioned_prompts
        else 0
    )

    # The same measured `mentions` behind each source's Visible badge, so the
    # score and the badges cannot disagree.
    total_sources = len(sources.source_intel)
    named_sources = sum(1 for s in sources.source_intel if s.mentions > 0)
    source_authority = round(named_sources / total_sources * 100) if total_sources else 0

    return [
        ScoreComponent(
            name="Presence",
            score=presence,
            # Failed prompts are out of the denominator, so say so: "4 of 10"
            # must not read as the whole run when 12 were asked.
            note=f"Mentioned in {mentioned_prompts} of {total_prompts} persona queries"
            + (f" ({failed} more couldn't be answered this run)" if (failed := _failed_prompts(results)) else ""),
        ),
        ScoreComponent(
            name="Share of Voice", score=sov, note=f"{sov}% of all product mentions"
        ),
        ScoreComponent(
            name="Recommendation Strength",
            score=rec_strength,
            note=(
                f"Average visibility score of {rec_strength} across {mentioned_prompts} mentioning queries"
                if mentioned_prompts
                else "Never mentioned, so no recommendation strength to measure"
            ),
        ),
        ScoreComponent(
            name="Ranking",
            score=ranking_score,
            note=f"Average position #{avg_rank} in listed options" if avg_rank else "Not ranked in any answer",
        ),
        ScoreComponent(
            name="Sentiment", score=sentiment_score, note=f"{pos} positive · {neu} neutral · {neg} negative"
        ),
        ScoreComponent(
            name="Source Authority",
            score=source_authority,
            note=(
                f"Named in {named_sources} of {total_sources} independent sources"
                + (
                    " (judged from page text where readable, else titles and snippets, so a zero is low-confidence)"
                    if named_sources == 0
                    else ""
                )
                if total_sources
                else "No independent sources found, so this is left out of the visibility score"
            ),
        ),
    ]


# Headline score: a weighted composite of the breakdown, so a brand drowned
# out by competitors can't post 90+ off rank and sentiment alone.
# Recommendation Strength is left out: it is rank x sentiment, which Ranking
# and Sentiment already contribute, so including it would count them twice.
_COMPOSITE_WEIGHTS = {
    "Presence": 0.30,
    "Share of Voice": 0.30,
    "Ranking": 0.20,
    "Sentiment": 0.10,
    "Source Authority": 0.10,
}


def compute_composite_score(score_breakdown: list[ScoreComponent], *, has_sources: bool = True) -> int:
    """With no sources there is nothing to measure Source Authority on, so it is
    left out and the other weights re-normalised — scoring it 0 would cut the
    headline by up to 10 points over a search that came back empty."""
    weights = {k: w for k, w in _COMPOSITE_WEIGHTS.items() if has_sources or k != "Source Authority"}
    total = sum(weights.values())
    weighted = sum(weights.get(c.name, 0.0) * c.score for c in score_breakdown)
    return round(weighted / total) if total else 0


def top_competitor(products: list[Product], brand: str) -> str | None:
    competitor_products = [p for p in products if p.name != brand and p.count > 0]
    if not competitor_products:
        return None
    return max(competitor_products, key=lambda p: p.count).name
