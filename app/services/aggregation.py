"""§06 Aggregation.

One pass over every persona result — no model call, everything needed is
already sitting in the per-persona results. Produces the Results screen's
overview cards (mention rate, average sentiment, top competitor) and the
share-of-voice product list, by running the same mention-detection logic
across every persona's response rather than just the brand's own.
"""
from __future__ import annotations

from collections import Counter

from app.models import Competitor, Overview, PersonaResult, Product, ScoreComponent, Sentiment, Sources

# Mirrors scoring._SENTIMENT_MULTIPLIER — duplicated rather than imported
# since that name is private to scoring.py and this is a small, stable
# constant; keep the two in sync if the weighting ever changes.
_SENTIMENT_WEIGHT = {Sentiment.positive: 1.0, Sentiment.neutral: 0.7, Sentiment.negative: 0.35}


def _mentions_any(result: PersonaResult, match_names: list[str]) -> bool:
    # A highlighted part's text is whatever alias literally matched (e.g.
    # "Tommy Hilfiger"), not necessarily the competitor's canonical display
    # name (e.g. "PVH (parent of Tommy Hilfiger and Calvin Klein)") — check
    # against every alias, not just the one name, or this silently misses
    # every sub-brand mention the same way the old exact-name check did.
    #
    # Also check every exchange, not just result.parts (the single
    # representative prompt _aggregate() picked for display) — a competitor
    # named only in one of a persona's other prompts is sitting right there
    # in result.exchanges[i].parts, but invisible to Share of Voice if only
    # the representative prompt's parts get inspected. exchanges already
    # includes the representative prompt's own analysis too, so this is a
    # strict superset of the old check, never a narrower one.
    candidates = {n.strip().lower() for n in match_names if n.strip()}
    all_parts = (part for exchange in result.exchanges for part in exchange.parts)
    return any(part.text.strip().lower() in candidates for part in all_parts)


def _prompt_mention_counts(results: dict[str, PersonaResult]) -> tuple[int, int]:
    """Mentioned/total across every individual prompt exchange, not one
    OR-aggregated boolean per persona (PersonaResult.mentioned) — OR-of-
    booleans gives a persona with more prompts a higher chance of counting
    as "mentioned" purely from having more trials, independent of actual
    visibility. Confirmed as a real, growing bias once multi-category prompt
    generation made per-persona prompt counts vary instead of being uniform."""
    all_exchanges = [ex for r in results.values() for ex in r.exchanges]
    mentioned = sum(1 for ex in all_exchanges if ex.mentioned)
    return mentioned, len(all_exchanges)


def build_overview(results: dict[str, PersonaResult]) -> Overview:
    persona_total = len(results)
    # Deliberately double-averaged (per-persona, then across personas) rather
    # than a flat average over every raw prompt — a persona with more
    # prompts shouldn't dominate the overview average any more than one with
    # fewer. This is the opposite bias from mentioned/total below, and both
    # are handled correctly for what each one actually measures.
    visibility_score = round(sum(r.vis for r in results.values()) / persona_total) if persona_total else 0

    mentioned, total = _prompt_mention_counts(results)
    mention_rate = (mentioned / total) if total else 0.0

    sentiment_counts = Counter(r.sentiment for r in results.values())
    avg_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else Sentiment.neutral

    return Overview(
        visibility_score=visibility_score,
        mentioned=mentioned,
        total=total,
        mention_rate=round(mention_rate, 3),
        avg_sentiment=avg_sentiment,
        top_competitor=None,  # filled in by build_products, which knows the competitor set
    )


def build_products(results: dict[str, PersonaResult], brand: str, competitors: list[Competitor]) -> list[Product]:
    # (display name, aliases to check, is_brand) — brand has no aliases of
    # its own in current scope, just its literal name.
    entries: list[tuple[str, list[str], bool]] = [(brand, [brand], True)]
    entries += [(c.name, c.match_names or [c.name], False) for c in competitors]

    counts = [
        (name, sum(1 for r in results.values() if _mentions_any(r, match_names)), is_brand)
        for name, match_names, is_brand in entries
    ]
    # Share of Voice is each product's slice of *all* product mentions, not
    # its mention rate across personas — dividing by persona count instead
    # let every product's share be counted independently against the same
    # denominator, so they summed to well over 100% whenever more than one
    # product got mentioned per persona (confirmed live: 50/38/25/13%,
    # summing to 126%, for counts of 4/3/2/1 out of 8 personas each).
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
    total = len(results) or 1
    mentioned_results = [r for r in results.values() if r.mentioned]
    mentioned_n = len(mentioned_results)

    mentioned_prompts, total_prompts = _prompt_mention_counts(results)
    presence = round(mentioned_prompts / total_prompts * 100) if total_prompts else 0

    brand_product = next((p for p in products if p.is_brand), None)
    sov = round((brand_product.share if brand_product else 0.0) * 100)

    rec_strength = round(sum(r.vis for r in mentioned_results) / mentioned_n) if mentioned_n else 0

    ranked = [r.rank for r in mentioned_results if r.rank is not None]
    avg_rank = round(sum(ranked) / len(ranked)) if ranked else None
    ranking_score = max(0, 100 - (avg_rank - 1) * 20) if avg_rank else 0

    sentiment_counts = Counter(r.sentiment for r in mentioned_results)
    pos, neu, neg = (
        sentiment_counts.get(Sentiment.positive, 0),
        sentiment_counts.get(Sentiment.neutral, 0),
        sentiment_counts.get(Sentiment.negative, 0),
    )
    sentiment_score = (
        round(sum(_SENTIMENT_WEIGHT[s] * c for s, c in sentiment_counts.items()) / mentioned_n * 100)
        if mentioned_n
        else 0
    )

    brand_lower = brand.lower()
    total_citations = len(sources.citations)
    named_citations = sum(1 for c in sources.citations if brand_lower in c.title.lower())
    source_authority = round(named_citations / total_citations * 100) if total_citations else 0

    return [
        ScoreComponent(
            name="Presence",
            score=presence,
            note=f"Mentioned in {mentioned_prompts} of {total_prompts} persona queries",
        ),
        ScoreComponent(
            name="Share of Voice", score=sov, note=f"{sov}% of all product mentions"
        ),
        ScoreComponent(
            name="Recommendation Strength",
            score=rec_strength,
            note=(
                f"Average visibility score of {rec_strength} across {mentioned_n} mentioning queries"
                if mentioned_n
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
                f"Named in {named_citations} of {total_citations} researched sources"
                if total_citations
                else "No sources found to be named in"
            ),
        ),
    ]


def top_competitor(products: list[Product], brand: str) -> str | None:
    competitor_products = [p for p in products if p.name != brand and p.count > 0]
    if not competitor_products:
        return None
    return max(competitor_products, key=lambda p: p.count).name
