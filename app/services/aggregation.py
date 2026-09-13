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

# Mirrors scoring._SENTIMENT_MULTIPLIER — duplicated rather than imported
# since that name is private to scoring.py and this is a small, stable
# constant; keep the two in sync if the weighting ever changes.
_SENTIMENT_WEIGHT = {Sentiment.positive: 1.0, Sentiment.neutral: 0.7, Sentiment.negative: 0.35}


def _all_exchanges(results: dict[str, PersonaResult]) -> list[PersonaExchange]:
    return [ex for r in results.values() for ex in r.exchanges]


def _exchange_mentions(exchange: PersonaExchange, canonical_name: str) -> bool:
    # Trust ResponsePart.name (set once, in scoring.py, from whatever alias
    # actually matched — static list or a per-answer LLM-recognized one) as
    # the single source of truth. This used to re-derive "was X mentioned"
    # by re-matching part.text against a locally-known alias list, which
    # went stale the moment mention detection started recognizing aliases
    # this second, independent check didn't know about (confirmed live: a
    # run where every mention came through under an LLM-recognized product
    # name showed 100% mention rate but 0% Share of Voice, because this
    # check's alias list only ever had the bare brand name).
    return any(part.name == canonical_name for part in exchange.parts)


def _prompt_mention_counts(results: dict[str, PersonaResult]) -> tuple[int, int]:
    """Mentioned/total across every individual prompt exchange, not one
    OR-aggregated boolean per persona (PersonaResult.mentioned) — OR-of-
    booleans gives a persona with more prompts a higher chance of counting
    as "mentioned" purely from having more trials, independent of actual
    visibility. Confirmed as a real, growing bias once multi-category prompt
    generation made per-persona prompt counts vary instead of being uniform.

    Every "how much was this mentioned" number in the API (Overview, Presence,
    Products/Share of Voice, Recommendation Strength, Ranking, Sentiment)
    counts at this same prompt level now — mixing prompt-level and
    persona-level counts across an otherwise-connected set of numbers is
    exactly what produced a visibly inconsistent Score Breakdown card
    (Presence said "4 of 24," Sentiment/Ranking/Recommendation Strength were
    silently still counting off 2 — the number of *personas* that ever
    mentioned the brand, not the 4 individual prompts that did)."""
    all_exchanges = _all_exchanges(results)
    mentioned = sum(1 for ex in all_exchanges if ex.mentioned)
    return mentioned, len(all_exchanges)


def build_overview(results: dict[str, PersonaResult], visibility_score: int) -> Overview:
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
    # (display name, is_brand) — no aliases needed here; _exchange_mentions
    # reads the canonical name scoring.py already attached to each part.
    entries: list[tuple[str, bool]] = [(brand, True)] + [(c.name, False) for c in competitors]

    # Counts every individual prompt exchange, not one OR'd boolean per
    # persona (confirmed live: with this counting a persona-count instead,
    # "Products mentioned" showed 2 for a brand while Overview's prompt-level
    # mention count showed 4 for the exact same run — two numbers that
    # should agree, both describing "how much was this brand mentioned,"
    # silently using different units).
    all_exchanges = _all_exchanges(results)
    counts = [
        (name, sum(1 for ex in all_exchanges if _exchange_mentions(ex, name)), is_brand)
        for name, is_brand in entries
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
    all_exchanges = _all_exchanges(results)
    mentioned_exchanges = [ex for ex in all_exchanges if ex.mentioned]
    mentioned_prompts = len(mentioned_exchanges)
    total_prompts = len(all_exchanges)

    presence = round(mentioned_prompts / total_prompts * 100) if total_prompts else 0

    brand_product = next((p for p in products if p.is_brand), None)
    sov = round((brand_product.share if brand_product else 0.0) * 100)

    # Recommendation Strength / Ranking / Sentiment all average over the same
    # mentioned_exchanges (prompt-level) rather than one representative
    # PersonaResult per persona — every "how much/how well was this brand
    # mentioned" number in this breakdown now shares the same denominator as
    # Presence, instead of Presence counting prompts while these counted
    # personas (confirmed live: Presence said "4 of 24," these were silently
    # still averaging over just 2 — the personas that ever mentioned the
    # brand at all, not the 4 prompts that actually did).
    rec_strength = round(sum(ex.vis for ex in mentioned_exchanges) / mentioned_prompts) if mentioned_prompts else 0

    ranked = [ex.rank for ex in mentioned_exchanges if ex.rank is not None]
    avg_rank = round(sum(ranked) / len(ranked)) if ranked else None
    ranking_score = max(0, 100 - (avg_rank - 1) * 20) if avg_rank else 0

    sentiment_counts = Counter(ex.sentiment for ex in mentioned_exchanges)
    pos, neu, neg = (
        sentiment_counts.get(Sentiment.positive, 0),
        sentiment_counts.get(Sentiment.neutral, 0),
        sentiment_counts.get(Sentiment.negative, 0),
    )
    sentiment_score = (
        round(sum(_SENTIMENT_WEIGHT[s] * c for s, c in sentiment_counts.items()) / mentioned_prompts * 100)
        if mentioned_prompts
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
                f"Named in {named_citations} of {total_citations} researched sources"
                if total_citations
                else "No sources found to be named in"
            ),
        ),
    ]


# Headline score = a real weighted composite of the breakdown above, not the
# separate rank/sentiment-only calculation it used to be — Share of Voice and
# Source Authority previously had zero influence on this number despite the
# breakdown's docstring implying they were "sub-scores behind" it (a brand
# drowned out by competitors in most answers could still post a 90+ purely
# off rank+sentiment on the answers it did appear in). Recommendation
# Strength is deliberately excluded here: it's rank_score × sentiment
# averaged over mentioned exchanges — the same signal Ranking + Sentiment
# already contribute below — so including it too would triple-count it.
_COMPOSITE_WEIGHTS = {
    "Presence": 0.30,
    "Share of Voice": 0.30,
    "Ranking": 0.20,
    "Sentiment": 0.10,
    "Source Authority": 0.10,
}


def compute_composite_score(score_breakdown: list[ScoreComponent]) -> int:
    weighted = sum(_COMPOSITE_WEIGHTS.get(c.name, 0.0) * c.score for c in score_breakdown)
    return round(weighted)


def top_competitor(products: list[Product], brand: str) -> str | None:
    competitor_products = [p for p in products if p.name != brand and p.count > 0]
    if not competitor_products:
        return None
    return max(competitor_products, key=lambda p: p.count).name
