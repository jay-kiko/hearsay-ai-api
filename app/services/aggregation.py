"""§06 Aggregation.

One pass over every persona result — no model call, everything needed is
already sitting in the per-persona results. Produces the Results screen's
overview cards (mention rate, average sentiment, top competitor) and the
share-of-voice product list, by running the same mention-detection logic
across every persona's response rather than just the brand's own.
"""
from __future__ import annotations

from collections import Counter

from app.models import Competitor, Overview, PersonaResult, Product, Sentiment


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


def build_overview(results: dict[str, PersonaResult]) -> Overview:
    total = len(results)
    mentioned = sum(1 for r in results.values() if r.mentioned)
    visibility_score = round(sum(r.vis for r in results.values()) / total) if total else 0
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
    total = len(results) or 1
    # (display name, aliases to check, is_brand) — brand has no aliases of
    # its own in current scope, just its literal name.
    entries: list[tuple[str, list[str], bool]] = [(brand, [brand], True)]
    entries += [(c.name, c.match_names or [c.name], False) for c in competitors]

    products = [
        Product(
            name=name,
            count=(count := sum(1 for r in results.values() if _mentions_any(r, match_names))),
            share=round(count / total, 3),
            is_brand=is_brand,
        )
        for name, match_names, is_brand in entries
    ]
    products.sort(key=lambda p: (-p.count, p.name))
    return products


def top_competitor(products: list[Product], brand: str) -> str | None:
    competitor_products = [p for p in products if p.name != brand and p.count > 0]
    if not competitor_products:
        return None
    return max(competitor_products, key=lambda p: p.count).name
