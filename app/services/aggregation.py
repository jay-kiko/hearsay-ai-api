"""§06 Aggregation.

One pass over every persona result — no model call, everything needed is
already sitting in the per-persona results. Produces the Results screen's
overview cards (mention rate, average sentiment, top competitor) and the
share-of-voice product list, by running the same mention-detection logic
across every persona's response rather than just the brand's own.
"""
from __future__ import annotations

from collections import Counter

from app.models import Overview, PersonaResult, Product, Sentiment


def _mentions_name(result: PersonaResult, name: str) -> bool:
    name_lower = name.strip().lower()
    return any(part.text.strip().lower() == name_lower for part in result.parts)


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


def build_products(results: dict[str, PersonaResult], brand: str, competitors: list[str]) -> list[Product]:
    total = len(results) or 1
    names = [brand, *competitors]
    counts = {name: sum(1 for r in results.values() if _mentions_name(r, name)) for name in names}

    products = [
        Product(name=name, count=count, share=round(count / total, 3), is_brand=(name == brand))
        for name, count in counts.items()
    ]
    products.sort(key=lambda p: (-p.count, p.name))
    return products


def top_competitor(products: list[Product], brand: str) -> str | None:
    competitor_products = [p for p in products if p.name != brand and p.count > 0]
    if not competitor_products:
        return None
    return max(competitor_products, key=lambda p: p.count).name
