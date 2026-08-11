"""Deterministic parsing and scoring — no model calls.

Given the free-text answer from the "buyer" Claude call, find where the brand
and each competitor are mentioned, split the text into highlightable spans,
derive a rank from the position of the first mention, and combine
mention/rank/sentiment into a single 0-100 visibility score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import ResponsePart, Sentiment

_RANK_STEP = 20  # each rank position back costs 20 points, floor 0
_SENTIMENT_MULTIPLIER = {
    Sentiment.positive: 1.0,
    Sentiment.neutral: 0.7,
    Sentiment.negative: 0.35,
}


def _pattern_for(name: str) -> re.Pattern[str] | None:
    name = name.strip()
    if not name:
        return None
    return re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)


@dataclass
class Mention:
    start: int
    end: int
    kind: str  # "brand" | "competitor"
    name: str


def find_mentions(text: str, brand: str, competitors: list[str]) -> list[Mention]:
    candidates: list[Mention] = []

    brand_pattern = _pattern_for(brand)
    if brand_pattern:
        for m in brand_pattern.finditer(text):
            candidates.append(Mention(m.start(), m.end(), "brand", brand))

    for competitor in competitors:
        pattern = _pattern_for(competitor)
        if not pattern:
            continue
        for m in pattern.finditer(text):
            candidates.append(Mention(m.start(), m.end(), "competitor", competitor))

    candidates.sort(key=lambda m: m.start)

    # Drop mentions that overlap an earlier (longer/first) match.
    merged: list[Mention] = []
    last_end = -1
    for mention in candidates:
        if mention.start < last_end:
            continue
        merged.append(mention)
        last_end = mention.end
    return merged


def build_parts(text: str, mentions: list[Mention]) -> list[ResponsePart]:
    if not mentions:
        return [ResponsePart(text=text, kind="normal")] if text else []

    parts: list[ResponsePart] = []
    cursor = 0
    for mention in mentions:
        if mention.start > cursor:
            parts.append(ResponsePart(text=text[cursor:mention.start], kind="normal"))
        parts.append(ResponsePart(text=text[mention.start:mention.end], kind=mention.kind))
        cursor = mention.end
    if cursor < len(text):
        parts.append(ResponsePart(text=text[cursor:], kind="normal"))
    return parts


def compute_rank(mentions: list[Mention], brand: str, competitors: list[str]) -> int | None:
    """1-indexed position of the brand among the first mention of each distinct name."""
    seen: list[str] = []
    for mention in mentions:
        key = mention.name.lower()
        if key not in seen:
            seen.append(key)
    if brand.lower() not in seen:
        return None
    return seen.index(brand.lower()) + 1


def compute_visibility_score(mentioned: bool, rank: int | None, sentiment: Sentiment) -> int:
    if not mentioned or rank is None:
        return 0
    rank_score = max(0, 100 - (rank - 1) * _RANK_STEP)
    score = rank_score * _SENTIMENT_MULTIPLIER.get(sentiment, 0.7)
    return max(0, min(100, round(score)))


def analyze_answer(
    *,
    text: str,
    brand: str,
    competitors: list[str],
    sentiment: Sentiment,
) -> tuple[bool, int | None, int, list[ResponsePart]]:
    mentions = find_mentions(text, brand, competitors)
    mentioned = any(m.kind == "brand" for m in mentions)
    rank = compute_rank(mentions, brand, competitors)
    vis = compute_visibility_score(mentioned, rank, sentiment)
    parts = build_parts(text, mentions)
    return mentioned, rank, vis, parts
