"""Deterministic parsing and scoring — no model calls.

Given the free-text answer from the "buyer" Claude call, find where the brand
and each competitor are mentioned, split the text into highlightable spans,
derive a rank from the position of the first mention, and combine
mention/rank/sentiment into a single 0-100 visibility score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import Competitor, ResponsePart, Sentiment

_RANK_STEP = 20  # each rank position back costs 20 points, floor 0
# Also the Sentiment sub-score's weights (aggregation.py): one definition.
SENTIMENT_MULTIPLIER = {
    Sentiment.positive: 1.0,
    Sentiment.neutral: 0.7,
    Sentiment.negative: 0.35,
}

# Stripped before matching so a formal/legal name still matches a plain
# mention — an AI answer says "Tapestry" or "PVH", essentially never
# "Tapestry, Inc." or "PVH Corp." verbatim. Matching the exact string as
# given (the old behavior) missed nearly every real-world mention of a
# multi-brand or corporate competitor.
_CORPORATE_SUFFIX_RE = re.compile(
    r"[,]?\s+(?:Inc|Incorporated|Corp|Corporation|Co|Company|Ltd|Limited|LLC|LLP|"
    r"Group|Holdings?|plc|PLC|GmbH|AG|S\.?A\.?|N\.?V\.?)\.?\s*$",
    re.IGNORECASE,
)


def _core_name(name: str) -> str:
    stripped = _CORPORATE_SUFFIX_RE.sub("", name).strip()
    return stripped or name


def _pattern_for(name: str) -> re.Pattern[str] | None:
    name = _core_name(name.strip())
    if not name:
        return None
    # Escape each word separately and rejoin with \s+ instead of escaping the
    # whole string in one go — incidental spacing differences (double space,
    # a line break between words) still match, only the literal words
    # themselves have to line up exactly.
    words = [re.escape(w) for w in name.split()]
    if not words:
        return None
    return re.compile(rf"\b{r'\s+'.join(words)}\b", re.IGNORECASE)


@dataclass
class Mention:
    start: int
    end: int
    kind: str  # "brand" | "competitor"
    name: str


@dataclass(frozen=True)
class BrandContext:
    """Names that count as the brand only in context. A parent or short name
    ("Disney" for Disney Cruise Line) also means unrelated things (the parks,
    Disney+), so it counts only in a sentence that also has a category word
    ("cruise", "ship"). Confirmed live: a Yahoo cruise guide said "Disney is
    an obvious choice" and "a Disney ship" but never "Disney Cruise Line",
    and read 0 mentions; its header also linked "Disney reveals Monstropolis
    coaster", which must not count."""

    names: tuple[str, ...] = ()
    category_terms: tuple[str, ...] = ()

    @classmethod
    def of(cls, names: list[str] | None, category_terms: list[str] | None) -> "BrandContext | None":
        names_t = tuple(n for n in (names or []) if n.strip())
        terms_t = tuple(t.strip().lower() for t in (category_terms or []) if t.strip())
        # Without category words there's no context to check, so a context
        # name never counts on its own.
        return cls(names_t, terms_t) if names_t and terms_t else None

    def term_pattern(self) -> re.Pattern[str]:
        # Prefix match so "cruise" also covers "cruises" and "cruising".
        return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in self.category_terms) + r")\w*", re.IGNORECASE)


_SENTENCE_END = re.compile(r"[.!?\n]")


def sentence_around(text: str, start: int, end: int) -> str:
    """The sentence (or line) containing text[start:end]."""
    begin = max((m.end() for m in _SENTENCE_END.finditer(text, 0, start)), default=0)
    after = _SENTENCE_END.search(text, end)
    return text[begin : after.start() if after else len(text)]


def find_mentions(
    text: str,
    brand: str,
    competitors: list[Competitor],
    brand_match_names: list[str] | None = None,
    *,
    context: BrandContext | None = None,
) -> list[Mention]:
    candidates: list[Mention] = []

    # Try every brand alias (sub-brand, product line, short form) — same
    # rationale as competitor aliases below — but record matches under the
    # canonical brand name so rank/Share-of-Voice count it as one entity.
    seen_brand_spans: set[tuple[int, int]] = set()
    for alias in brand_match_names or [brand]:
        pattern = _pattern_for(alias)
        if not pattern:
            continue
        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            if span in seen_brand_spans:
                continue
            seen_brand_spans.add(span)
            candidates.append(Mention(m.start(), m.end(), "brand", brand))

    # Context names come after the full aliases, so where both match at the
    # same place ("Disney Wish" vs "Disney") the longer, unconditional one wins.
    if context:
        terms = context.term_pattern()
        for name in context.names:
            pattern = _pattern_for(name)
            if not pattern:
                continue
            for m in pattern.finditer(text):
                span = (m.start(), m.end())
                if span in seen_brand_spans or not terms.search(sentence_around(text, *span)):
                    continue
                seen_brand_spans.add(span)
                candidates.append(Mention(m.start(), m.end(), "brand", brand))

    for competitor in competitors:
        # A holding company is almost always mentioned by a sub-brand's name,
        # not its own (confirmed live: zero matches for "PVH" against an
        # answer that named "Tommy Hilfiger" and "Calvin Klein" explicitly)
        # — try every alias, but record the mention under the canonical
        # display name so rank/Share-of-Voice count it as one entity.
        seen_spans: set[tuple[int, int]] = set()
        for alias in competitor.match_names or [competitor.name]:
            pattern = _pattern_for(alias)
            if not pattern:
                continue
            for m in pattern.finditer(text):
                span = (m.start(), m.end())
                if span in seen_spans:
                    continue  # two aliases matching the identical span
                seen_spans.add(span)
                candidates.append(Mention(m.start(), m.end(), "competitor", competitor.name))

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


def name_matches(name: str, text: str) -> list[tuple[int, str]]:
    """(position, text as written) for every place the matcher would count
    `name` in `text` — the same rule find_mentions applies."""
    pattern = _pattern_for(name)
    return [(m.start(), m.group(0)) for m in pattern.finditer(text)] if pattern else []


def count_brand_mentions(
    text: str, brand: str, brand_match_names: list[str] | None = None, *, context: BrandContext | None = None
) -> int:
    """How many times `text` names the brand under any known alias. The one
    place anything outside this module should ask "does this text mention the
    brand" — source evidence (grounding.py) goes through here so it agrees
    with the per-prompt mention detection instead of re-deriving its own."""
    return sum(1 for m in find_mentions(text, brand, [], brand_match_names, context=context) if m.kind == "brand")


def build_parts(text: str, mentions: list[Mention]) -> list[ResponsePart]:
    if not mentions:
        return [ResponsePart(text=text, kind="normal")] if text else []

    parts: list[ResponsePart] = []
    cursor = 0
    for mention in mentions:
        if mention.start > cursor:
            parts.append(ResponsePart(text=text[cursor:mention.start], kind="normal"))
        parts.append(ResponsePart(text=text[mention.start:mention.end], kind=mention.kind, name=mention.name))
        cursor = mention.end
    if cursor < len(text):
        parts.append(ResponsePart(text=text[cursor:], kind="normal"))
    return parts


def compute_rank(mentions: list[Mention], brand: str) -> int | None:
    """1-indexed position of the brand among the first mention of each distinct name."""
    seen: list[str] = []
    for mention in mentions:
        key = mention.name.lower()
        if key not in seen:
            seen.append(key)
    if brand.lower() not in seen:
        return None
    return seen.index(brand.lower()) + 1


def rank_score(rank: int) -> int:
    """0-100 for a 1-indexed position: first is 100, each place back costs 20."""
    return max(0, 100 - (rank - 1) * _RANK_STEP)


def compute_visibility_score(mentioned: bool, rank: int | None, sentiment: Sentiment) -> int:
    if not mentioned or rank is None:
        return 0
    score = rank_score(rank) * SENTIMENT_MULTIPLIER.get(sentiment, 0.7)
    return max(0, min(100, round(score)))


def analyze_answer(
    *,
    text: str,
    brand: str,
    competitors: list[Competitor],
    sentiment: Sentiment,
    brand_match_names: list[str] | None = None,
    context: BrandContext | None = None,
) -> tuple[bool, int | None, int, list[ResponsePart]]:
    mentions = find_mentions(text, brand, competitors, brand_match_names, context=context)
    mentioned = any(m.kind == "brand" for m in mentions)
    rank = compute_rank(mentions, brand)
    vis = compute_visibility_score(mentioned, rank, sentiment)
    parts = build_parts(text, mentions)
    return mentioned, rank, vis, parts
