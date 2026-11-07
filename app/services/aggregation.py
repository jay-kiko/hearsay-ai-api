"""§06 Aggregation.

One pass over every persona result — no model call, everything needed is
already sitting in the per-persona results. Produces the Results screen's
overview cards (mention rate, average sentiment, top competitor) and the
share-of-voice product list, by running the same mention-detection logic
across every persona's response rather than just the brand's own.
"""
from __future__ import annotations

import math
from collections import Counter

from app.models import (
    AnswerSample,
    Competitor,
    DetailRow,
    HeadlineBreakdown,
    HeadlineRow,
    Overview,
    PersonaExchange,
    PersonaIn,
    PersonaResult,
    Product,
    ScoreComponent,
    ScoreDetail,
    Sentiment,
    SourceIntel,
    Sources,
)
from app.services.source_quality import UNVERIFIED
from app.services.scoring import SENTIMENT_POINTS, name_matches, rank_score



def _all_exchanges(results: dict[str, PersonaResult]) -> list[PersonaExchange]:
    return [ex for r in results.values() for ex in r.exchanges]


def all_answers(results: dict[str, PersonaResult]) -> list[AnswerSample]:
    """Every answer to every prompt: what every mention-based number counts
    (invariant 8), so they agree with each other. An exchange from before
    prompts were asked several times stands for its one answer."""
    return [
        sample
        for ex in _all_exchanges(results)
        for sample in (ex.samples or [AnswerSample(
            mentioned=ex.mentioned, sentiment=ex.sentiment, rank=ex.rank, vis=ex.vis,
            names=list(dict.fromkeys(p.name for p in ex.parts if p.name)),
        )])
    ]


def mention_counts(results: dict[str, PersonaResult]) -> tuple[int, int]:
    """Mentioned/total across every answer to every prompt. Not one OR'd
    boolean per persona: that lets a persona with more prompts count as
    "mentioned" just from having more tries."""
    answers = all_answers(results)
    return sum(1 for a in answers if a.mentioned), len(answers)


def answers_per_prompt(results: dict[str, PersonaResult]) -> int:
    """How many answers each prompt got (the most any prompt got), for notes."""
    return max((len(ex.samples) for ex in _all_exchanges(results)), default=1) or 1


def _failed_prompts(results: dict[str, PersonaResult]) -> int:
    return sum(r.failed_prompts for r in results.values())


def build_overview(results: dict[str, PersonaResult], visibility_score: int) -> Overview:
    mentioned, total = mention_counts(results)
    mention_rate = (mentioned / total) if total else 0.0

    # Same basis as the breakdown's Sentiment: only prompts that mentioned the
    # brand. Persona-level sentiment also counts non-mentions (forced Neutral),
    # which put "Neutral" next to "4 positive · 0 neutral · 0 negative".
    sentiment_counts = Counter(a.sentiment for a in all_answers(results) if a.mentioned)
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


def _canonical_others(names: list[str]) -> dict[str, str]:
    """Maps each unlisted brand name to one spelling: case variants and a
    longer form of a shorter name ("Brooks Running" -> "Brooks") count as one
    brand. The most-used spelling wins."""
    by_count = Counter(names)
    ordered = sorted(by_count, key=lambda n: (-by_count[n], len(n)))
    canonical: dict[str, str] = {}
    for name in ordered:
        canonical[name] = next(
            (c for c in dict.fromkeys(canonical.values())
             if c.lower() == name.lower() or name_matches(c, name, caseless=True)),
            name,
        )
    return canonical


def build_products(results: dict[str, PersonaResult], brand: str, competitors: list[Competitor]) -> list[Product]:
    """Answers naming each brand: the brand, listed competitors, and every
    other brand the answers named (`listed=False`). Counting only the listed
    ones measured share against an incomplete field (confirmed live: unlisted
    Saucony and Brooks came up more often than listed Hoka)."""
    answers = all_answers(results)
    listed = [(brand, True)] + [(c.name, False) for c in competitors]
    listed_names = {name for name, _ in listed}
    others = _canonical_others([n for a in answers for n in a.names if n not in listed_names])

    counts: Counter[str] = Counter()
    for a in answers:
        counts.update({others.get(n, n) for n in a.names})
    entries = [(name, is_brand, True) for name, is_brand in listed]
    entries += [(name, False, False) for name in dict.fromkeys(others.values())]

    # Each product's slice of *all* product mentions, so shares sum to 100%
    # (dividing by persona count summed to 126% when answers named several).
    total_mentions = sum(counts[name] for name, _, _ in entries) or 1
    products = [
        Product(name=name, count=counts[name], share=round(counts[name] / total_mentions, 3),
                is_brand=is_brand, listed=is_listed)
        for name, is_brand, is_listed in entries
    ]
    products.sort(key=lambda p: (-p.count, p.name))
    return products


def share_of_voice(products: list[Product]) -> tuple[int, str]:
    """(score, note). The score is the brand's mentions relative to the
    most-mentioned brand (leader = 100), not its raw share: with six brands
    named, even a clear leader rarely passes 30-40% of all mentions, so the
    raw share read as a failing grade (confirmed live: Disney led with 10
    mentions to Royal Caribbean's 9 and scored a red 29/100). Raw share also
    fell whenever more competitors were listed, with nothing about the brand
    changing. The note keeps the real share and adds the rank."""
    brand = next((p for p in products if p.is_brand), None)
    named = [p for p in products if p.count > 0]
    if brand is None or brand.count == 0:
        return 0, f"Not named in any answer; {len(named)} other brand{'s' if len(named) != 1 else ''} were"
    leader = max(named, key=lambda p: p.count)
    score = score_round(brand.count / leader.count * 100)
    rank = 1 + sum(1 for p in named if p.count > brand.count)
    note = f"{score_round(brand.share * 100)}% of all product mentions · #{rank} of {len(named)} brands named"
    if rank > 1:
        note += f" (leader: {leader.name}, {leader.count} mentions to your {brand.count})"
    return score, note


def build_score_breakdown(
    results: dict[str, PersonaResult], products: list[Product], brand: str, sources: Sources,
    personas: list[PersonaIn] | None = None,
) -> list[ScoreComponent]:
    """Six deterministic sub-scores behind the single visibilityScore — no
    extra AI call, everything here is math over data the pipeline/grounding
    already produced. Each carries its working (`details`), built from the
    same variables as its score, so what a client reads can't disagree with
    the number. `personas` only names them in Presence's rows."""
    answers = all_answers(results)
    mentioned_exchanges = [a for a in answers if a.mentioned]
    mentioned_prompts = len(mentioned_exchanges)
    total_prompts = len(answers)
    per_prompt = answers_per_prompt(results)

    presence = score_round(mentioned_prompts / total_prompts * 100) if total_prompts else 0

    sov, sov_note = share_of_voice(products)

    # Recommendation Strength / Ranking / Sentiment average over the same
    # mentioned exchanges Presence counts, so the card shares one denominator.
    rec_strength = score_round(sum(ex.vis for ex in mentioned_exchanges) / mentioned_prompts) if mentioned_prompts else 0

    # Each answer's own position score, averaged. Rounding the average
    # position first moved the score in steps of 20: #1.4 read 100, #1.6 80.
    ranked = [ex.rank for ex in mentioned_exchanges if ex.rank is not None]
    avg_rank = sum(ranked) / len(ranked) if ranked else None
    ranking_score = score_round(sum(rank_score(r) for r in ranked) / len(ranked)) if ranked else 0

    sentiment_counts = Counter(ex.sentiment for ex in mentioned_exchanges)
    pos, neu, neg = (
        sentiment_counts.get(Sentiment.positive, 0),
        sentiment_counts.get(Sentiment.neutral, 0),
        sentiment_counts.get(Sentiment.negative, 0),
    )
    sentiment_score = (
        score_round(sum(SENTIMENT_POINTS[s] * c for s, c in sentiment_counts.items()) / mentioned_prompts)
        if mentioned_prompts
        else 0
    )

    # The same measured `mentions` behind each source's Visible badge, so the
    # score and the badges cannot disagree. Only sources whose page could be
    # read count: an unreadable page is unknown, not "not named".
    total_sources = len(sources.source_intel)
    readable = measured_sources(sources)
    named_sources = sum(1 for s in readable if s.mentions > 0)
    unverified = total_sources - len(readable)
    source_authority = score_round(named_sources / len(readable) * 100) if readable else 0

    titles = {p.id: p.title for p in personas or []}
    persona_rows = []
    for pid, result in results.items():
        m, t = mention_counts({pid: result})
        persona_rows.append(DetailRow(label=titles.get(pid, pid), value=f"{m} of {t} answers"))

    return [
        ScoreComponent(
            name="Presence",
            score=presence,
            # Failed prompts are out of the denominator, so say so: "4 of 10"
            # must not read as the whole run when 12 were asked.
            note=f"Mentioned in {mentioned_prompts} of {total_prompts} "
            + (f"answers (each query asked {per_prompt} times)" if per_prompt > 1 else "persona queries")
            + (f" ({failed} more couldn't be answered this run)" if (failed := _failed_prompts(results)) else ""),
            details=ScoreDetail(
                measures="How often AI answers name your brand when buyers ask. Each buyer question is asked "
                         f"{per_prompt} time{'s' if per_prompt != 1 else ''}, since answers vary, and every answer counts.",
                calculation=[f"{mentioned_prompts} of {total_prompts} answers named your brand",
                             f"{mentioned_prompts} ÷ {total_prompts} = {presence}%"] if total_prompts else ["No answers to count"],
                rows_title="By persona",
                rows=persona_rows,
            ),
        ),
        ScoreComponent(
            name="Share of Voice", score=sov, note=sov_note,
            details=_share_of_voice_detail(products, sov),
        ),
        ScoreComponent(
            name="Recommendation Strength",
            score=rec_strength,
            note=(
                f"Average visibility score of {rec_strength} across {mentioned_prompts} mentioning answers"
                if mentioned_prompts
                else "Never mentioned, so no recommendation strength to measure"
            ),
            details=ScoreDetail(
                measures="How strongly the answers that name you put you forward: each answer's position score "
                         "(#1 = 100, 20 less per place), kept in full when the tone is positive, 70% when neutral and "
                         "35% when negative. Not part of the Visibility Score, which counts position and tone "
                         "separately.",
                calculation=(
                    [f"Average over the {mentioned_prompts} answers that named you = {rec_strength}"]
                    if mentioned_prompts else ["No answers named your brand, so there is nothing to average"]
                ),
                rows_title="Answers by score",
                rows=[DetailRow(label=f"Scored {vis}", value=_answers(n))
                      for vis, n in sorted(Counter(ex.vis for ex in mentioned_exchanges).items(), reverse=True)],
            ),
        ),
        ScoreComponent(
            name="Ranking",
            score=ranking_score,
            note=f"Average position #{avg_rank:.1f} among every brand named" if avg_rank else "Not ranked in any answer",
            details=ScoreDetail(
                measures="Where you appear among the brands each answer names, including brands not on your list. "
                         "#1 scores 100, and each place lower 20 less.",
                calculation=(
                    ["(" + " + ".join(f"{n} × {rank_score(r)}" for r, n in sorted(Counter(ranked).items()))
                     + f") ÷ {len(ranked)} = {ranking_score}",
                     f"Average position: #{avg_rank:.1f}"]
                    if ranked else ["Your brand wasn't ranked in any answer"]
                ),
                rows_title="Answers by position",
                rows=[DetailRow(label=f"#{r}", value=f"{_answers(n)} ({rank_score(r)} points)")
                      for r, n in sorted(Counter(ranked).items())],
            ),
        ),
        ScoreComponent(
            name="Sentiment", score=sentiment_score, note=f"{pos} positive · {neu} neutral · {neg} negative",
            details=ScoreDetail(
                measures="How positively the answers that name you talk about you. A positive answer is worth "
                         f"{SENTIMENT_POINTS[Sentiment.positive]}, a neutral one {SENTIMENT_POINTS[Sentiment.neutral]} "
                         f"and a negative one {SENTIMENT_POINTS[Sentiment.negative]}.",
                calculation=(
                    [f"({pos} × {SENTIMENT_POINTS[Sentiment.positive]} + {neu} × {SENTIMENT_POINTS[Sentiment.neutral]}"
                     f" + {neg} × {SENTIMENT_POINTS[Sentiment.negative]}) ÷ {mentioned_prompts} = {sentiment_score}"]
                    if mentioned_prompts else ["No answers named your brand, so there is no tone to score"]
                ),
                rows_title="Answers naming you",
                rows=[DetailRow(label="Positive", value=_answers(pos)), DetailRow(label="Neutral", value=_answers(neu)),
                      DetailRow(label="Negative", value=_answers(neg))],
            ),
        ),
        ScoreComponent(
            name="Source Authority",
            score=source_authority,
            note=_source_authority_note(named_sources, len(readable), unverified),
            details=ScoreDetail(
                measures="How many of the independent pages buyers' own searches turn up name you, counted on "
                         "the page itself. Pages we couldn't read are left out rather than counted against you.",
                calculation=(
                    [f"{named_sources} of {len(readable)} readable sources name you",
                     f"{named_sources} ÷ {len(readable)} = {source_authority}%"]
                    if readable else ["No readable sources, so this is left out of the Visibility Score"]
                ),
                rows_title="Sources",
                rows=[DetailRow(label=s.domain, value="Names you" if s.mentions > 0 else "Doesn't name you")
                      for s in readable]
                + [DetailRow(label=s.domain, value="Couldn't be read")
                   for s in sources.source_intel if s.visibility == UNVERIFIED],
            ),
        ),
    ]


def score_round(value: float) -> int:
    """Half up, as a client checking the working would round: Python's round()
    sends halves to the even number, so (100 + 3 x 50) / 4 = 62.5 read 62."""
    return int(math.floor(value + 0.5))


def _possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _answers(n: int) -> str:
    return f"{n} answer{'s' if n != 1 else ''}"


def _share_of_voice_detail(products: list[Product], score: int) -> ScoreDetail:
    brand = next((p for p in products if p.is_brand), None)
    named = [p for p in products if p.count > 0]
    measures = ("Your share of brand mentions, compared with the most-mentioned brand: the leader scores 100. "
                "Brands not on your competitor list count too.")
    listed = [p for p in products if p.listed is not False and p.count > 0]
    others = [p for p in products if p.listed is False and p.count > 0]
    rows = [DetailRow(label=p.name, value=f"{_answers(p.count)} · {score_round(p.share * 100)}% of mentions") for p in listed]
    if others:
        rows.append(DetailRow(
            label=f"{len(others)} brand{'s' if len(others) != 1 else ''} not on your list",
            value=f"{_answers(sum(p.count for p in others))} in total · "
                  f"{score_round(sum(p.share for p in others) * 100)}% of mentions",
        ))
    if brand is None or brand.count == 0:
        return ScoreDetail(measures=measures, calculation=["No answer named your brand, so 0"],
                           rows_title="Answers naming each brand", rows=rows)
    leader = max(named, key=lambda p: p.count)
    calculation = (
        [f"You are the most-named brand ({_answers(brand.count)}), so {score}"]
        if leader.count == brand.count else
        [f"Your {_answers(brand.count)} ÷ {_possessive(leader.name)} {leader.count} × 100 = {score}"]
    )
    calculation.append(f"Your share of all brand mentions: {score_round(brand.share * 100)}%")
    return ScoreDetail(measures=measures, calculation=calculation, rows_title="Answers naming each brand", rows=rows)


def measured_sources(sources: Sources) -> list[SourceIntel]:
    """Sources whose brand presence was actually measured (page or snippets
    read). The one definition used by Source Authority and by whether it
    counts toward the headline score."""
    return [s for s in sources.source_intel if s.visibility != UNVERIFIED]


def _source_authority_note(named: int, readable: int, unverified: int) -> str:
    if not readable:
        if unverified:
            return (f"None of the {unverified} sources could be read, so this is left out of the "
                    "visibility score")
        return "No sources found, so this is left out of the visibility score"
    note = f"Named in {named} of {readable} sources we could read"
    if unverified:
        note += f" ({unverified} more block automated reading)"
    return note


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


# Averaged over only the answers that name the brand, so they say how well it
# does when it comes up, not how often. In the headline they count in
# proportion to Presence: confirmed, one mention in 12 answers (ranked first,
# positive) scored Ranking 100 and Sentiment 100 and took the headline from 0
# to 39.
_SCALED_BY_PRESENCE = ("Ranking", "Sentiment")


def headline_breakdown(score_breakdown: list[ScoreComponent], *, has_sources: bool = True) -> HeadlineBreakdown:
    """The Visibility Score and the points each part contributes: the one
    definition the score comes from, so the table a client reads always adds
    up to it. With no sources there is nothing to measure Source Authority on,
    so it is left out and the other weights re-normalised — scoring it 0 would
    cut the headline by up to 10 points over a search that came back empty."""
    weights = {k: w for k, w in _COMPOSITE_WEIGHTS.items() if has_sources or k != "Source Authority"}
    total = sum(weights.values())
    presence = next((c.score for c in score_breakdown if c.name == "Presence"), 0) / 100
    rows = []
    for c in score_breakdown:
        if c.name not in weights:
            continue
        factor = presence if c.name in _SCALED_BY_PRESENCE else 1.0
        weight = weights[c.name] / total if total else 0.0
        rows.append(HeadlineRow(name=c.name, score=c.score, weight=round(weight, 4), factor=round(factor, 4),
                                points=round(weight * c.score * factor, 2)))
    score = score_round(sum(weights[c.name] * c.score * (presence if c.name in _SCALED_BY_PRESENCE else 1.0)
                      for c in score_breakdown if c.name in weights) / total) if total else 0
    notes = [
        "Ranking and Sentiment count in proportion to Presence: they describe how you do when you're named, "
        "so a brand named rarely can't score high on them alone.",
        "Recommendation Strength isn't added on top: it combines position and tone, which Ranking and "
        "Sentiment already count.",
    ]
    if not has_sources:
        notes.append("Source Authority is left out because no source could be measured, and the other weights "
                     "are scaled up to fill its share.")
    return HeadlineBreakdown(score=score, rows=rows, notes=notes)


def compute_composite_score(score_breakdown: list[ScoreComponent], *, has_sources: bool = True) -> int:
    return headline_breakdown(score_breakdown, has_sources=has_sources).score


def top_competitor(products: list[Product], brand: str) -> str | None:
    competitor_products = [p for p in products if p.name != brand and p.count > 0]
    if not competitor_products:
        return None
    return max(competitor_products, key=lambda p: p.count).name
