"""Scoring consistency, insights parsing and prompt screening."""
import pytest
from pydantic import ValidationError

from app.models import (
    AnalysisRequest, Competitor, Opportunity, PersonaExchange, PersonaIn, PersonaResult, RadarCategory,
    ScoreComponent, Sentiment,
)
from app.services import aggregation, insights_gen, prompt_gen


def _ex(mentioned, sentiment):
    return PersonaExchange(prompt="q", mentioned=mentioned, sentiment=sentiment, rank=1 if mentioned else None,
                           vis=50, quote="", parts=[])


def test_composite_leaves_source_authority_out_when_there_are_no_sources():
    breakdown = [ScoreComponent(name=n, score=s, note="") for n, s in
                 [("Presence", 80), ("Share of Voice", 80), ("Ranking", 80), ("Sentiment", 80), ("Source Authority", 0)]]
    assert aggregation.compute_composite_score(breakdown, has_sources=False) == 80
    assert aggregation.compute_composite_score(breakdown) == 72  # a real 0 still counts


def test_overview_sentiment_uses_only_prompts_that_mentioned_the_brand():
    # 4 positive mentions out of 12 prompts; the rest are non-mentions (Neutral)
    exchanges = [_ex(True, Sentiment.positive)] * 4 + [_ex(False, Sentiment.neutral)] * 8
    results = {f"p{i}": PersonaResult(prompt="q", mentioned=True, sentiment=Sentiment.neutral, vis=0, rank=None,
                                      quote="", parts=[], opportunity="High", exchanges=exchanges[i * 3:(i + 1) * 3])
               for i in range(4)}
    assert aggregation.build_overview(results, 50).avg_sentiment == Sentiment.positive


def test_one_bad_insight_entry_does_not_drop_the_rest():
    good = {"title": "t", "type": "Source gap", "impact": "High", "effort": "Low", "detail": "d", "action": "a"}
    bad = {**good, "impact": "Huge"}
    assert len(insights_gen._parse_each([good, bad, "junk"], Opportunity)) == 1
    radar = insights_gen._parse_each([{"name": "x", "score": 150}, {"name": "y"}], RadarCategory)
    assert [r.name for r in radar] == ["x"]


def test_insights_prompt_uses_prompt_level_counts_and_measured_sources():
    from app.models import SourceIntel, Sources
    persona = PersonaIn(id="p1", title="Parent", role="r", pains="p", criteria="c")
    result = PersonaResult(prompt="q", mentioned=True, sentiment=Sentiment.positive, vis=50, rank=1, quote="",
                           parts=[], opportunity="High", exchanges=[_ex(True, Sentiment.positive), _ex(False, Sentiment.neutral)])
    intel = SourceIntel(domain="a.example", url="https://a.example/x", category="PUBLISHER", influence="High reach",
                        cited=1, keywords=[], personas=["Parent"], competitors=["Rival"], visibility="Not visible",
                        competitor_mentions={"Rival": 3}, gap="competitor-owned")
    text = insights_gen._user_prompt(
        brand="B", industry="I", buyer_context=None, brand_summary=None, market=None, competitors=[],
        personas=[persona], results={"p1": result}, products=[],
        sources=Sources(citations=[], publishers=[], communities=[], source_intel=[intel]),
    )
    assert "mentioned in 1 of 2 buyer queries" in text
    assert '"gap": "competitor-owned"' in text and '"Rival": 3' in text


# ── prompt screening ─────────────────────────────────────────────────

COMPS = [Competitor(name="Royal Caribbean", match_names=["Royal Caribbean"])]
PERSONA = PersonaIn(id="p1", title="Parent", role="r", pains="p", criteria="c")


async def test_generated_prompts_naming_a_brand_or_rival_are_dropped(monkeypatch):
    async def fake(**_):
        return {"personas": [{"personaId": "p1", "prompts": [
            "Is Disney Cruise worth it for kids?", "Royal Caribbean or something else?", "best kids club cruise from SG"]}]}

    monkeypatch.setattr(prompt_gen, "call_structured", fake)
    out = await prompt_gen.generate_prompts(api_key="k", brand="Disney Cruise Line", industry="Cruises",
                                            buyer_context=None, personas=[PERSONA], prompts_per_persona=3,
                                            competitors=COMPS, brand_match_names=["Disney Cruise Line", "Disney Cruise"])
    assert out == {"p1": ["best kids club cruise from SG"]}


async def test_malformed_output_falls_back_to_a_neutral_question_not_a_500(monkeypatch):
    async def fake(**_):
        return {"personas": [{"prompts": ["no id"]}, "junk", {"personaId": "p1", "prompts": "not a list"}]}

    monkeypatch.setattr(prompt_gen, "call_structured", fake)
    out = await prompt_gen.generate_prompts(api_key="k", brand="B", industry="Travel — Cruises", buyer_context=None,
                                            market="Singapore", personas=[PERSONA], prompts_per_persona=3)
    [question] = out["p1"]
    assert question == "Which cruises do people in Singapore actually recommend, and why?"
    assert "Parent" not in question and "—" not in question


async def test_seed_adaptation_that_adds_a_brand_falls_back_to_the_seed(monkeypatch):
    async def fake(**_):
        return {"personas": [{"personaId": "p1", "prompt": "Is Royal Caribbean good for kids?"}]}

    monkeypatch.setattr(prompt_gen, "call_structured", fake)
    out = await prompt_gen.adapt_seed_prompt(api_key="k", brand="Disney Cruise Line", industry="Cruises",
                                             personas=[PERSONA], seed_prompt="best cruise for kids?", competitors=COMPS)
    assert out == {"p1": "best cruise for kids?"}


def test_analysis_request_is_bounded():
    base = dict(brand="B", industry="I", access_code="x", prompts={})
    with pytest.raises(ValidationError):
        AnalysisRequest(**base, personas=[PERSONA] * 11)
    with pytest.raises(ValidationError):
        AnalysisRequest(**{**base, "prompts": {"p1": ["q"] * 11}}, personas=[PERSONA])
    AnalysisRequest(**{**base, "prompts": {"p1": ["q"] * 3}}, personas=[PERSONA] * 4)


async def test_custom_persona_expansion_is_told_not_to_name_brands(monkeypatch):
    from app.services import persona_gen
    seen = {}

    async def fake(*, system, **_):
        seen["system"] = system
        return {"title": "T", "initials": "T", "desc": "d", "role": "r", "pains": "p", "criteria": "c"}

    monkeypatch.setattr(persona_gen, "call_structured", fake)
    await persona_gen.expand_custom_persona(api_key="k", description="a mum", industry="Cruises",
                                            competitors=COMPS, buyer_context=None)
    assert persona_gen._NO_BRANDS_RULE in seen["system"]
