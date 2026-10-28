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


# ── job errors reach the client as JSON, without internals ────────────

def test_error_events_are_valid_json_without_internal_detail():
    import json

    from app.errors import user_facing_error
    from app.job_store import JobStore

    store = JobStore(ttl_seconds=60)
    job = store.create_job(["p1"])
    queue = store.subscribe(job.job_id)
    store.fail(job.job_id, user_facing_error(RuntimeError("db at 10.0.0.5 said no")))
    event = queue.get_nowait()
    payload = json.loads(event.split("data: ", 1)[1])
    assert payload == {"message": "Something went wrong during the analysis."}
    late = store.subscribe(job.job_id).get_nowait()  # replay for a late subscriber
    assert json.loads(late.split("data: ", 1)[1]) == payload


# ── request size limits ───────────────────────────────────────────────

def test_requests_reject_oversized_input():
    from pydantic import ValidationError

    from app.models import AnalysisRequest, DetectRequest, PromptsRequest

    persona = {"id": "p1", "title": "T", "role": "r", "pains": "p", "criteria": "c"}
    base = {"brand": "B", "industry": "i", "accessCode": "AAAA-BBBB", "personas": [persona], "prompts": {"p1": ["q"]}}
    AnalysisRequest(**base)
    for bad in (
        {"prompts": {"p1": ["x" * 501]}},
        {"personas": [persona] * 11},
        {"competitors": [{"name": f"C{i}", "matchNames": [f"C{i}"]} for i in range(16)]},
        {"brandSummary": "x" * 2001},
    ):
        with pytest.raises(ValidationError):
            AnalysisRequest(**{**base, **bad})
    with pytest.raises(ValidationError):
        DetectRequest(query="x" * 501, accessCode="A")
    with pytest.raises(ValidationError):
        PromptsRequest(**{**base, "promptsPerPersona": 1000})


def test_detect_output_fits_the_request_limits():
    from app.services.detect import _normalize_brand_match_names, _normalize_competitors

    names = _normalize_brand_match_names(["X", "Disney", "disney", *[f"Ship {i}" for i in range(40)]], "Disney Cruise Line")
    assert names[0] == "Disney Cruise Line" and "X" not in names and len(names) == 25  # derived 1-letter alias dropped
    assert _normalize_brand_match_names([], "X") == ["X"]  # a brand really called X keeps its own name
    comps = _normalize_competitors([{"name": f"Line {i}", "matchNames": ["L"]} for i in range(20)] + ["Q"])
    assert len(comps) == 15 and all(c.match_names == [c.name] for c in comps)


def test_every_model_produced_field_fits_the_request_it_is_sent_back_in():
    from app.models import AnalysisRequest, CategorySuggestion, Competitor, fit

    long = "word " * 1000
    assert len(fit(long, 120)) <= 120 and fit(long, 120).endswith("…")
    assert fit("short", 120) == "short"
    category = CategorySuggestion(name=fit(long, 120), buyer_context=fit(long, 2000))
    AnalysisRequest(brand=fit(long, 120), industry=fit(long, 500), brandSummary=fit(long, 2000),
                    buyerContext=category.buyer_context, accessCode="A",
                    competitors=[Competitor(name="X", match_names=["X"])],  # user-typed 1-letter rival
                    personas=[{"id": "p1", "title": "T", "role": "r", "pains": "p", "criteria": "c"}],
                    prompts={"p1": ["q"]})


async def test_category_suggestions_are_fitted(monkeypatch):
    from app.services import category_gen

    async def fake(**_):
        return {"categories": [{"name": "n" * 400, "buyerContext": "c " * 3000}]}

    monkeypatch.setattr(category_gen, "call_structured", fake)
    [suggestion] = await category_gen.generate_categories(api_key="k", brand="B", industry="i", competitors=[],
                                                          buyer_context=None, brand_summary=None)
    assert len(suggestion.name) <= 120 and len(suggestion.buyer_context) <= 2000


async def test_prompt_output_budget_scales_and_extra_prompts_are_cut(monkeypatch):
    from app.models import PersonaIn
    from app.services import prompt_gen

    seen = {}

    async def fake_structured(*, max_tokens, **_):
        seen["max_tokens"] = max_tokens
        return {"personas": [{"personaId": f"p{i}", "prompts": [f"q{i}-{j}" for j in range(12)]} for i in range(10)]}

    monkeypatch.setattr(prompt_gen, "call_structured", fake_structured)
    personas = [PersonaIn(id=f"p{i}", title="T", role="r", pains="p", criteria="c") for i in range(10)]
    out = await prompt_gen.generate_prompts(api_key="k", brand="B", industry="i", buyer_context=None,
                                            personas=personas, prompts_per_persona=10)
    assert seen["max_tokens"] > 2048
    assert all(len(p) == 10 for p in out.values())


def test_insights_ask_for_distinct_points_without_a_padding_floor():
    from app.services import insights_gen

    assert "distinct point" in insights_gen._SYSTEM and "rather than padding" in insights_gen._SYSTEM
    description = insights_gen._INPUT_SCHEMA["properties"]["opportunities"]["description"]
    assert "4-6" not in description and "at least 2" in description
