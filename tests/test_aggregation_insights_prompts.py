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
    # Ranking and Sentiment count at Presence's strength (80%): 24+24+12.8+6.4 over 0.9
    assert aggregation.compute_composite_score(breakdown, has_sources=False) == 75
    assert aggregation.compute_composite_score(breakdown) == 67  # a real 0 still counts


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
    assert "mentioned in 1 of 2 answers" in text
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



def test_failed_prompts_are_reported_not_hidden():
    from app.models import PersonaExchange, PersonaResult, Sentiment, Sources
    from app.services import aggregation

    ex = PersonaExchange(prompt="q", mentioned=True, sentiment=Sentiment.positive, rank=1, vis=100, quote="", parts=[])
    result = PersonaResult(prompt="q", mentioned=True, sentiment=Sentiment.positive, vis=100, rank=1, quote="",
                           parts=[], exchanges=[ex], opportunity="Defend", failed_prompts=2)
    results = {"p1": result}
    assert aggregation.build_overview(results, 50).failed_prompts == 2
    products = aggregation.build_products(results, "B", [])
    presence = aggregation.build_score_breakdown(results, products, "B", Sources(citations=[], publishers=[], communities=[]))[0]
    assert presence.note == "Mentioned in 1 of 1 persona queries (2 more couldn't be answered this run)"


def test_empty_optional_fields_from_the_form_are_accepted():
    """Confirmed live: an empty Market field 422'd persona generation."""
    from app.models import AnalysisRequest, GeneratePersonasRequest, PromptsRequest

    persona = {"id": "p1", "title": "T", "role": "r", "pains": "p", "criteria": "c"}
    assert GeneratePersonasRequest(brand="B", industry="i", accessCode="A", market="").market is None
    assert GeneratePersonasRequest(brand="B", industry="i", accessCode="A", market="  ").market is None
    assert PromptsRequest(brand="B", industry="i", accessCode="A", personas=[persona], market="").market is None
    request = AnalysisRequest(
        brand="B", industry="i", accessCode="A", market="", personas=[persona],
        prompts={"p1": ["q", "", "  "]}, brandMatchNames=["B", ""],
        competitors=[{"name": "R", "matchNames": ["R", ""]}],
    )
    assert request.market is None and request.prompts == {"p1": ["q"]}
    assert request.brand_match_names == ["B"] and request.competitors[0].match_names == ["R"]


def test_share_of_voice_scores_against_the_leader_not_raw_share():
    """Confirmed live: Disney led 10 to 9 and scored a red 29/100."""
    from app.models import Product
    from app.services.aggregation import share_of_voice

    def products(brand_count, *rivals):
        total = brand_count + sum(rivals) or 1
        return [Product(name="Disney", count=brand_count, share=brand_count / total, is_brand=True)] + [
            Product(name=f"R{i}", count=c, share=c / total) for i, c in enumerate(rivals)
        ]

    assert share_of_voice(products(10, 9, 6, 5, 2, 2)) == (100, "29% of all product mentions · #1 of 6 brands named")
    score, note = share_of_voice(products(3, 10, 2))
    assert score == 30 and note.endswith("#2 of 3 brands named (leader: R0, 10 mentions to your 3)")
    assert share_of_voice(products(5, 5))[0] == 100  # tied for first
    assert share_of_voice(products(0, 4, 1)) == (0, "Not named in any answer; 2 other brands were")
    # adding unnamed rivals changes nothing
    assert share_of_voice(products(10, 9, 0, 0))[0] == 100



# ── one lucky answer, unlisted brands, several answers per prompt ────

from app.models import AnswerSample, Competitor, PersonaExchange, Sources


def _prompt(*samples):
    first = samples[0]
    return PersonaExchange(prompt="q", mentioned=first.mentioned, sentiment=first.sentiment, rank=first.rank,
                           vis=first.vis, quote="", parts=[], samples=list(samples))


def _answer(names, brand_rank=None, sentiment=Sentiment.positive):
    mentioned = brand_rank is not None
    return AnswerSample(mentioned=mentioned, sentiment=sentiment if mentioned else Sentiment.neutral,
                        rank=brand_rank, vis=100 if brand_rank == 1 else (80 if mentioned else 0), names=names)


def _results(exchanges):
    return {"p": PersonaResult(prompt="q", mentioned=True, sentiment=Sentiment.neutral, vis=0, rank=None, quote="",
                               parts=[], opportunity="High", exchanges=exchanges)}


def _headline(exchanges):
    results = _results(exchanges)
    products = aggregation.build_products(results, "Nike", [Competitor(name="Hoka", match_names=["Hoka"])])
    breakdown = aggregation.build_score_breakdown(results, products, "Nike",
                                                  Sources(citations=[], publishers=[], communities=[]))
    return aggregation.compute_composite_score(breakdown, has_sources=False)


def test_one_lucky_mention_no_longer_jumps_the_headline():
    # Confirmed before the fix: 1 mention in 12 answers (first, positive) scored 39.
    never = [_prompt(_answer(["Hoka"])) for _ in range(12)]
    once = [_prompt(_answer(["Nike", "Hoka"], 1))] + never[:11]
    assert _headline(never) == 0
    assert _headline(once) <= 10


def test_every_answer_to_a_prompt_counts():
    results = _results([_prompt(_answer(["Nike"], 1), _answer(["Hoka"]), _answer(["Nike"], 1))])
    assert aggregation.mention_counts(results) == (2, 3)
    breakdown = aggregation.build_score_breakdown(results, aggregation.build_products(results, "Nike", []), "Nike",
                                                  Sources(citations=[], publishers=[], communities=[]))
    assert "2 of 3 answers (each query asked 3 times)" in breakdown[0].note


def test_unlisted_brands_count_in_share_of_voice_as_one_brand_each():
    results = _results([_prompt(_answer(["Saucony", "Brooks", "Nike"], 3)), _prompt(_answer(["Brooks Running", "Hoka"])),
                        _prompt(_answer(["saucony"]))])
    products = {p.name: p for p in aggregation.build_products(results, "Nike", [Competitor(name="Hoka", match_names=["Hoka"])])}
    assert products["Saucony"].count == 2 and products["Brooks"].count == 2  # spelling variants merged
    assert not products["Saucony"].listed and products["Hoka"].listed and products["Nike"].is_brand
    assert abs(sum(p.share for p in products.values()) - 1) < 0.01


# ── prompts that reword one question ─────────────────────────────────

# Confirmed live: each selected category was generated in its own call, and
# every persona got one scenario three times over.
REWORDED = [
    "First cruise for our 4 and 7 year old from Singapore, which ship has the best kids club and character meet-ups?",
    "Which cruise from Singapore has the best kids club for a 4 and 7 year old, and not packed?",
    "First cruise with a 4 and 7 year old from Singapore, which ship's kids club is actually good and not packed?",
]
TEENS = [
    "My 13 and 16 yo will roll their eyes at kiddie stuff. Any Singapore cruises with proper teen clubs?",
    "Our 13 and 16 yo think everything is babyish. Any short getaway from SG with proper teen-only zones and thrill stuff?",
]
DISTINCT = [
    "Which cruise lines sailing out of Marina Bay have kids clubs that work for a 4 and 7 year old?",
    "June hols cruise from Singapore under $4k for family of 4, any ship where the kids club doesn't get fully booked?",
    "Ah Ma turning 70, want a 3 night cruise from Singapore with proper halal food onboard. Which lines are reliable?",
]


def test_near_duplicates_catch_rewordings_but_not_distinct_questions():
    assert len(prompt_gen.near_duplicate_pairs(REWORDED)) == 3
    assert prompt_gen.near_duplicate_pairs(TEENS) == [(0, 1)]
    assert prompt_gen.near_duplicate_pairs(DISTINCT) == []


def _categories():
    return [prompt_gen.CategoryPlan(name="Family cruises", buyer_context="Parents choosing a cruise", count=2),
            prompt_gen.CategoryPlan(name="Short getaways", buyer_context="Couples picking a weekend cruise", count=1)]


async def test_categories_are_written_in_one_call_in_order(monkeypatch):
    calls = []

    async def fake(**kw):
        calls.append(kw)
        return {"personas": [{"personaId": "p1", "prompts": DISTINCT}]}

    monkeypatch.setattr(prompt_gen, "call_structured", fake)
    out = await prompt_gen.generate_prompts(api_key="k", brand="B", industry="Cruises", buyer_context=None,
                                            personas=[PERSONA], prompts_per_persona=3, categories=_categories())
    assert out == {"p1": DISTINCT} and len(calls) == 1
    user = calls[0]["user"]
    assert "prompts 1-2: Family cruises" in user and "prompt 3: Short getaways" in user
    assert user.index("Family cruises") < user.index("Short getaways")


async def test_a_reworded_set_is_asked_for_once_more_and_the_better_set_kept(monkeypatch):
    replies = [{"personas": [{"personaId": "p1", "prompts": REWORDED}]},
               {"personas": [{"personaId": "p1", "prompts": DISTINCT}]}]
    calls = []

    async def fake(**kw):
        calls.append(kw)
        return replies[len(calls) - 1]

    monkeypatch.setattr(prompt_gen, "call_structured", fake)
    out = await prompt_gen.generate_prompts(api_key="k", brand="B", industry="Cruises", buyer_context=None,
                                            personas=[PERSONA], prompts_per_persona=3)
    assert out == {"p1": DISTINCT}
    assert len(calls) == 2 and "repeated one question" in calls[1]["user"] and "Parent" in calls[1]["user"]


async def test_a_retry_that_is_no_better_keeps_the_first_set(monkeypatch):
    calls = []

    async def fake(**kw):
        calls.append(kw)
        return {"personas": [{"personaId": "p1", "prompts": REWORDED}]}

    monkeypatch.setattr(prompt_gen, "call_structured", fake)
    out = await prompt_gen.generate_prompts(api_key="k", brand="B", industry="Cruises", buyer_context=None,
                                            personas=[PERSONA], prompts_per_persona=3)
    assert out == {"p1": REWORDED} and len(calls) == 2  # asked once more, never in a loop


# ── categories and personas built around one brand ───────────────────

async def test_categories_only_the_brand_fills_or_that_name_a_company_are_dropped(monkeypatch):
    from app.services import category_gen

    async def fake(**_):
        return {"categories": [
            # Confirmed live for Disney: a facet only it fills.
            {"name": "Premium Themed Entertainment Cruises", "buyerContext": "Families paying more for branded characters",
             "competitorsHere": []},
            {"name": "Disney Character Sailings", "buyerContext": "Fans of the characters", "competitorsHere": ["Royal Caribbean"]},
            {"name": "Kids Club Cruises", "buyerContext": "Parents comparing Royal Caribbean and others", "competitorsHere": ["Royal Caribbean"]},
            {"name": "Short Family Cruises", "buyerContext": "Parents picking a 3-night family sailing",
             "competitorsHere": ["royal caribbean"]},
        ]}

    monkeypatch.setattr(category_gen, "call_structured", fake)
    out = await category_gen.generate_categories(
        api_key="k", brand="Disney Cruise Line", industry="Cruises", competitors=COMPS, buyer_context=None,
        brand_summary=None, brand_names=["Disney Cruise Line", "Disney"],
    )
    assert [c.name for c in out] == ["Short Family Cruises"]


async def test_persona_text_never_names_the_brand(monkeypatch):
    from app.services import persona_gen

    async def fake(**_):
        return {"personas": [{"title": "Adult Disney Superfan", "initials": "ADS", "desc": "Loves Disney shows at sea",
                              "role": "Planner", "pains": "Royal Caribbean felt crowded", "criteria": "Shows"}]}

    monkeypatch.setattr(persona_gen, "call_structured", fake)
    [p] = await persona_gen.generate_personas(
        api_key="k", industry="Cruises", competitors=COMPS, buyer_context=None, persona_count=1,
        brand="Disney Cruise Line", brand_names=["Disney Cruise Line", "Disney"],
    )
    assert p.title == "Adult Superfan" and p.initials == "AS"
    assert "Disney" not in p.desc and "Royal Caribbean" not in p.pains


async def test_a_persona_entry_that_is_not_an_object_is_skipped_not_a_crash(monkeypatch):
    from app.services import persona_gen

    async def fake(**_):
        return {"personas": ["junk", {"title": "Planner", "desc": "d", "role": "r", "pains": "p", "criteria": "c"}]}

    monkeypatch.setattr(persona_gen, "call_structured", fake)
    out = await persona_gen.generate_personas(api_key="k", industry="Cruises", competitors=[], buyer_context=None,
                                              persona_count=2)
    assert [p.title for p in out] == ["Planner"]
