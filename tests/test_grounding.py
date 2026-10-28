"""run_grounding with the model + network edges faked. Encodes the
client-reported source bugs (owned sites, wrong/dead links, visibility that
disagrees with the score) and the brand-free search that replaced the
brand lookup."""
from types import SimpleNamespace

import pytest

from app.models import Competitor, PersonaIn, PersonaResult, Product, SourceIntel, Sources
from app.services import aggregation, grounding
from app.services import source_quality as sq

BRAND = "Disney Cruise Line"
ALIASES = ["Disney Cruise Line", "Disney Cruise", "DCL"]
COMPETITORS = [Competitor(name="Royal Caribbean", match_names=["Royal Caribbean"])]
PERSONAS = [
    PersonaIn(id="p1", title="School Holiday Parents", role="r", pains="x", criteria="y"),
    PersonaIn(id="p2", title="Weekend Couple", role="r", pains="x", criteria="y"),
]
PROMPTS = {
    "p1": ["Which cruise lines from Singapore have proper kids clubs for a 5 and 8 year old?", "second q"],
    "p2": ["Any 3 night sailings out of Singapore with an adults-only pool?"],
}

# Page text the (faked) page fetch "reads". noevidence.example's page discusses
# the brand at length though its title/snippet never say so.
BODIES = {
    "https://noevidence.example/cruise-guide": "Our guide. " + "We booked Disney Cruise Line for the kids. " * 4,
    "https://rivalblog.example/couples": "Royal Caribbean is great for couples. Royal Caribbean again.",
}


def _search(qid, query):
    return SimpleNamespace(type="server_tool_use", id=qid, name="web_search", input={"query": query})


def _results(qid, *pairs):
    return SimpleNamespace(type="web_search_tool_result", tool_use_id=qid,
                           content=[{"url": u, "title": t} for u, t in pairs])


def _cite(url, title, text):
    return {"url": url, "title": title, "cited_text": text}


def _fake_response(leak=False):
    return SimpleNamespace(content=[
        _search("s1", "cruise lines from Singapore proper kids clubs 5 and 8 year old"),
        _results("s1",
                 ("https://www.reddit.com/r/cruise/", "r/cruise"),  # bare hit, no evidence
                 ("https://www.reddit.com/r/cruise/comments/abc/dcl_vs_rc", "DCL vs RCI?"),
                 ("https://momblog.example/disney-cruise-review", "Disney Cruise review"),
                 ("https://deadblog.example/old-post", "Old Disney Cruise post"),
                 ("https://noevidence.example/cruise-guide", "General cruise guide"),
                 ("https://disneycruise.disney.go.com/", "Disney Cruise Line official"),
                 ("https://singaporedisneycruise.com/deals", "Disney Cruise deals"),
                 ("https://cruisedeals.example/singapore", "Book cruises now")),
        _search("s2", "3 night sailings Singapore adults-only pool"),
        _results("s2", ("https://rivalblog.example/couples", "Couples cruising from SG"),
                 ("https://royalcaribbean.com/cruises", "Royal Caribbean cruises")),
        *([_search("s3", "Disney Cruise Line reviews Singapore"),
           _results("s3", ("https://brandlookup.example/disney", "Disney Cruise Line reviews"))] if leak else []),
        SimpleNamespace(type="text", text="Found forums and blogs.", citations=[
            _cite("https://www.reddit.com/r/cruise/comments/abc/dcl_vs_rc", "DCL vs RCI?", "We did Disney Cruise last year and loved it"),
            _cite("https://momblog.example/disney-cruise-review", "Disney Cruise review", "Our Disney Cruise trip"),
        ]),
    ])


ORIGINS = {"reddit.com": "independent", "momblog.example": "independent", "deadblog.example": "independent",
           "noevidence.example": "independent", "rivalblog.example": "independent",
           "cruisedeals.example": "affiliate", "brandlookup.example": "independent"}


@pytest.fixture
def calls(monkeypatch):
    seen = {"search": [], "filter_user": None, "filter_system": None}

    def install(leak=False):
        async def fake_web_search(**kw):
            seen["search"].append(kw)
            return _fake_response(leak)

        async def fake_structured(*, tool_name, user, system, **_):
            if tool_name == grounding._FILTER_TOOL_NAME:
                seen["filter_user"], seen["filter_system"] = user, system
                return {"relevantDomains": list(ORIGINS)}
            return {"sources": [{"domain": d, "type": "Community" if d == "reddit.com" else "Blog",
                                 "keywords": ["cruise"], "origin": o} for d, o in ORIGINS.items()]}

        async def fake_pages(urls, **_):
            return {u: sq.PageCheck(alive=("deadblog" not in u), text=BODIES.get(u, "")) for u in urls}

        monkeypatch.setattr(grounding, "call_web_search", fake_web_search)
        monkeypatch.setattr(grounding, "call_structured", fake_structured)
        monkeypatch.setattr(grounding.sq, "check_pages", fake_pages)

    install()
    seen["install"] = install
    return seen


async def _run(market="Singapore"):
    return await grounding.run_grounding(
        api_key="k", brand=BRAND, industry="Cruises", competitors=COMPETITORS,
        brand_match_names=ALIASES, personas=PERSONAS, prompts=PROMPTS, market=market,
    )


# ── the search itself is brand-free ──────────────────────────────────

async def test_search_sends_one_question_per_persona_and_never_the_brand(calls):
    await _run()
    kw = calls["search"][0]
    sent = (kw["system"] + kw["user"]).lower()
    for name in ("disney", "dcl", "royal caribbean"):
        assert name not in sent
    assert "1. Which cruise lines from Singapore" in kw["user"] and "2. Any 3 night sailings" in kw["user"]
    assert "second q" not in kw["user"]  # one question per persona
    assert kw["max_uses"] == 2
    # SG is rejected as a country code; city + timezone is accepted
    assert kw["user_location"] == {"type": "approximate", "city": "Singapore", "timezone": "Asia/Singapore"}


async def test_relevance_filter_never_sees_brand_or_competitor_names(calls):
    await _run()
    # Page titles may contain a brand (that's the page); the instructions and
    # context we add must not.
    context = calls["filter_user"].split("Pages (domain: title):")[0] + calls["filter_system"]
    for name in ("disney", "dcl", "royal caribbean"):
        assert name not in context.lower()


async def test_a_search_that_names_the_brand_is_dropped_and_logged(calls):
    calls["install"](leak=True)
    sources, _ = await _run()
    assert "brandlookup.example" not in {s.domain for s in sources.source_intel}
    dropped = [q for q in sources.queries if q.dropped]
    assert [q.query for q in dropped] == ["Disney Cruise Line reviews Singapore"]


# ── filtering, links, measurement ────────────────────────────────────

async def test_owned_competitor_affiliate_and_dead_sources_are_dropped(calls):
    sources, sitelist = await _run()
    domains = {s.domain for s in sources.source_intel}
    assert domains == {"reddit.com", "momblog.example", "noevidence.example", "rivalblog.example"}
    assert {c.domain for c in sources.citations} == domains
    assert {p.name for p in sources.publishers} | {c.name for c in sources.communities} == domains
    assert {e.name for e in sitelist} <= domains


async def test_link_points_at_the_page_with_the_evidence_not_the_first_hit(calls):
    sources, _ = await _run()
    reddit = next(s for s in sources.source_intel if s.domain == "reddit.com")
    assert reddit.url.startswith("https://www.reddit.com/r/cruise/comments/")


async def test_brand_and_competitors_are_measured_and_gap_classified(calls):
    sources, _ = await _run()
    by = {s.domain: s for s in sources.source_intel}
    assert by["reddit.com"].visibility == "Visible" and by["reddit.com"].gap is None
    # page text names the brand though title/snippet don't
    assert by["noevidence.example"].visibility == "Visible" and "Disney Cruise Line" in by["noevidence.example"].evidence
    rival = by["rivalblog.example"]
    assert rival.visibility == "Not visible" and rival.mentions == 0
    assert rival.competitor_mentions == {"Royal Caribbean": rival.competitor_mentions["Royal Caribbean"]}
    assert rival.competitor_mentions["Royal Caribbean"] >= 2
    assert rival.competitors == ["Royal Caribbean"] and rival.gap == "competitor-owned"


async def test_sources_are_attributed_to_the_persona_whose_question_surfaced_them(calls):
    sources, _ = await _run()
    by = {s.domain: s for s in sources.source_intel}
    assert by["momblog.example"].personas == ["School Holiday Parents"]
    assert by["rivalblog.example"].personas == ["Weekend Couple"]
    assert {q.persona for q in sources.queries} == {"School Holiday Parents", "Weekend Couple"}
    assert sources.coverage_note is None


async def test_source_authority_agrees_with_the_badges(calls):
    sources, _ = await _run()
    named = sum(1 for s in sources.source_intel if s.mentions > 0)
    results = {"p1": PersonaResult(prompt="q", mentioned=False, sentiment="Neutral", vis=0, rank=None, quote="", parts=[], opportunity="Critical Gap", exchanges=[])}
    breakdown = aggregation.build_score_breakdown(results, [Product(name=BRAND, count=0, share=0.0, is_brand=True)], BRAND, sources)
    authority = next(c for c in breakdown if c.name == "Source Authority")
    assert authority.score == round(named / len(sources.source_intel) * 100)


async def test_zero_named_sources_is_flagged_low_confidence():
    intel = [SourceIntel(domain="a.example", url="https://a.example/x", category="PUBLISHER", influence="Low authority",
                         cited=1, keywords=[], personas=[], competitors=[], visibility="Not visible")]
    sources = Sources(citations=[], publishers=[], communities=[], source_intel=intel)
    results = {"p1": PersonaResult(prompt="q", mentioned=False, sentiment="Neutral", vis=0, rank=None, quote="", parts=[], opportunity="Critical Gap", exchanges=[])}
    breakdown = aggregation.build_score_breakdown(results, [Product(name=BRAND, count=0, share=0.0, is_brand=True)], BRAND, sources)
    authority = next(c for c in breakdown if c.name == "Source Authority")
    assert authority.score == 0 and "low-confidence" in authority.note


# ── degradation ──────────────────────────────────────────────────────

async def test_enrichment_failure_shows_no_sources_rather_than_unverified_ones(calls, monkeypatch):
    async def failing(*, tool_name, **_):
        if tool_name == grounding._FILTER_TOOL_NAME:
            return {"relevantDomains": []}  # empty => filter fails open
        raise RuntimeError("enrichment down")

    monkeypatch.setattr(grounding, "call_structured", failing)
    sources, sitelist = await _run()
    # cruisedeals.example (affiliate, not named after any brand) must not slip through
    assert sources.source_intel == [] and sitelist == []
    assert "independence" in sources.coverage_note


async def test_model_returned_domains_are_normalised(calls, monkeypatch):
    async def www_everywhere(*, tool_name, **_):
        if tool_name == grounding._FILTER_TOOL_NAME:
            return {"relevantDomains": [f"www.{d}" for d in ORIGINS] + ["HTTPS://Reddit.com/r/x"]}
        return {"sources": [{"domain": f"WWW.{d.upper()}", "type": "Blog", "keywords": [], "origin": o}
                            for d, o in ORIGINS.items()]}

    monkeypatch.setattr(grounding, "call_structured", www_everywhere)
    sources, _ = await _run()
    assert {s.domain for s in sources.source_intel} == {"reddit.com", "momblog.example", "noevidence.example", "rivalblog.example"}


async def test_badge_is_measured_on_the_linked_page_not_a_dead_sibling(calls, monkeypatch):
    async def response(**_):
        return SimpleNamespace(content=[
            _search("s1", PROMPTS["p1"][0]),
            _results("s1", ("https://blog.example/disney-cruise-review", "Disney Cruise review"),
                     ("https://blog.example/kids-clubs", "Kids clubs compared")),
        ])

    async def structured(*, tool_name, **_):
        if tool_name == grounding._FILTER_TOOL_NAME:
            return {"relevantDomains": ["blog.example"]}
        return {"sources": [{"domain": "blog.example", "type": "Blog", "keywords": [], "origin": "independent"}]}

    async def pages(urls, **_):  # the page naming the brand is dead
        return {u: sq.PageCheck(alive="review" not in u, text="Royal Caribbean kids club" if "kids" in u else "") for u in urls}

    monkeypatch.setattr(grounding, "call_web_search", response)
    monkeypatch.setattr(grounding, "call_structured", structured)
    monkeypatch.setattr(grounding.sq, "check_pages", pages)
    sources, _ = await _run()
    src = sources.source_intel[0]
    assert src.url == "https://blog.example/kids-clubs"
    assert src.visibility == "Not visible" and src.mentions == 0 and src.evidence == ""
    assert src.gap == "competitor-owned"


async def test_failed_searches_are_reported_as_failures(calls, monkeypatch):
    async def all_errors(**_):
        return SimpleNamespace(content=[
            _search("s1", PROMPTS["p1"][0]),
            SimpleNamespace(type="web_search_tool_result", tool_use_id="s1",
                            content={"type": "web_search_tool_result_error", "error_code": "unavailable"}),
        ])

    monkeypatch.setattr(grounding, "call_web_search", all_errors)
    sources, _ = await _run()
    assert sources.source_intel == []
    assert sources.queries[0].failed
    assert "failed" in sources.coverage_note and "thin" not in sources.coverage_note


async def test_reach_counts_distinct_buyer_questions(calls):
    sources, sitelist = await _run()
    by = {s.domain: s for s in sources.source_intel}
    assert by["rivalblog.example"].influence == "High reach · 1 of 2 buyer questions"
    assert "AI" not in by["reddit.com"].influence
    assert by["reddit.com"].cited == 2  # distinct pages, not raw hits (2 results + 1 citation)


async def test_no_prompts_means_no_search_and_an_explicit_note(calls):
    sources, sitelist = await grounding.run_grounding(
        api_key="k", brand=BRAND, industry="Cruises", competitors=COMPETITORS, personas=PERSONAS, prompts={},
    )
    assert calls["search"] == [] and sources.source_intel == [] and sitelist == []
    assert "No buyer questions" in sources.coverage_note


async def test_search_failure_is_an_explicit_note(calls, monkeypatch):
    async def boom(**_):
        raise RuntimeError("down")

    monkeypatch.setattr(grounding, "call_web_search", boom)
    sources, _ = await _run()
    assert sources.source_intel == [] and "failed" in sources.coverage_note


def test_query_attribution_handles_trimmed_searches_and_falls_back_to_order():
    # real questions + the trimmed searches the model ran for them (live run)
    qs = [
        "Which cruises from Marina Bay Cruise Centre have proper kids clubs for a 5 and 8 year old?",
        "Planning a cruise for my parents in their 70s plus my siblings and their kids. Which ships from Singapore have connecting cabins?",
        "2 night cruise from Singapore with a decent adults-only area so we can escape the kids?",
        "Daughter turning 10 and I want one splurge cruise she'll remember. Which line does the best themed experiences for that age?",
    ]
    got = grounding.attribute_queries({
        "a": "cruises from Marina Bay Cruise Centre kids clubs",
        "b": "ships from Singapore connecting cabins family",
        "c": "2 night cruise from Singapore adults-only area",
        "d": "themed experiences for 10 year old cruise",
    }, qs)
    assert got == {"a": 0, "b": 1, "c": 2, "d": 3}
    # nothing convincing -> next unclaimed question; nothing left -> None
    assert grounding.attribute_queries({"a": "kids clubs cruises Marina Bay", "x": "weather today"}, qs[:2]) == {"a": 0, "x": 1}
    assert grounding.attribute_queries({"a": "kids clubs cruises Marina Bay", "x": "weather today"}, qs[:1]) == {"a": 0, "x": None}


def test_market_to_search_location():
    assert grounding._location_for("Singapore")["city"] == "Singapore"
    assert "country" not in grounding._location_for("SG")
    assert grounding._location_for("Malaysia") == {"type": "approximate", "country": "MY"}
    assert grounding._location_for("Atlantis") is None
    assert grounding._location_for(None) is None


def test_prompt_writer_receives_persona_description():
    persona = PersonaIn(id="p1", title="Budget Parent", desc="Planning a first family cruise", role="r", pains="x", criteria="y")
    from app.services import prompt_gen
    text = prompt_gen._user_prompt("B", "Cruises", "ctx", None, None, [persona], 3)
    assert "Planning a first family cruise" in text


async def test_question_choice_rotates_so_later_categories_get_searched(calls):
    prompts = {"p1": ["cat A for p1", "cat B for p1"], "p2": ["cat A for p2", "cat B for p2"]}
    await grounding.run_grounding(api_key="k", brand=BRAND, industry="Cruises", competitors=COMPETITORS,
                                  brand_match_names=ALIASES, personas=PERSONAS, prompts=prompts)
    user = calls["search"][-1]["user"]
    assert "1. cat A for p1" in user and "2. cat B for p2" in user


def test_reach_label_has_no_tier_with_a_single_question():
    assert grounding._reach_label(1, 1) == "1 of 1 buyer question"
    assert grounding._reach_label(3, 4) == "High reach · 3 of 4 buyer questions"


def test_reach_has_one_definition_for_label_and_sitelist():
    from app.models import Publisher

    assert grounding.reach_share(1, 1) is None and grounding.reach_share(3, 4) == 75
    [single] = grounding._build_sitelist([Publisher(name="a.com", type="Blog", mentions=1)], {"a.com": 1}, 1)
    assert single.score is None and grounding._reach_label(1, 1) == "1 of 1 buyer question"


async def test_a_failed_page_check_shows_no_unchecked_links(calls, monkeypatch):
    async def broken(urls, **_):
        raise RuntimeError("fetcher down")

    monkeypatch.setattr(grounding.sq, "check_pages", broken)
    sources, sitelist = await _run()
    assert sources.source_intel == [] and sitelist == []
    assert sources.coverage_note.startswith("Source links couldn't be checked")
    assert sources.queries  # the searches that ran are still reported
