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


def _response_for(query, leak=False):
    """The fake response's blocks for the one question a call searched: s1
    (and the summary citations) for the kids-club question, s2 (and the leaked
    brand search) for the couples one."""
    blocks = _fake_response(leak).content
    ids = {"s1"} if "kids clubs" in query else {"s2", "s3"}
    return SimpleNamespace(content=[
        b for b in blocks
        if getattr(b, "id", None) in ids or getattr(b, "tool_use_id", None) in ids
        or (b.type == "text" and "s1" in ids)
    ])


ORIGINS = {"reddit.com": "independent", "momblog.example": "independent", "deadblog.example": "independent",
           "noevidence.example": "independent", "rivalblog.example": "vendor",
           "cruisedeals.example": "affiliate", "brandlookup.example": "independent"}


@pytest.fixture
def calls(monkeypatch):
    seen = {"search": [], "filter_user": None, "filter_system": None, "web_fetch": [], "web_fetch_pages": {},
            "page_kinds": {}, "page_kind_user": None}

    def install(leak=False):
        async def fake_web_search(**kw):
            seen["search"].append(kw)
            return _response_for(kw["user"], leak)

        async def fake_structured(*, tool_name, user, system, **_):
            if tool_name == grounding._FILTER_TOOL_NAME:
                seen["filter_user"], seen["filter_system"] = user, system
                return {"relevantDomains": list(ORIGINS)}
            if tool_name == grounding._PAGE_KIND_TOOL_NAME:
                seen["page_kind_user"] = user
                urls = [line[5:] for line in user.splitlines() if line.startswith("URL: ")]
                return {"pages": [{"url": u, "kind": seen["page_kinds"].get(u, "article")} for u in urls]}
            return {"sources": [{"domain": d, "type": "Community" if d == "reddit.com" else "Blog",
                                 "keywords": ["cruise"], "origin": o} for d, o in ORIGINS.items()]}

        async def fake_pages(urls, **_):
            return {u: sq.PageCheck(alive=("deadblog" not in u), text=BODIES.get(u, "")) for u in urls}

        async def fake_web_fetch(*, url, **_):
            # Never the network: pages a test puts in web_fetch_pages read, others fail.
            seen["web_fetch"].append(url)
            return seen["web_fetch_pages"].get(url)

        monkeypatch.setattr(grounding, "call_web_search", fake_web_search)
        monkeypatch.setattr(grounding, "call_structured", fake_structured)
        monkeypatch.setattr(grounding.sq, "check_pages", fake_pages)
        monkeypatch.setattr(grounding, "call_web_fetch", fake_web_fetch)

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
    # One call per persona's question, each limited to that one search.
    assert [kw["user"] for kw in calls["search"]] == [PROMPTS["p1"][0], PROMPTS["p2"][0]]
    assert all(kw["max_uses"] == 1 for kw in calls["search"])
    for kw in calls["search"]:
        sent = (kw["system"] + kw["user"]).lower()
        for name in ("disney", "dcl", "royal caribbean"):
            assert name not in sent
        assert "user_location" not in kw  # it didn't localise anything (confirmed live)


async def test_search_names_the_market_when_the_question_does_not(calls):
    await _run(market="Makati, Philippines")
    assert calls["search"][0]["user"] == (
        "Which cruise lines from Singapore have proper kids clubs for a 5 and 8 year old in Philippines"
    )


def test_search_query_adds_the_country_only_when_no_part_of_the_market_is_named():
    q = "Which cloud accounting app gives BIR-ready reports?"
    assert grounding.search_query(q, "Makati, Philippines") == (
        "Which cloud accounting app gives BIR-ready reports in Philippines")
    assert grounding.search_query("Best app for a cafe in Makati?", "Makati, Philippines") == (
        "Best app for a cafe in Makati?")
    assert grounding.search_query("payroll add-ons in kenya?", "Nairobi, Kenya") == "payroll add-ons in kenya?"
    assert grounding.search_query("cheap flights to Perugia", "Lima, Peru") == "cheap flights to Perugia in Peru"
    assert grounding.search_query("Which option has CPF payroll built in?", "Singapore") == (
        "Which option has CPF payroll built in Singapore")
    assert grounding.search_query(q, None) == q and grounding.search_query(q, "") == q


async def test_the_relevance_filter_is_told_to_drop_other_markets_pages(calls):
    await _run()
    assert "Market: Singapore" in calls["filter_user"]
    assert "buyers in a different country" in calls["filter_system"]


async def test_one_failed_search_keeps_the_others_and_is_logged(calls, monkeypatch):
    async def first_fails(**kw):
        if "kids clubs" in kw["user"]:
            raise RuntimeError("down")
        return _response_for(kw["user"])

    monkeypatch.setattr(grounding, "call_web_search", first_fails)
    sources, _ = await _run()
    failed = [q for q in sources.queries if q.failed]
    assert len(failed) == 1 and failed[0].persona == "School Holiday Parents"
    assert {s.domain for s in sources.source_intel} == {"rivalblog.example"}


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


def _authority(visibilities):
    intel = [SourceIntel(domain=f"{i}.example", url=f"https://{i}.example/x", category="PUBLISHER", influence="",
                         cited=1, keywords=[], personas=[], competitors=[], visibility=v, mentions=int(v == "Visible") * 2)
             for i, v in enumerate(visibilities)]
    sources = Sources(citations=[], publishers=[], communities=[], source_intel=intel)
    results = {"p1": PersonaResult(prompt="q", mentioned=False, sentiment="Neutral", vis=0, rank=None, quote="", parts=[], opportunity="Critical Gap", exchanges=[])}
    breakdown = aggregation.build_score_breakdown(results, [Product(name=BRAND, count=0, share=0.0, is_brand=True)], BRAND, sources)
    return next(c for c in breakdown if c.name == "Source Authority"), sources


async def test_source_authority_counts_only_pages_we_could_read():
    authority, sources = _authority(["Visible", "Not visible", "Couldn't verify", "Couldn't verify"])
    assert authority.score == 50  # 1 of 2 readable, not 1 of 4
    assert authority.note == "Named in 1 of 2 sources we could read (2 more block automated reading)"
    assert aggregation.measured_sources(sources)

    authority, sources = _authority(["Couldn't verify"])
    assert authority.note.startswith("None of the 1 sources could be read")
    assert not aggregation.measured_sources(sources)  # so it's left out of the headline score


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
    async def www_everywhere(*, tool_name, user="", **_):
        if tool_name == grounding._FILTER_TOOL_NAME:
            return {"relevantDomains": [f"www.{d}" for d in ORIGINS] + ["HTTPS://Reddit.com/r/x"]}
        if tool_name == grounding._PAGE_KIND_TOOL_NAME:
            return {"pages": [{"url": line[5:], "kind": "article"} for line in user.splitlines() if line.startswith("URL: ")]}
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



def test_prompt_writer_receives_persona_description():
    persona = PersonaIn(id="p1", title="Budget Parent", desc="Planning a first family cruise", role="r", pains="x", criteria="y")
    from app.services import prompt_gen
    text = prompt_gen._user_prompt("B", "Cruises", "ctx", None, None, [persona], 3)
    assert "Planning a first family cruise" in text


async def test_question_choice_rotates_so_later_categories_get_searched(calls):
    prompts = {"p1": ["cat A for p1", "cat B for p1"], "p2": ["cat A for p2", "cat B for p2"]}
    await grounding.run_grounding(api_key="k", brand=BRAND, industry="Cruises", competitors=COMPETITORS,
                                  brand_match_names=ALIASES, personas=PERSONAS, prompts=prompts)
    assert [kw["user"] for kw in calls["search"]] == ["cat A for p1", "cat B for p2"]


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


# ── a page we couldn't read is unknown, and links point at the mention ──

def _rivalblog_blocked(monkeypatch):
    """rivalblog 403s our fetcher (alive unknown, no text); the rest as usual."""
    async def pages(urls, **_):
        return {u: sq.PageCheck(alive=None) if "rivalblog" in u else sq.PageCheck(alive="deadblog" not in u, text=BODIES.get(u, ""))
                for u in urls}

    monkeypatch.setattr(grounding.sq, "check_pages", pages)


RIVALBLOG = "https://rivalblog.example/couples"


async def test_a_blocked_page_is_read_through_web_fetch_and_measured(calls, monkeypatch):
    from datetime import datetime, timedelta, timezone

    from app.services.anthropic_client import FetchedPage

    _rivalblog_blocked(monkeypatch)
    old_copy = datetime.now(timezone.utc) - timedelta(days=60)
    calls["web_fetch_pages"][RIVALBLOG] = FetchedPage(
        text="Couples love it. Disney Cruise Line also has adult areas. Royal Caribbean too.", retrieved_at=old_copy
    )
    sources, _ = await _run()
    rival = {s.domain: s for s in sources.source_intel}["rivalblog.example"]
    assert rival.visibility in ("Visible", "Weak visibility") and rival.mentions >= 1
    assert rival.competitor_mentions["Royal Caribbean"] >= 1
    assert rival.link == RIVALBLOG + "#:~:text=Disney%20Cruise%20Line"
    assert rival.read_from_copy == old_copy.date().isoformat()  # a stale copy is never shown as live


async def test_a_page_no_reader_can_open_is_dropped_but_stays_in_the_sitelist(calls, monkeypatch):
    _rivalblog_blocked(monkeypatch)  # and web_fetch can't read it either
    sources, sitelist = await _run()
    assert "rivalblog.example" not in {s.domain for s in sources.source_intel}
    assert "rivalblog.example" in {e.name for e in sitelist}
    assert "1 more page turned up but couldn't be read" in sources.coverage_note


async def test_web_fetch_is_only_a_fallback(calls):
    await _run()
    fetched = set(calls["web_fetch"])
    assert not any(u in fetched for u in BODIES)  # pages we read ourselves are never sent
    assert not any("deadblog" in u for u in fetched)  # dead pages are never revived from a copy


async def test_web_fetch_is_capped_per_run(calls, monkeypatch):
    async def all_blocked(urls, **_):
        return {u: sq.PageCheck(alive=None) for u in urls}

    monkeypatch.setattr(grounding.sq, "check_pages", all_blocked)
    monkeypatch.setattr(grounding.get_settings(), "web_fetch_max_pages", 2)
    await _run()
    assert len(calls["web_fetch"]) == 2


async def test_link_scrolls_to_the_first_on_page_mention(calls):
    sources, _ = await _run()
    by = {s.domain: s for s in sources.source_intel}
    page = by["noevidence.example"]
    assert page.link == page.url + "#:~:text=Disney%20Cruise%20Line"
    # mentioned only in the title/slug: nothing on the page to point at
    assert by["rivalblog.example"].link == by["rivalblog.example"].url


def test_text_fragment_link_encoding():
    assert sq.text_fragment_link("https://a.example/p", "Love, Bonito") == "https://a.example/p#:~:text=Love%2C%20Bonito"
    assert sq.text_fragment_link("https://a.example/p", "Co-op") == "https://a.example/p#:~:text=Co%2Dop"
    assert sq.text_fragment_link("https://a.example/p#top", "X Y") == "https://a.example/p#top:~:text=X%20Y"
    assert sq.text_fragment_link("https://a.example/p", "  ") == "https://a.example/p"


async def test_sources_count_the_parent_name_in_context_and_drop_searches_that_use_it(calls, monkeypatch):
    async def pages(urls, **_):
        text = {"https://rivalblog.example/couples": "For couples, Disney is a calm pick on its newest ship. Royal Caribbean too."}
        return {u: sq.PageCheck(alive="deadblog" not in u, text=text.get(u, BODIES.get(u, ""))) for u in urls}

    monkeypatch.setattr(grounding.sq, "check_pages", pages)
    sources, _ = await grounding.run_grounding(
        api_key="k", brand=BRAND, industry="Cruises", competitors=COMPETITORS, brand_match_names=ALIASES,
        personas=PERSONAS, prompts=PROMPTS, market="Singapore",
        brand_context_names=["Disney"], category_terms=["cruise", "ship"],
    )
    rival = {s.domain: s for s in sources.source_intel}["rivalblog.example"]
    assert rival.mentions == 1 and rival.visibility == "Weak visibility"


async def test_a_search_naming_the_parent_brand_is_dropped(calls):
    calls["install"](leak=True)  # one search is "Disney Cruise Line reviews Singapore"
    sources, _ = await grounding.run_grounding(
        api_key="k", brand="Some Cruise Brand", industry="Cruises", competitors=[], brand_match_names=["Some Cruise Brand"],
        personas=PERSONAS, prompts=PROMPTS, market="Singapore", brand_context_names=["Disney"], category_terms=["cruise"],
    )
    assert any(q.dropped for q in sources.queries)


def test_coverage_note_says_none_rather_than_only_zero():
    asked = [("P", "q")]
    log = [grounding.SourceQuery(persona="P", question="q", query="q")]
    assert grounding._coverage_note(asked, log, {"0:a": "q"}, set(), []) == (
        "No usable sources came back for these buyer questions.")
    one = [SourceIntel.model_construct()]
    assert grounding._coverage_note(asked, log, {"0:a": "q"}, set(), one).startswith("Only 1 usable source came")


async def test_vendor_articles_are_shown_labelled_and_stores_never(calls):
    sources, _ = await _run()
    origin = {s.domain: s.origin for s in sources.source_intel}
    assert origin["rivalblog.example"] == "vendor"
    assert origin["reddit.com"] == "independent"
    assert "cruisedeals.example" not in origin  # a store ("affiliate"), never a source


async def test_a_vendor_page_selling_itself_is_not_a_source(calls):
    calls["page_kinds"]["https://rivalblog.example/couples"] = "product_or_service"
    sources, _ = await _run()
    assert "rivalblog.example" not in {s.domain for s in sources.source_intel}
    # only vendor pages are checked; independent ones never are
    assert "rivalblog.example" in calls["page_kind_user"] and "reddit.com" not in calls["page_kind_user"]


async def test_vendor_pages_are_hidden_when_their_check_fails(calls, monkeypatch):
    real = grounding.call_structured

    async def page_check_fails(*, tool_name, **kw):
        if tool_name == grounding._PAGE_KIND_TOOL_NAME:
            raise RuntimeError("down")
        return await real(tool_name=tool_name, **kw)

    monkeypatch.setattr(grounding, "call_structured", page_check_fails)
    sources, _ = await _run()
    domains = {s.domain for s in sources.source_intel}
    assert "rivalblog.example" not in domains and "reddit.com" in domains


def test_front_pages_are_never_source_links():
    for url in ("https://sync.ke/", "https://ke.endasportswear.com", "https://dora.pe/es-pe/", "https://x.com/en"):
        assert sq.is_homepage(url), url
    for url in ("https://www.reddit.com/r/cruise/", "https://blog.example/kids-clubs", "https://x.com/sg/blog"):
        assert not sq.is_homepage(url), url
