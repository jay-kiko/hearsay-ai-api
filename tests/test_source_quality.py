from app.services import source_quality as sq
from app.services.scoring import count_brand_mentions

BRAND = "Disney Cruise Line"
ALIASES = ["Disney Cruise Line", "Disney Cruise", "DCL"]


def mentions(text: str) -> int:
    return count_brand_mentions(text, BRAND, ALIASES)


# ── owned / competitor domains ───────────────────────────────────────

def test_entity_domains_are_excluded():
    slugs = sq.entity_slugs([BRAND, *ALIASES, "Royal Caribbean"])
    assert sq.is_entity_domain("disneycruise.disney.go.com", slugs)
    assert sq.is_entity_domain("singaporedisneycruise.com", slugs)  # affiliate/fan site named after the brand
    assert sq.is_entity_domain("royalcaribbean.com", slugs)
    assert not sq.is_entity_domain("reddit.com", slugs)
    assert not sq.is_entity_domain("cruisecritic.com", slugs)


def test_short_names_do_not_match_unrelated_hosts():
    slugs = sq.entity_slugs(["DCL", "Ana"])  # too short to trust against hostnames
    assert slugs == set()
    assert not sq.is_entity_domain("scandinaviandcl.example", slugs)


def test_host_of_strips_www_and_port():
    assert sq.host_of("https://www.Reddit.com:443/r/x") == "reddit.com"
    assert sq.host_of("not a url") == ""


# ── evidence-based visibility ────────────────────────────────────────

def test_alias_in_passage_counts_even_when_title_lacks_exact_brand():
    # The old check needed the exact brand string in the title; this page's title
    # names neither, but the quoted passage names an alias.
    page = sq.SourcePage(
        url="https://blog.example/a",
        title="Our week at sea with the kids",
        cited_texts=["We sailed Disney Cruise and the kids club was great."],
        cited=True,
    )
    assert sq.page_mentions(page, mentions) == 1


def test_visibility_thresholds():
    assert sq.visibility_from_evidence(0) == "Not visible"
    assert sq.visibility_from_evidence(1) == "Weak visibility"
    assert sq.visibility_from_evidence(2) == "Visible"
    assert sq.visibility_from_evidence(5) == "Visible"


def test_rank_pages_prefers_brand_evidence_then_cited():
    bare_result = sq.SourcePage(url="https://x.example/1", title="Cruise packing list")
    cited_no_brand = sq.SourcePage(url="https://x.example/2", title="Cruise tips", cited_texts=["pack light"], cited=True)
    with_brand = sq.SourcePage(url="https://x.example/3", title="DCL honest review")
    ranked = sq.rank_pages([bare_result, cited_no_brand, with_brand], mentions)
    assert [p.url for p in ranked] == ["https://x.example/3", "https://x.example/2", "https://x.example/1"]


def test_evidence_snippet_returns_first_mentioning_text():
    page = sq.SourcePage(url="u", title="Hello", cited_texts=["Disney Cruise was fine"], cited=True)
    assert sq.evidence_snippet([page], mentions) == "Disney Cruise was fine"
    assert sq.evidence_snippet([sq.SourcePage(url="u", title="Nothing here")], mentions) == ""


# ── link health ──────────────────────────────────────────────────────

def test_judge_response():
    assert sq.judge_response("https://a.example/post/1", 200, "https://a.example/post/1") is True
    assert sq.judge_response("https://a.example/post/1", 404, "https://a.example/post/1") is False
    assert sq.judge_response("https://a.example/post/1", 410, "https://a.example/post/1") is False
    # bot-blocking says nothing about whether the page exists
    assert sq.judge_response("https://a.example/post/1", 403, "https://a.example/post/1") is None
    assert sq.judge_response("https://a.example/post/1", 429, "https://a.example/post/1") is None


def test_deep_link_redirected_to_homepage_is_dead():
    assert sq.judge_response("https://a.example/post/1", 200, "https://a.example/") is False
    # a homepage URL staying on the homepage is fine
    assert sq.judge_response("https://a.example/", 200, "https://a.example/") is True


def test_public_http_url_guard_rejects_integer_and_short_ip_forms():
    for url in ("http://2130706433/", "http://0x7f000001/", "http://127.1/", "http://intranet/", "http://[::1]/"):
        assert not sq.is_public_http_url(url), url


async def test_resolves_public_rejects_names_that_resolve_internally():
    assert not await sq.resolves_public("127.0.0.1")
    assert not await sq.resolves_public("localhost")
    assert not await sq.resolves_public("no-such-host.invalid")


def test_tls_errors_are_recognised():
    import ssl
    err = ConnectionError("x")
    err.__cause__ = ssl.SSLError("bad cert")
    assert sq._is_tls_error(err)
    assert sq._is_tls_error(Exception("[SSL: CERTIFICATE_VERIFY_FAILED]"))
    assert not sq._is_tls_error(Exception("Name or service not known"))


def test_public_http_url_guard():
    assert sq.is_public_http_url("https://example.com/x")
    assert not sq.is_public_http_url("ftp://example.com/x")
    assert not sq.is_public_http_url("http://localhost:8000/")
    assert not sq.is_public_http_url("http://127.0.0.1/")
    assert not sq.is_public_http_url("http://169.254.169.254/latest/meta-data")
    assert not sq.is_public_http_url("http://10.0.0.5/")
    assert not sq.is_public_http_url("http://db.internal/")


# ── page text and URL-slug evidence ──────────────────────────────────

def test_url_slug_counts_as_evidence_when_search_gave_no_passage():
    # the asiaone case from the gold run: title/passage said nothing, URL is obviously about the brand
    page = sq.SourcePage(url="https://www.asiaone.com/lifestyle/disney-cruise-singapore-review", title="Review")
    assert page.slug_text() == "lifestyle disney cruise singapore review"
    assert sq.page_mentions(page, mentions) == 1


def test_page_body_mentions_are_counted_and_capped():
    page = sq.SourcePage(url="https://a.example/x", title="Guide", body="Disney Cruise Line " * 50)
    assert sq.page_mentions(page, mentions) == 5  # capped, no snippet evidence
    assert sq.visibility_from_evidence(sq.page_mentions(page, mentions)) == "Visible"


def test_evidence_snippet_falls_back_to_page_text_excerpt():
    body = "x " * 300 + "We loved Disney Cruise Line because of the kids club. " + "y " * 300
    page = sq.SourcePage(url="https://a.example/x", title="Guide", body=body)
    snippet = sq.evidence_snippet([page], mentions)
    assert "Disney Cruise Line" in snippet and len(snippet) <= 230


def test_html_to_text_strips_scripts_and_tags():
    raw = "<html><script>var a='Disney Cruise';</script><style>p{}</style><body><h1>Hi &amp; bye</h1><p>text</p></body></html>"
    text = sq.html_to_text(raw)
    assert text == "Hi & bye text"


def test_proxy_wrappers_cannot_smuggle_an_entity_site_past_the_host_check():
    url = "https://translate.google.com/translate?u=https://www.singaporedisneycruise.com/deals&hl=en"
    assert sq.is_proxy_host("translate.google.com")
    assert "singaporedisneycruise.com" in sq.hosts_in(url)
    slugs = sq.entity_slugs([BRAND, *ALIASES])
    assert any(sq.is_entity_domain(h, slugs) for h in sq.hosts_in(url))
    archived = "https://web.archive.org/web/2020/https://disneycruise.disney.go.com/"
    assert any(sq.is_entity_domain(h, slugs) for h in sq.hosts_in(archived))
    assert sq.hosts_in("https://reddit.com/r/x") == ["reddit.com"]
