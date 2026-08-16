"""§05 Source grounding.

One Claude call per job (not per persona), with the native `web_search` tool
enabled, asking how the brand and its competitors are discussed in the
category. Real citation URLs get classified by domain into publishers vs.
communities. The AI Influence Sitelist reuses this same grounding call for
publisher identification and scoring, but CPM/availability/buyable stay
placeholder — that's ad-inventory data no search turns up (see §05/§08 of the
design doc).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from urllib.parse import urlparse

from app.config import get_settings
from app.models import Citation, Community, Competitor, Publisher, SitelistEntry, Sources
from app.services.anthropic_client import call_structured, call_web_search

logger = logging.getLogger("hearsay.grounding")

_SYSTEM = (
    "Use the web_search tool to research how the given brand and its competitors "
    "are actually discussed in their category right now. Search for comparison "
    "articles, review sites, and community discussion (Reddit, forums, Hacker "
    "News). The brand name may be shared by other, unrelated companies or "
    "products in a completely different industry — stay focused on results "
    "genuinely about this brand in this specific category, alongside the named "
    "competitors, and disregard anything about a different company that happens "
    "to share the name. Then briefly summarize what you found in 2-3 sentences."
)

_FILTER_TOOL_NAME = "filter_relevant_domains"
_FILTER_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "relevantDomains": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The domains from the candidate list that are genuinely about the specific "
                "brand being researched, in its actual industry — excluding any domain that's "
                "actually about a different, unrelated company or product sharing the same name."
            ),
        }
    },
    "required": ["relevantDomains"],
}

_COMMUNITY_DOMAINS = (
    "reddit.com",
    "news.ycombinator.com",
    "ycombinator.com",
    "quora.com",
    "stackoverflow.com",
    "stackexchange.com",
    "discourse.org",
    "indiehackers.com",
)


def _query(brand: str, industry: str, competitors: list[Competitor], market: str | None) -> str:
    competitor_list = ", ".join(c.name for c in competitors) if competitors else "its main competitors"
    market_clause = f" Focus specifically on the {market} market." if market else ""
    return (
        f"How is '{brand}' discussed and reviewed within the '{industry}' category, "
        f"compared to {competitor_list}?{market_clause} Find real articles, review sites, and "
        f"community threads. Note: '{brand}' may be a name shared by unrelated products "
        f"in other industries — only report on results genuinely about this brand in "
        f"the '{industry}' category."
    )


def _is_community(domain: str) -> bool:
    domain = domain.lower()
    if any(domain == d or domain.endswith("." + d) for d in _COMMUNITY_DOMAINS):
        return True
    return "forum" in domain


def _extract_raw_citations(response) -> list[dict[str, str]]:
    """Best-effort extraction of {url, title} pairs from a web_search-enabled response."""
    found: list[dict[str, str]] = []

    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)

        # Inline citations attached to generated text arrive as plain dicts
        # (e.g. {"url": ..., "title": ...}), not typed objects — getattr()
        # on a dict always misses and silently returns the fallback, which
        # is exactly why this used to come back empty no matter how good
        # the search results were.
        citations = getattr(block, "citations", None) or []
        for citation in citations:
            url = _field(citation, "url")
            title = _field(citation, "title")
            if url:
                found.append({"url": url, "title": title or url})

        # Raw web_search tool results — also plain dicts inside block.content.
        if block_type == "web_search_tool_result":
            content = getattr(block, "content", None) or []
            for result in content:
                url = _field(result, "url")
                title = _field(result, "title")
                if url:
                    found.append({"url": url, "title": title or url})

    return found


def _field(item, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


async def _filter_relevant(
    raw_citations: list[dict[str, str]],
    *,
    brand: str,
    industry: str,
    competitors: list[Competitor],
    api_key: str,
    model: str,
) -> list[dict[str, str]]:
    """Web search for an ambiguous brand name can surface an unrelated company
    that happens to share it — confirmed live (a PM software brand's search
    results included a completely unrelated ADHD-focused app of the same
    name). One cheap structured pass over the candidate domains, anchored on
    the actual industry/competitors, before anything gets classified into
    citations/publishers/communities. Fails open (keeps everything) on any
    error, rather than let a filtering hiccup silently empty real results.
    """
    if not raw_citations:
        return raw_citations

    domains: dict[str, str] = {}
    for item in raw_citations:
        try:
            domain = urlparse(item["url"]).netloc.replace("www.", "")
        except (KeyError, ValueError):
            continue
        if domain:
            domains.setdefault(domain, item.get("title", domain))

    if not domains:
        return raw_citations

    candidate_list = "\n".join(f"- {domain}: {title}" for domain, title in domains.items())
    competitor_list = ", ".join(c.name for c in competitors) if competitors else "none specified"

    try:
        result = await call_structured(
            api_key=api_key,
            model=model,
            system=(
                "You confirm which search-result domains are genuinely about a specific real "
                "brand, versus an unrelated company or product that happens to share its name. "
                "Keep a domain only if its title plausibly relates to the given brand in the "
                "given industry, alongside its named competitors."
            ),
            user=(
                f"Brand: {brand}\nIndustry: {industry}\nCompetitors: {competitor_list}\n\n"
                f"Candidate domains found while researching this brand:\n{candidate_list}\n\n"
                f"Which of these are genuinely about '{brand}' in the '{industry}' category?"
            ),
            tool_name=_FILTER_TOOL_NAME,
            tool_description="Return the domains that are genuinely relevant.",
            input_schema=_FILTER_INPUT_SCHEMA,
            max_tokens=1024,
        )
    except Exception:
        logger.exception("relevance filter failed, keeping all citations unfiltered")
        return raw_citations

    keep = {d.strip().lower() for d in result.get("relevantDomains", []) if isinstance(d, str)}
    if not keep:
        return raw_citations  # nothing usable came back — fail open, not empty

    return [
        item
        for item in raw_citations
        if urlparse(item.get("url", "")).netloc.replace("www.", "").lower() in keep
    ]


def _classify(raw_citations: list[dict[str, str]]) -> Sources:
    domain_counts: dict[str, int] = defaultdict(int)
    domain_titles: dict[str, str] = {}

    for item in raw_citations:
        try:
            domain = urlparse(item["url"]).netloc.replace("www.", "")
        except (KeyError, ValueError):
            continue
        if not domain:
            continue
        domain_counts[domain] += 1
        domain_titles.setdefault(domain, item.get("title", domain))

    citations: list[Citation] = []
    publishers: list[Publisher] = []
    communities: list[Community] = []

    for domain, count in sorted(domain_counts.items(), key=lambda kv: -kv[1]):
        title = domain_titles[domain]
        citations.append(Citation(domain=domain, title=title, count=count))
        if _is_community(domain):
            communities.append(Community(name=domain, platform=domain.split(".")[0].title(), mentions=count))
        else:
            publishers.append(Publisher(name=domain, type="Publisher", mentions=count))

    return Sources(citations=citations, publishers=publishers, communities=communities)


def _build_sitelist(publishers: list[Publisher]) -> list[SitelistEntry]:
    return [
        SitelistEntry(
            name=publisher.name,
            category="Publisher",
            score=min(100, publisher.mentions * 15),
            inventory=[],
            availability="Unknown",
            cpm="—",
            buyable=False,
        )
        for publisher in publishers
    ]


async def run_grounding(
    *, api_key: str, brand: str, industry: str, competitors: list[Competitor], market: str | None = None
) -> tuple[Sources, list[SitelistEntry]]:
    settings = get_settings()
    try:
        response = await call_web_search(
            api_key=api_key,
            # Only the raw citations get used downstream (_extract_raw_citations) —
            # the synthesized prose is never read, so the fast model is enough here.
            model=settings.anthropic_fast_model,
            system=_SYSTEM,
            user=_query(brand, industry, competitors, market),
            # Several search rounds plus extended thinking easily exceed the
            # 1536 default, truncating mid-synthesis (stop_reason max_tokens)
            # before all citations are even collected.
            max_tokens=4096,
        )
    except Exception:
        logger.exception("grounding call failed, returning empty sources")
        return Sources(citations=[], publishers=[], communities=[]), []

    raw_citations = _extract_raw_citations(response)
    raw_citations = await _filter_relevant(
        raw_citations,
        brand=brand,
        industry=industry,
        competitors=competitors,
        api_key=api_key,
        model=settings.anthropic_fast_model,
    )
    sources = _classify(raw_citations)
    sitelist = _build_sitelist(sources.publishers)
    return sources, sitelist
