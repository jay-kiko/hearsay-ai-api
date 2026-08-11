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
from app.models import Citation, Community, Publisher, SitelistEntry, Sources
from app.services.anthropic_client import call_web_search

logger = logging.getLogger("hearsay.grounding")

_SYSTEM = (
    "Use the web_search tool to research how the given brand and its competitors "
    "are actually discussed in their category right now. Search for comparison "
    "articles, review sites, and community discussion (Reddit, forums, Hacker "
    "News). Then briefly summarize what you found in 2-3 sentences."
)

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


def _query(brand: str, industry: str, competitors: list[str]) -> str:
    competitor_list = ", ".join(competitors) if competitors else "its main competitors"
    return (
        f"How is '{brand}' discussed and reviewed within the '{industry}' category, "
        f"compared to {competitor_list}? Find real articles, review sites, and "
        f"community threads."
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

        # Inline citations attached to generated text.
        citations = getattr(block, "citations", None) or []
        for citation in citations:
            url = getattr(citation, "url", None)
            title = getattr(citation, "title", None)
            if url:
                found.append({"url": url, "title": title or url})

        # Raw web_search tool results.
        if block_type == "web_search_tool_result":
            content = getattr(block, "content", None) or []
            for result in content:
                url = getattr(result, "url", None)
                title = getattr(result, "title", None)
                if url:
                    found.append({"url": url, "title": title or url})

    return found


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
    *, api_key: str, brand: str, industry: str, competitors: list[str]
) -> tuple[Sources, list[SitelistEntry]]:
    settings = get_settings()
    try:
        response = await call_web_search(
            api_key=api_key,
            model=settings.anthropic_model,
            system=_SYSTEM,
            user=_query(brand, industry, competitors),
        )
    except Exception:
        logger.exception("grounding call failed, returning empty sources")
        return Sources(citations=[], publishers=[], communities=[]), []

    raw_citations = _extract_raw_citations(response)
    sources = _classify(raw_citations)
    sitelist = _build_sitelist(sources.publishers)
    return sources, sitelist
