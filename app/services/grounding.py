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
from dataclasses import dataclass
from urllib.parse import urlparse

from app.config import get_settings
from app.models import Citation, Community, Competitor, PersonaIn, Publisher, SitelistEntry, SourceIntel, Sources
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


@dataclass
class _ClassifiedDomains:
    citations: list[Citation]
    publishers: list[Publisher]
    communities: list[Community]
    domain_titles: dict[str, str]
    domain_urls: dict[str, str]


def _classify(raw_citations: list[dict[str, str]]) -> _ClassifiedDomains:
    domain_counts: dict[str, int] = defaultdict(int)
    domain_titles: dict[str, str] = {}
    domain_urls: dict[str, str] = {}

    for item in raw_citations:
        try:
            domain = urlparse(item["url"]).netloc.replace("www.", "")
        except (KeyError, ValueError):
            continue
        if not domain:
            continue
        domain_counts[domain] += 1
        domain_titles.setdefault(domain, item.get("title", domain))
        domain_urls.setdefault(domain, item["url"])

    citations: list[Citation] = []
    publishers: list[Publisher] = []
    communities: list[Community] = []

    for domain, count in sorted(domain_counts.items(), key=lambda kv: -kv[1]):
        title = domain_titles[domain]
        citations.append(Citation(domain=domain, title=title, count=count))
        if _is_community(domain):
            communities.append(Community(name=domain, platform=domain.split(".")[0].title(), mentions=count))
        else:
            # "Publisher" is a placeholder here — _enrich_sources classifies
            # the real type (Review platform / Tech news / Vendor blog / ...)
            # afterward and patches it in below; this is only the fallback
            # if that call fails or skips a domain.
            publishers.append(Publisher(name=domain, type="Publisher", mentions=count))

    return _ClassifiedDomains(
        citations=citations,
        publishers=publishers,
        communities=communities,
        domain_titles=domain_titles,
        domain_urls=domain_urls,
    )


_ENRICH_TOOL_NAME = "enrich_source_intel"
_ENRICH_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "type": {
                        "type": "string",
                        "description": "e.g. 'Review platform', 'Tech news', 'Vendor blog', 'Community', 'Publisher'.",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-4 short search phrases this source plausibly ranks for in this category.",
                    },
                    "competitors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Which of the given competitor names this source plausibly discusses, if any.",
                    },
                    "personas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Which of the given persona titles would find this source's topic relevant, if any.",
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["Visible", "Weak visibility", "Not visible"],
                        "description": "How prominently the brand itself appears to be covered here, per the research synthesis.",
                    },
                },
                "required": ["domain", "type", "keywords", "competitors", "personas", "visibility"],
            },
        }
    },
    "required": ["sources"],
}

_ENRICH_SYSTEM = (
    "You are given a list of real domains found while researching how a brand is discussed in its category, "
    "the titles found for each, and a short research synthesis. For each domain, infer: its type (a review "
    "platform, tech news outlet, vendor blog, community, etc.), a few keywords it plausibly ranks for in this "
    "category, which of the given competitors it plausibly discusses, which of the given personas would find "
    "its topic relevant, and how visible the brand itself appears to be there. This is a best-effort read from "
    "limited evidence, not certain fact — infer plausibly rather than leaving fields empty, but never invent a "
    "competitor or persona name that wasn't given to you."
)


async def _enrich_sources(
    *,
    domains: dict[str, tuple[str, int]],  # domain -> (title, count)
    synthesis: str,
    brand: str,
    competitors: list[Competitor],
    personas: list[PersonaIn],
    api_key: str,
    model: str,
) -> dict[str, dict]:
    """Best-effort qualitative read per domain — grounding runs once per job
    with its own standalone web search, entirely separate from the
    per-persona pipeline, so there's no real measured link between a
    specific persona's answer and a specific source. Fails open (empty dict)
    on any error, same posture as _filter_relevant."""
    if not domains:
        return {}

    domain_list = "\n".join(f"- {domain}: {title} ({count} results)" for domain, (title, count) in domains.items())
    competitor_names = [c.name for c in competitors]
    persona_titles = [p.title for p in personas]

    try:
        result = await call_structured(
            api_key=api_key,
            model=model,
            system=_ENRICH_SYSTEM,
            user=(
                f"Brand: {brand}\n"
                f"Competitors: {', '.join(competitor_names) if competitor_names else 'none specified'}\n"
                f"Personas: {', '.join(persona_titles) if persona_titles else 'none specified'}\n\n"
                f"Research synthesis:\n{synthesis or '(none captured)'}\n\n"
                f"Domains found:\n{domain_list}"
            ),
            tool_name=_ENRICH_TOOL_NAME,
            tool_description="Return the enriched intel for every domain.",
            input_schema=_ENRICH_INPUT_SCHEMA,
            # Confirmed live: with a realistic ~24-domain result set, 2048
            # wasn't enough to finish even one entry before hitting
            # stop_reason="max_tokens" — the truncated tool-use JSON came
            # back as an empty {} rather than a partial array, so this isn't
            # a "fewer results than ideal" degradation, it's a hard zero.
            # Same failure mode grounding's own web_search call already had
            # to fix by raising its max_tokens well past the 1536 default.
            max_tokens=4096,
        )
    except Exception:
        logger.exception("source enrichment failed, falling back to placeholder source intel")
        return {}

    return {
        entry["domain"]: entry
        for entry in result.get("sources", [])
        if isinstance(entry, dict) and isinstance(entry.get("domain"), str)
    }


def _build_sitelist(publishers: list[Publisher]) -> list[SitelistEntry]:
    return [
        SitelistEntry(
            name=publisher.name,
            category=publisher.type,
            score=min(100, publisher.mentions * 15),
            inventory=[],
            availability="Unknown",
            cpm="—",
            buyable=False,
        )
        for publisher in publishers
    ]


def _extract_synthesis_text(response) -> str:
    return "".join(block.text for block in getattr(response, "content", []) or [] if getattr(block, "type", None) == "text").strip()


def _influence_label(count: int, category: str) -> str:
    tier = "High" if count >= 5 else "Medium" if count >= 3 else "Low"
    noun = "AI influence" if category == "REVIEW" else "community influence" if category == "COMMUNITY" else "authority"
    return f"{tier} {noun}"


async def run_grounding(
    *,
    api_key: str,
    brand: str,
    industry: str,
    competitors: list[Competitor],
    personas: list[PersonaIn] | None = None,
    market: str | None = None,
) -> tuple[Sources, list[SitelistEntry]]:
    settings = get_settings()
    empty_sources = Sources(citations=[], publishers=[], communities=[], source_intel=[])
    try:
        response = await call_web_search(
            api_key=api_key,
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
        return empty_sources, []

    raw_citations = _extract_raw_citations(response)
    raw_citations = await _filter_relevant(
        raw_citations,
        brand=brand,
        industry=industry,
        competitors=competitors,
        api_key=api_key,
        model=settings.anthropic_fast_model,
    )
    classified = _classify(raw_citations)

    # The model's own written summary of what it found — previously
    # generated and discarded; now the main evidence _enrich_sources reasons
    # over, alongside each domain's title.
    synthesis = _extract_synthesis_text(response)
    # Capped to the most-cited domains regardless of how many grounding
    # turns up — an uncapped list is exactly what caused the real truncation
    # above (24 domains blew through 2048 tokens before finishing even one
    # entry); the long tail of single-mention domains is also the least
    # reliable signal to enrich anyway, so they keep their placeholder
    # defaults rather than pushing out a token budget that has to stay
    # predictable. Same top-N-by-count approach insights_gen.py already uses.
    _MAX_ENRICHED_DOMAINS = 15
    domain_counts = {c.domain: c.count for c in classified.citations}
    top_domains = sorted(classified.domain_titles.items(), key=lambda kv: -domain_counts.get(kv[0], 0))
    enrichment = await _enrich_sources(
        domains={
            domain: (title, domain_counts.get(domain, 0))
            for domain, title in top_domains[:_MAX_ENRICHED_DOMAINS]
        },
        synthesis=synthesis,
        brand=brand,
        competitors=competitors,
        personas=personas or [],
        api_key=api_key,
        model=settings.anthropic_fast_model,
    )

    # Enrichment output isn't a hard type guarantee (same defensive posture
    # as detect.py/category_gen.py) — a malformed entry here must degrade to
    # placeholders, not fail a job whose actual persona results already
    # succeeded, so every field pulled from it below is defensively coerced.
    _VALID_VISIBILITY = {"Visible", "Weak visibility", "Not visible"}

    def _clean_type(raw: object, fallback: str) -> str:
        return raw if isinstance(raw, str) and raw.strip() else fallback

    def _clean_visibility(raw: object) -> str:
        return raw if raw in _VALID_VISIBILITY else "Not visible"

    def _clean_list(raw: object) -> list[str]:
        # Forced tool calls have returned array-typed fields as comma-joined
        # strings before (see detect.py/category_gen.py) — iterating a bare
        # string here would silently produce one entry per character.
        if not isinstance(raw, list):
            return []
        return [v for v in raw if isinstance(v, str)]

    publishers = [
        Publisher(name=p.name, type=_clean_type(enrichment.get(p.name, {}).get("type"), p.type), mentions=p.mentions)
        for p in classified.publishers
    ]
    sources = Sources(
        citations=classified.citations,
        publishers=publishers,
        communities=classified.communities,
    )

    source_intel: list[SourceIntel] = []
    publisher_types = {p.name: p.type for p in publishers}
    for citation in classified.citations:
        domain = citation.domain
        is_community = _is_community(domain)
        entry = enrichment.get(domain, {}) if isinstance(enrichment.get(domain), dict) else {}
        publisher_type = publisher_types.get(domain, "Publisher")
        category = "COMMUNITY" if is_community else ("REVIEW" if "review" in publisher_type.lower() else "PUBLISHER")
        source_intel.append(
            SourceIntel(
                domain=domain,
                url=classified.domain_urls.get(domain, f"https://{domain}"),
                category=category,
                influence=_influence_label(citation.count, category),
                cited=citation.count,
                keywords=_clean_list(entry.get("keywords")),
                personas=_clean_list(entry.get("personas")),
                competitors=_clean_list(entry.get("competitors")),
                visibility=_clean_visibility(entry.get("visibility")),
            )
        )
    sources.source_intel = source_intel

    sitelist = _build_sitelist(sources.publishers)
    return sources, sitelist
