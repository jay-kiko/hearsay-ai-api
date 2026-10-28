"""Source-quality helpers: which search hits count as an independent source,
which page to link, and whether the brand is actually evidenced there.

Kept free of model calls and of app imports so the rules are unit-testable on
their own (tests/test_source_quality.py) — grounding.py wires them to the
search response. The invariants these implement are in CLAUDE.md:

  * a source is a specific third-party page, never the brand's or a
    competitor's own site;
  * visibility is measured from page evidence, never inferred by a model;
  * a link must land on the page, not a dead URL or the site's homepage.
"""
from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
import ssl
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from urllib.parse import unquote, urlparse

# Shorter slugs ("dcl", "ana") collide with unrelated hostnames far too often
# to treat "slug appears in the domain" as "this is the brand's site".
_MIN_SLUG_LEN = 5

_VISIBLE_MIN_EVIDENCE = 2


def host_of(url: str) -> str:
    """Lower-cased hostname without port or a leading www., or '' if unparseable."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def entity_slugs(names: Iterable[str]) -> set[str]:
    """Slugs of the brand/competitor names whose presence in a hostname marks
    that site as owned by (or built around) the entity."""
    return {slug for slug in (_slug(n) for n in names if n) if len(slug) >= _MIN_SLUG_LEN}


def is_entity_domain(domain: str, slugs: set[str]) -> bool:
    """True when the hostname is built around a brand/competitor name — the
    official site, a regional or booking-portal variant of it, or a fan/affiliate
    site named after it. All of those are the entity talking about itself, which
    isn't independent opinion."""
    host = _slug(domain)
    return any(slug in host for slug in slugs)


# Translation/cache/archive hosts serve someone else's page under their own
# domain, so the host says nothing about whose content it is — and the proxied
# page is exactly what we mean to exclude when it is the brand's own site.
_PROXY_HOSTS = (
    "translate.google.com",
    "translate.goog",
    "webcache.googleusercontent.com",
    "web.archive.org",
    "archive.org",
    "archive.ph",
    "archive.today",
)


def is_proxy_host(host: str) -> bool:
    return any(host == p or host.endswith("." + p) for p in _PROXY_HOSTS)


def hosts_in(url: str) -> list[str]:
    """The URL's own host plus any host named in its query string or path
    (`?u=https://…`, `/web/2020/https://…`), so an entity site can't be reached
    through a wrapper."""
    hosts = [host_of(url)]
    for embedded in re.findall(r"https?://[^\s&#?]+", unquote(url)[8:]):
        hosts.append(host_of(embedded))
    return [h for h in hosts if h]


@dataclass
class SourcePage:
    url: str
    title: str
    # Passages the model quoted from this page when writing its summary
    # (web_search citations). Empty for pages that only appeared as search hits.
    cited_texts: list[str] = field(default_factory=list)
    cited: bool = False
    # Visible text of the page itself, when the link check could read it. Empty
    # for blocked or script-rendered pages, which then rest on the snippets.
    body: str = ""

    def slug_text(self) -> str:
        """The URL path as words ('disney-cruise-singapore-review' -> 'disney
        cruise singapore review'). Slugs name what the page is about, and are the
        only evidence we have for pages the search returned without a passage."""
        path = urlparse(self.url).path
        return re.sub(r"[-_/.]+", " ", path).strip()

    def evidence_texts(self) -> list[str]:
        return [t for t in [self.title, *self.cited_texts, self.slug_text()] if t]


_BODY_MENTION_CAP = 5  # one very long page shouldn't outweigh several independent ones


def page_mentions(page: SourcePage, count_mentions: Callable[[str], int]) -> int:
    """How strongly the brand is evidenced on this page: the number of snippets
    (title, each quoted passage, URL slug) that name it, plus its mentions in
    the page text itself (capped)."""
    snippets = sum(1 for text in page.evidence_texts() if count_mentions(text) > 0)
    body = min(count_mentions(page.body), _BODY_MENTION_CAP) if page.body else 0
    return snippets + body


def rank_pages(pages: list[SourcePage], count_mentions: Callable[[str], int]) -> list[SourcePage]:
    """Best page first: the one with brand evidence, then one the model actually
    cited over a bare search hit. Stable, so ties keep search order."""
    return sorted(pages, key=lambda p: (-page_mentions(p, count_mentions), not p.cited))


def visibility_from_evidence(evidence_count: int) -> str:
    """Maps measured evidence to the badge. 'Not visible' means no evidence in
    the titles/passages available — not proof the page never names the brand."""
    if evidence_count >= _VISIBLE_MIN_EVIDENCE:
        return "Visible"
    if evidence_count == 1:
        return "Weak visibility"
    return "Not visible"


def _excerpt(text: str, start: int, limit: int) -> str:
    text = " ".join(text.split())
    start = max(0, min(start, len(text)))
    begin = max(0, start - limit // 3)
    piece = text[begin : begin + limit].strip()
    return ("…" if begin else "") + piece + ("…" if begin + limit < len(text) else "")


def gap_for(visibility: str, competitor_mentions: dict[str, int]) -> str | None:
    """For a source the brand isn't visible on: is it already owned by a rival
    (they're named there) or still open? None when the brand is visible."""
    if visibility != "Not visible":
        return None
    return "competitor-owned" if competitor_mentions else "open"


def evidence_snippet(pages: list[SourcePage], count_mentions: Callable[[str], int], limit: int = 220) -> str:
    """The first title/passage naming the brand, else a stretch of page text
    around where the page names it. Empty when there is no evidence."""
    for page in pages:
        for text in page.evidence_texts():
            if count_mentions(text) > 0:
                clean = " ".join(text.split())
                return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"
    for page in pages:
        if page.body and count_mentions(page.body) > 0:
            # Find the first position the matcher fires at by growing a prefix.
            lo, hi = 0, len(page.body)
            while lo < hi:
                mid = (lo + hi) // 2
                if count_mentions(page.body[:mid]) > 0:
                    hi = mid
                else:
                    lo = mid + 1
            return _excerpt(page.body, lo - 40, limit)
    return ""


# ── Link health ──────────────────────────────────────────────────────

_DEAD_STATUSES = {404, 410}
_BROWSER_UA = "Mozilla/5.0 (compatible; HearsayLinkCheck/1.0)"


def is_public_http_url(url: str) -> bool:
    """Search results are public pages; refuse anything that would make the
    server request an internal address. Syntax only — names are resolved by
    resolves_public() before each request."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        port = parsed.port
    except ValueError:
        return False
    # Public pages live on the default ports; anything else is a way to poke
    # at non-web services on a public host.
    if port is not None and port not in (80, 443):
        return False
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        pass
    # Hosts with no dot, or made only of digits/hex ("2130706433", "0x7f000001",
    # "127.1"), are integer/short IP forms the resolver turns into internal
    # addresses; real public sites have a dotted name with a letter TLD.
    labels = host.split(".")
    return len(labels) >= 2 and labels[-1].isalpha()


async def resolves_public(host: str) -> bool:
    """True only if every address the host resolves to is a public one, so
    names like 169.254.169.254.nip.io can't reach metadata or private services.
    (Resolved again by the HTTP client when connecting; this closes the plain
    case, not a deliberate DNS-rebinding race.)"""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    addresses = {info[4][0] for info in infos}
    return bool(addresses) and all(ipaddress.ip_address(a.split("%")[0]).is_global for a in addresses)


def _path_depth(url: str) -> int:
    return len([seg for seg in urlparse(url).path.split("/") if seg])


def judge_response(requested_url: str, status: int, final_url: str) -> bool | None:
    """True = page is there, False = dead, None = can't tell.

    Bot-blocking statuses (403, 429, 999...) say nothing about whether the page
    exists, so they stay unknown rather than dead. A deep link that redirects to
    the site's homepage is dead for our purposes: the user would land on a
    front page, not the post we cited.
    """
    if status in _DEAD_STATUSES:
        return False
    if 200 <= status < 400:
        if _path_depth(requested_url) > 0 and _path_depth(final_url) == 0:
            return False
        return True
    return None


class _BlockedAddress(Exception):
    pass


def _is_tls_error(exc: BaseException) -> bool:
    cause = exc
    while cause is not None:
        if isinstance(cause, ssl.SSLError):
            return True
        cause = cause.__cause__ or cause.__context__
    return "ssl" in str(exc).lower() or "certificate" in str(exc).lower()


@dataclass
class PageCheck:
    alive: bool | None  # True = there, False = dead, None = can't tell
    text: str = ""  # visible text of the page when readable


_MAX_BODY_BYTES = 200_000
_SKIP_ELEMENTS = frozenset({"script", "style", "noscript", "template"})
_TAG_NAME_RE = re.compile(r"/?([a-zA-Z][a-zA-Z0-9-]*)")


def html_to_text(raw: str) -> str:
    """Visible text of an HTML page, in one linear pass.

    Pages come from search results, i.e. attacker-influenced. A regex
    (`<script.*?</script>`) and the stdlib HTMLParser are both quadratic on a
    page full of unclosed `<script` tags (confirmed: 64KB took 1.7s with the
    regex, and HTMLParser didn't finish 200KB in two minutes), which froze
    the event loop for every request. Every scan here starts where the last
    one ended, and an unterminated tag/comment/script ends the parse."""
    lowered = raw.lower()
    n = len(raw)
    out: list[str] = []
    i = 0
    while i < n:
        lt = raw.find("<", i)
        if lt == -1:
            out.append(raw[i:])
            break
        out.append(raw[i:lt])
        if raw.startswith("<!--", lt):
            end = raw.find("-->", lt + 4)
            i = n if end == -1 else end + 3
            continue
        gt = raw.find(">", lt + 1)
        if gt == -1:
            break
        i = gt + 1
        tag = _TAG_NAME_RE.match(raw, lt + 1)
        if tag and not tag.group(0).startswith("/") and tag.group(1).lower() in _SKIP_ELEMENTS:
            close = lowered.find("</" + tag.group(1).lower(), i)
            close_gt = raw.find(">", close) if close != -1 else -1
            i = n if close_gt == -1 else close_gt + 1
        out.append(" ")
    return " ".join(html.unescape("".join(out)).split())


async def check_pages(urls: list[str], *, timeout: float = 6.0, concurrency: int = 8) -> dict[str, PageCheck]:
    """One GET per URL, reading at most ~200KB of HTML: judges whether the page
    is alive and returns its visible text so the brand can be counted in the
    page itself. Never raises: a failure to check is alive=None, except DNS and
    connection errors, which are dead."""
    import httpx  # local: keeps the pure helpers above importable without it

    results: dict[str, PageCheck] = {}
    semaphore = asyncio.Semaphore(concurrency)

    async def _guard(request: "httpx.Request") -> None:
        # Runs on every redirect hop too, so a public page can't bounce us to
        # an internal address.
        url = str(request.url)
        if not is_public_http_url(url) or not await resolves_public(request.url.host):
            raise _BlockedAddress(url)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        max_redirects=3,
        headers={"User-Agent": _BROWSER_UA},
        event_hooks={"request": [_guard]},
    ) as client:

        async def _one(url: str) -> None:
            if not is_public_http_url(url):
                results[url] = PageCheck(alive=False)
                return
            async with semaphore:
                try:
                    async with client.stream("GET", url) as response:
                        alive = judge_response(url, response.status_code, str(response.url))
                        text = ""
                        if alive and "html" in response.headers.get("content-type", "").lower():
                            raw = bytearray()
                            async for chunk in response.aiter_bytes():
                                raw.extend(chunk)
                                if len(raw) >= _MAX_BODY_BYTES:
                                    break
                            decoded = bytes(raw).decode(response.encoding or "utf-8", errors="replace")
                            # Off the event loop: CPU work on a page we don't control.
                            text = await asyncio.to_thread(html_to_text, decoded)
                        results[url] = PageCheck(alive=alive, text=text)
                except _BlockedAddress:
                    # The page (or a redirect) points at an internal address:
                    # never a link we show.
                    results[url] = PageCheck(alive=False)
                except httpx.ConnectError as exc:
                    # TLS handshake failures arrive as ConnectError too; a site
                    # with a certificate quirk still exists, so that's unknown.
                    results[url] = PageCheck(alive=None if _is_tls_error(exc) else False)
                except Exception:  # noqa: BLE001 - timeouts and other quirks: unknown, not dead
                    results[url] = PageCheck(alive=None)

        await asyncio.gather(*(_one(u) for u in set(urls)))
    return results
