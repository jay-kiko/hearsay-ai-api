"""Singapore programmatic ad inventory (a DSP "traffic discovery" export).

Labels sources, never chooses them: when a source found by the buyer-question
search (grounding.py) is in this inventory, the sitelist shows its real CPM /
reach / creative types and marks it buyable (`sitelist_fields`). The model tags
(category, editorial, niches) are kept for the actions page.

The data file (app/data/sg_inventory.json) is built offline from the CSV by
app/scripts/build_inventory.py and tagged by app/scripts/tag_inventory.py. Parsing is pure and has no model
or app imports so it can be unit-tested directly.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("hearsay.inventory")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sg_inventory.json"
APPS_PATH = DATA_PATH.with_name("sg_apps.json")

_DOMAIN_RE = re.compile(r"[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}")
_TRAILING_PAREN_RE = re.compile(r"\(([^()]+)\)\s*$")
_CPM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")


# Android package names (com.livescore, tunein.player, com.king.candycrushsaga)
# are shaped exactly like domains, so shape alone can't tell them apart. A real
# site ends in a country code or one of these generic TLDs; a package ends in
# an app name. Packages also lead with a reverse-DNS prefix, which no real
# multi-label hostname does.
_GENERIC_TLDS = {
    "com", "net", "org", "info", "io", "ai", "app", "dev", "tv", "me", "co", "news", "xyz", "online", "site",
    "blog", "tech", "store", "shop", "club", "live", "media", "world", "today", "life", "pro", "biz", "london",
    "edu", "gov", "asia", "wiki", "page", "cloud", "design",
}
_CC_TLDS = set(
    "ac ad ae af ag ai al am ao aq ar as at au aw ax az ba bb bd be bf bg bh bi bj bm bn bo br bs bt bw by bz "
    "ca cc cd cf cg ch ci ck cl cm cn co cr cu cv cw cx cy cz de dj dk dm do dz ec ee eg er es et eu fi fj fk "
    "fm fo fr ga gd ge gf gg gh gi gl gm gn gp gq gr gs gt gu gw gy hk hm hn hr ht hu id ie il im in io iq ir "
    "is it je jm jo jp ke kg kh ki km kn kp kr kw ky kz la lb lc li lk lr ls lt lu lv ly ma mc md me mg mh mk "
    "ml mm mn mo mp mq mr ms mt mu mv mw mx my mz na nc ne nf ng ni nl no np nr nu nz om pa pe pf pg ph pk pl "
    "pm pn pr ps pt pw py qa re ro rs ru rw sa sb sc sd se sg sh si sk sl sm sn so sr ss st su sv sx sy sz tc "
    "td tf tg th tj tk tl tm tn to tr tt tv tw tz ua ug uk us uy uz va vc ve vg vi vn vu wf ws ye yt za zm zw".split()
)
_PACKAGE_PREFIXES = {"com", "org", "net", "edu", "gov"}


def _clean_domain(token: str) -> str | None:
    token = token.strip().lower()
    token = token[4:] if token.startswith("www.") else token
    if not _DOMAIN_RE.fullmatch(token):
        return None
    labels = token.split(".")
    if labels[-1] not in _CC_TLDS and labels[-1] not in _GENERIC_TLDS:
        return None
    if len(labels) >= 3 and labels[0] in _PACKAGE_PREFIXES:
        return None
    return token


@dataclass
class ParseResult:
    sites: list[dict] = field(default_factory=list)
    # One entry per distinct app (iOS id / Android package), creative types merged.
    # Apps aren't pages a source can link to, but their publishers often have a
    # website that is — see app.scripts.tag_inventory.
    apps: list[dict] = field(default_factory=list)
    # Rows whose site name has no usable domain (e.g. 'merriam webster.com'),
    # kept whole so they can be repaired and added back (see repair_unresolved).
    unresolved: list[dict] = field(default_factory=list)
    bad_rows: int = 0


def _merge_creative(entry: dict, creative: str, bids: int, cpm_min: float, cpm_max: float) -> None:
    prior = entry["creative"].get(creative)
    if prior:  # same item+type listed twice: keep the larger volume, widen the CPM range
        prior["bidRequests"] = max(prior["bidRequests"], bids)
        prior["cpmMin"] = min(prior["cpmMin"], cpm_min)
        prior["cpmMax"] = max(prior["cpmMax"], cpm_max)
    else:
        entry["creative"][creative] = {"bidRequests": bids, "cpmMin": cpm_min, "cpmMax": cpm_max}
    entry["bidRequests"] = sum(c["bidRequests"] for c in entry["creative"].values())


def _add_site(merged: dict[str, dict], domain: str, title: str, creative: str, bids: int, lo: float, hi: float) -> None:
    entry = merged.setdefault(domain, {"domain": domain, "name": title or domain, "creative": {}, "bidRequests": 0})
    _merge_creative(entry, creative, bids, lo, hi)


def parse_csv(path: str | Path) -> ParseResult:
    """Cleans the export into one entry per web domain, plus the apps.

    A domain appears once per creative type (Banner/Video/Native), so those rows
    merge into a single entry with per-type bid volume and CPM range.
    """
    result = ParseResult()
    merged: dict[str, dict] = {}
    apps: dict[str, dict] = {}

    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw_name = (row.get("App/Site name") or "").strip()
            creative = (row.get("Creative Type") or "").strip()
            cpm = _CPM_RE.match(row.get("Estimated CPM") or "")
            try:
                bids = int((row.get("Bid requests") or "").replace(",", ""))
            except ValueError:
                bids = -1
            if not raw_name or not creative or not cpm or bids < 0:
                result.bad_rows += 1
                continue
            lo, hi = float(cpm.group(1)), float(cpm.group(2))

            paren = _TRAILING_PAREN_RE.search(raw_name)
            token = (paren.group(1) if paren else raw_name).strip()
            title = (raw_name[: paren.start()].strip() if paren else "")
            domain = _clean_domain(token)
            if domain is not None:
                _add_site(merged, domain, title, creative, bids, lo, hi)
            elif re.fullmatch(r"\d+", token) or re.fullmatch(r"[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)+", token):
                app = apps.setdefault(token, {
                    "id": token, "platform": "ios" if token.isdigit() else "android",
                    "name": title or token, "creative": {}, "bidRequests": 0,
                })
                _merge_creative(app, creative, bids, lo, hi)
            else:
                result.unresolved.append(
                    {"name": raw_name, "creative": creative, "bidRequests": bids, "cpmMin": lo, "cpmMax": hi}
                )

    result.sites = sorted(merged.values(), key=lambda e: -e["bidRequests"])
    result.apps = sorted(apps.values(), key=lambda e: -e["bidRequests"])
    return result


def repair_candidates(name: str) -> list[str]:
    """Plausible real domains for a mangled site name ('merriam webster.com' ->
    merriam-webster.com / merriamwebster.com). Only candidates — the caller must
    verify one resolves before using it."""
    base = name.strip().lower()
    out = []
    for candidate in (base.replace(" ", "-"), base.replace(" ", "")):
        domain = _clean_domain(candidate)
        if domain and domain not in out:
            out.append(domain)
    return out


def add_repaired(result: ParseResult, repairs: dict[str, str]) -> list[dict]:
    """Moves unresolved rows whose name has a verified repair (name -> domain)
    into result.sites. Returns the rows that are still unresolved."""
    merged = {s["domain"]: s for s in result.sites}
    still = []
    for row in result.unresolved:
        domain = repairs.get(row["name"])
        if domain:
            _add_site(merged, domain, domain, row["creative"], row["bidRequests"], row["cpmMin"], row["cpmMax"])
        else:
            still.append(row)
    result.sites = sorted(merged.values(), key=lambda e: -e["bidRequests"])
    result.unresolved = still
    return still


# ── Lookup ───────────────────────────────────────────────────────────

@lru_cache
def load_sites() -> dict[str, dict]:
    try:
        sites = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("inventory file %s missing; Singapore inventory features disabled", DATA_PATH)
        return {}
    return {s["domain"]: s for s in sites}


# Platforms that host other people's sites on subdomains: buying tumblr.com
# doesn't target someone.tumblr.com, so only an exact match counts for these.
_HOSTING_PLATFORMS = {
    "tumblr.com", "blogspot.com", "wordpress.com", "substack.com", "medium.com", "github.io",
    "wixsite.com", "weebly.com", "notion.site", "squarespace.com", "webflow.io", "carrd.co",
}


def lookup(domain: str, sites: dict[str, dict] | None = None) -> dict | None:
    """Inventory entry for a domain or any of its subdomains
    (sg.news.yahoo.com -> yahoo.com), except user sites on hosting platforms."""
    sites = load_sites() if sites is None else sites
    domain = domain.lower().removeprefix("www.")
    labels = domain.split(".")
    for i in range(len(labels) - 1):
        parent = ".".join(labels[i:])
        if i > 0 and parent in _HOSTING_PLATFORMS:
            return None
        hit = sites.get(parent)
        if hit:
            return hit
    return None


def applies_to(market: str | None) -> bool:
    """The inventory is Singapore-only; elsewhere the sitelist keeps placeholders."""
    return bool(market and re.search(r"\b(singapore|sg)\b", market.lower()))


def sitelist_fields(domain: str) -> dict | None:
    """CPM / availability / creative types for a domain, or None if not in the
    inventory. Where the publisher also has apps in the inventory, their reach
    and CPM are included, labelled in-app."""
    site = lookup(domain)
    if not site:
        return None
    web = site.get("creative", {})
    app = (site.get("app") or {}).get("creative", {})
    cpms = [c["cpmMin"] for c in (*web.values(), *app.values())]
    cpms_hi = [c["cpmMax"] for c in (*web.values(), *app.values())]
    if not cpms:
        return None
    parts = []
    if web:
        parts.append(f"{sum(c['bidRequests'] for c in web.values()):,} web bid requests")
    if app:
        parts.append(f"{sum(c['bidRequests'] for c in app.values()):,} in-app bid requests")
    return {
        "inventory": [*web, *(f"In-app {t}" for t in app)],
        "cpm": f"{min(cpms):.2f} - {max(cpms_hi):.2f}",
        "availability": " + ".join(parts),
        "buyable": True,
    }
