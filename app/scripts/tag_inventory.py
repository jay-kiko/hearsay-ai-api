"""One-time (re-runnable) tagging of app/data/sg_inventory.json with a model.

    ANTHROPIC_API_KEY=... python -m app.scripts.tag_inventory [--retag]

Run offline after app.scripts.build_inventory, then commit the JSON. Tags are
what the request-time shortlist reads, so the model never has to look at every
site per analysis. Uses the quality model: it runs once per site, and a wrong
`editorial` flag silently hides (or admits) a source for every future analysis.
Only untagged sites are tagged unless --retag is given.

Mobile apps (sg_apps.json) can't be sources themselves, but a publisher app
usually has a website that can: a model names each app's publisher and main
website, the website is verified to resolve, and then either the app's reach is
merged into the existing site entry or the site is added as a new (tagged) entry
marked via="app". Games and tools with no editorial site stay in sg_apps.json
only.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from app.config import get_settings
from app.services.anthropic_client import call_structured
from app.services import source_quality as sq
from app.services.inventory import APPS_PATH, DATA_PATH

CATEGORIES = [
    "general news", "business & finance news", "technology", "travel", "food & dining", "lifestyle & fashion",
    "parenting & family", "health & wellness", "sports", "gaming", "entertainment & celebrity",
    "education & reference", "utilities & tools", "shopping & classifieds", "automotive", "property",
    "science & culture", "regional non-English news", "other",
]
SITE_TYPES = ["news publisher", "blog", "forum or community", "review or comparison site", "magazine",
              "reference", "tool or calculator", "game", "video or audio", "marketplace", "other"]
REGIONS = ["singapore", "regional (asia)", "global"]

_TOOL = "tag_sites"
_SCHEMA = {
    "type": "object",
    "properties": {
        "sites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "siteType": {"type": "string", "enum": SITE_TYPES},
                    "region": {"type": "string", "enum": REGIONS, "description": "Whose audience the site is mainly written for."},
                    "editorial": {
                        "type": "boolean",
                        "description": (
                            "True only if the site publishes articles, reviews, guides or discussion in which "
                            "people could plausibly write about or compare brands, products or services. False "
                            "for games, calculators and tools, weather/radio/streaming players, wallpapers, "
                            "lyrics, and pure data/reference lookups."
                        ),
                    },
                    "niches": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-4 short topics it covers, e.g. 'family travel', 'personal finance', 'gadgets'.",
                    },
                },
                "required": ["domain", "category", "siteType", "region", "editorial", "niches"],
            },
        }
    },
    "required": ["sites"],
}
_SYSTEM = (
    "You label websites from a Singapore ad-inventory list so they can later be matched to a brand's "
    "industry. Judge each from its domain and title (and your own knowledge of the site). Be strict with "
    "`editorial`: it decides whether the site can ever be used as a place where people discuss brands. "
    "Return one entry per input domain, using the exact domain string given."
)
_BATCH = 40


def _valid(entry: object, domains: set[str]) -> bool:
    return (
        isinstance(entry, dict)
        and entry.get("domain") in domains
        and entry.get("category") in CATEGORIES
        and entry.get("siteType") in SITE_TYPES
        and entry.get("region") in REGIONS
        and isinstance(entry.get("editorial"), bool)
        and isinstance(entry.get("niches"), list)
    )


async def _tag_batch(api_key: str, model: str, batch: list[dict]) -> list[dict]:
    lines = "\n".join(f"{s['domain']} | {s['name'][:90]}" for s in batch)
    result = await call_structured(
        api_key=api_key, model=model, system=_SYSTEM, user=f"Sites (domain | title):\n{lines}",
        tool_name=_TOOL, tool_description="Return the labels for every site.", input_schema=_SCHEMA, max_tokens=8000,
    )
    domains = {s["domain"] for s in batch}
    return [e for e in result.get("sites", []) if _valid(e, domains)]


APP_CATEGORIES = ["game", "news", "sports", "utilities", "messaging & social", "shopping & classifieds",
                  "health & fitness", "video, audio & entertainment", "finance", "travel & transport",
                  "education & reference", "food", "other"]
_APP_TOOL = "tag_apps"
_APP_SCHEMA = {
    "type": "object",
    "properties": {
        "apps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "publisher": {"type": "string", "description": "The company or brand behind the app."},
                    "category": {"type": "string", "enum": APP_CATEGORIES},
                    "webDomain": {
                        "type": "string",
                        "description": (
                            "The publisher's main public website domain (e.g. 'myfitnesspal.com'), only if you "
                            "are confident it is real and is the same brand. Empty string if unsure, or if the "
                            "app is a game or tool with no real web presence. Never guess."
                        ),
                    },
                },
                "required": ["id", "publisher", "category", "webDomain"],
            },
        }
    },
    "required": ["apps"],
}
_APP_SYSTEM = (
    "You identify mobile apps from a Singapore ad-inventory list: who publishes each, what kind of app it is, "
    "and the publisher's main website if there is one. Return one entry per input id using the exact id given."
)


async def _tag_app_batch(api_key: str, model: str, batch: list[dict]) -> list[dict]:
    lines = "\n".join(f"{a['id']} | {a['name'][:80]} | {a['platform']}" for a in batch)
    result = await call_structured(
        api_key=api_key, model=model, system=_APP_SYSTEM, user=f"Apps (id | store name | platform):\n{lines}",
        tool_name=_APP_TOOL, tool_description="Return the labels for every app.", input_schema=_APP_SCHEMA, max_tokens=6000,
    )
    ids = {a["id"] for a in batch}
    return [
        e for e in result.get("apps", [])
        if isinstance(e, dict) and e.get("id") in ids and isinstance(e.get("publisher"), str)
        and e.get("category") in APP_CATEGORIES and isinstance(e.get("webDomain"), str)
    ]


def _combine(apps: list[dict]) -> dict:
    """One publisher's apps -> a single in-app creative summary."""
    creative: dict[str, dict] = {}
    for app in apps:
        for kind, c in app["creative"].items():
            cur = creative.setdefault(kind, {"bidRequests": 0, "cpmMin": c["cpmMin"], "cpmMax": c["cpmMax"]})
            cur["bidRequests"] += c["bidRequests"]
            cur["cpmMin"] = min(cur["cpmMin"], c["cpmMin"])
            cur["cpmMax"] = max(cur["cpmMax"], c["cpmMax"])
    return {
        "bidRequests": sum(c["bidRequests"] for c in creative.values()),
        "creative": creative,
        "names": sorted({a["name"] for a in apps}),
    }


async def _tag_apps(api_key: str, model: str, sites: list[dict], retag: bool) -> None:
    apps = json.loads(APPS_PATH.read_text(encoding="utf-8"))
    todo = [a for a in apps if retag or "publisher" not in a]
    print(f"tagging {len(todo)} of {len(apps)} apps")
    semaphore = asyncio.Semaphore(3)

    async def run(batch):
        async with semaphore:
            return await _tag_app_batch(api_key, model, batch)

    by_id = {a["id"]: a for a in apps}
    batches = [todo[i : i + _BATCH] for i in range(0, len(todo), _BATCH)]
    for result in await asyncio.gather(*(run(b) for b in batches), return_exceptions=True):
        if isinstance(result, Exception):
            print("app batch failed:", repr(result)[:200])
            continue
        for e in result:
            by_id[e["id"]].update(publisher=e["publisher"], category=e["category"], webDomain=sq.host_of(f"https://{e['webDomain']}") if e["webDomain"] else "")

    # The model can name a domain that doesn't exist; keep only ones that resolve.
    domains = sorted({a["webDomain"] for a in apps if a.get("webDomain")} - {s["domain"] for s in sites})
    checks = await sq.check_pages([f"https://{d}/" for d in domains]) if domains else {}
    verified = {d for d in domains if checks.get(f"https://{d}/", sq.PageCheck(alive=False)).alive is not False}
    known = {s["domain"] for s in sites}
    for a in apps:
        a["webVerified"] = bool(a.get("webDomain")) and (a["webDomain"] in known or a["webDomain"] in verified)
    APPS_PATH.write_text(json.dumps(apps, indent=1, ensure_ascii=False), encoding="utf-8")

    # Rebuild the app-derived parts of the site list from scratch (idempotent),
    # reusing tags already paid for on previously derived entries.
    prior_tags = {s["domain"]: s for s in sites if s.get("via") == "app"}
    sites[:] = [s for s in sites if s.get("via") != "app"]
    for s in sites:
        s.pop("app", None)
    by_domain = {s["domain"]: s for s in sites}

    groups: dict[str, list[dict]] = {}
    for a in apps:
        if a["webVerified"]:
            groups.setdefault(a["webDomain"], []).append(a)

    new_entries = []
    for domain, members in groups.items():
        summary = _combine(members)
        if domain in by_domain:
            by_domain[domain]["app"] = summary
        else:
            entry = {"domain": domain, "name": members[0]["publisher"], "creative": {}, "bidRequests": 0,
                     "via": "app", "app": summary}
            for key in ("category", "siteType", "editorial", "region", "niches"):
                if key in prior_tags.get(domain, {}):
                    entry[key] = prior_tags[domain][key]
            new_entries.append(entry)

    untagged = [e for e in new_entries if "editorial" not in e]
    print(f"{len(groups)} publisher websites from apps: {len(groups) - len(new_entries)} merged into existing sites, "
          f"{len(new_entries)} new ({len(untagged)} to tag)")
    if untagged:
        by_new = {e["domain"]: e for e in new_entries}
        for tagged in await asyncio.gather(*(run_web(api_key, model, untagged[i : i + _BATCH]) for i in range(0, len(untagged), _BATCH))):
            for t in tagged:
                by_new[t["domain"]].update(
                    category=t["category"], siteType=t["siteType"], region=t["region"], editorial=t["editorial"],
                    niches=[n for n in t["niches"] if isinstance(n, str)][:4],
                )
    sites.extend(new_entries)
    unresolved_apps = sum(1 for a in apps if not a["webVerified"])
    print(f"{unresolved_apps} apps have no verified website (games/tools); kept in sg_apps.json only")


async def run_web(api_key: str, model: str, batch: list[dict]) -> list[dict]:
    return await _tag_batch(api_key, model, batch)


async def main(retag: bool) -> None:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    model = get_settings().anthropic_quality_model
    sites = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    todo = [s for s in sites if retag or "editorial" not in s]
    print(f"tagging {len(todo)} of {len(sites)} sites with {model}")

    semaphore = asyncio.Semaphore(3)

    async def run(batch):
        async with semaphore:
            return await _tag_batch(api_key, model, batch)

    batches = [todo[i : i + _BATCH] for i in range(0, len(todo), _BATCH)]
    by_domain = {s["domain"]: s for s in sites}
    for result in await asyncio.gather(*(run(b) for b in batches), return_exceptions=True):
        if isinstance(result, Exception):
            print("batch failed:", repr(result)[:200])
            continue
        for e in result:
            site = by_domain[e["domain"]]
            site.update(
                category=e["category"], siteType=e["siteType"], region=e["region"], editorial=e["editorial"],
                niches=[n for n in e["niches"] if isinstance(n, str)][:4],
            )
    if APPS_PATH.exists():
        await _tag_apps(api_key, model, sites, retag)
    DATA_PATH.write_text(json.dumps(sites, indent=1, ensure_ascii=False), encoding="utf-8")
    missing = [s["domain"] for s in sites if "editorial" not in s]
    print(f"done; {len(missing)} still untagged" + (f": {missing[:10]}" if missing else ""))


if __name__ == "__main__":
    asyncio.run(main("--retag" in sys.argv))
