"""Builds app/data/sg_inventory.json (web sites) and sg_apps.json (mobile apps)
from a DSP traffic-discovery CSV.

    python -m app.scripts.build_inventory /path/to/export.csv

Overwrites both files but keeps existing model tags for items still present, then
run app.scripts.tag_inventory to tag anything new. Rows whose site name is
mangled ('merriam webster.com') are repaired only when a hyphenated or joined
form actually resolves; the rest are listed for you to fix in the CSV.
"""
from __future__ import annotations

import asyncio
import json
import sys

from app.services import source_quality as sq
from app.services.inventory import (
    APPS_PATH,
    DATA_PATH,
    add_repaired,
    parse_csv,
    repair_candidates,
)

_TAG_KEYS_SITE = ("category", "siteType", "editorial", "region", "niches", "via", "app")
_TAG_KEYS_APP = ("publisher", "category", "webDomain", "webVerified")


async def verified_repairs(unresolved: list[dict]) -> dict[str, str]:
    """name -> domain for rows where a candidate repair is a site that exists."""
    candidates = {row["name"]: repair_candidates(row["name"]) for row in unresolved}
    urls = [f"https://{d}/" for ds in candidates.values() for d in ds]
    checks = await sq.check_pages(urls)
    repairs = {}
    for name, domains in candidates.items():
        for domain in domains:
            # None = the site answered but blocks bots (the domain exists); False =
            # dead / doesn't resolve. Candidates are mechanical rewrites, not model
            # output, so "exists" is enough to trust one.
            if checks.get(f"https://{domain}/", sq.PageCheck(alive=False)).alive is not False:
                repairs[name] = domain
                break
    return repairs


def _load(path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def main(csv_path: str) -> None:
    prev_sites = {s["domain"]: s for s in _load(DATA_PATH)}
    prev_apps = {a["id"]: a for a in _load(APPS_PATH)}

    parsed = parse_csv(csv_path)
    repairs = asyncio.run(verified_repairs(parsed.unresolved)) if parsed.unresolved else {}
    unresolved = add_repaired(parsed, repairs)

    for site in parsed.sites:
        for key in _TAG_KEYS_SITE:
            if key in prev_sites.get(site["domain"], {}):
                site[key] = prev_sites[site["domain"]][key]
    # Derived entries (a publisher's website found via its app) are rebuilt by
    # tag_inventory from sg_apps.json; keep them so a rebuild doesn't lose their tags.
    web_domains = {s["domain"] for s in parsed.sites}
    parsed.sites += [s for d, s in prev_sites.items() if s.get("via") == "app" and d not in web_domains]
    for app in parsed.apps:
        for key in _TAG_KEYS_APP:
            if key in prev_apps.get(app["id"], {}):
                app[key] = prev_apps[app["id"]][key]

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(parsed.sites, indent=1, ensure_ascii=False), encoding="utf-8")
    APPS_PATH.write_text(json.dumps(parsed.apps, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(parsed.sites)} sites to {DATA_PATH}")
    print(f"wrote {len(parsed.apps)} apps to {APPS_PATH}")
    if repairs:
        print(f"repaired {len(repairs)} mangled names (verified live): " + ", ".join(f"{k} -> {v}" for k, v in repairs.items()))
    if parsed.bad_rows:
        print(f"skipped {parsed.bad_rows} malformed rows")
    if unresolved:
        print(f"{len(unresolved)} rows still have no usable domain (fix in the CSV if you want them):")
        for row in unresolved:
            print("  -", row["name"])


if __name__ == "__main__":
    main(sys.argv[1])
