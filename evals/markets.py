"""Live eval of source localisation: the same brands across several markets,
including smaller ones, so a change that helps Singapore can't quietly leave
Kenya with US pages. Costs real API calls.

Prompts are generated once and reused, so before/after grounding runs search
the same buyer questions and differ only by the grounding change:

    python -m evals.markets prompts out/mk_prompts.json
    python -m evals.markets ground out/mk_prompts.json out/mk_before.json
    python -m evals.markets judge out/mk_before.json [out/mk_after.json ...]

The judge's labels are a screen, like evals.metrics: read a sample of the
URLs before trusting a number.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import unicodedata
from collections import Counter

from app.config import get_settings
from app.models import Competitor, PersonaIn
from app.services.anthropic_client import call_structured
from app.services.grounding import run_grounding
from app.services.persona_gen import generate_personas
from app.services.prompt_gen import generate_prompts

MARKETS = [
    "Singapore",
    "Makati, Philippines",
    "Nairobi, Kenya",
    "Poland",
    "Lima, Peru",
    "Dubai, United Arab Emirates",
]

# One B2B, one consumer brand: software reviews are legitimately global, shoes
# are bought locally and pull in stores, which must never be sources.
BRANDS = [
    dict(
        name="xero",
        brand="Xero",
        aliases=["Xero"],
        industry="Accounting software",
        buyer="Small business owners and bookkeepers choosing cloud accounting software",
        summary="Xero is cloud accounting software for small businesses: invoicing, bank reconciliation, "
                "payroll add-ons and a large app marketplace.",
        comps=[("QuickBooks", ["QuickBooks", "QuickBooks Online"]), ("Zoho Books", ["Zoho Books"]),
               ("Sage", ["Sage", "Sage Accounting"])],
    ),
    dict(
        name="nike_running",
        brand="Nike",
        aliases=["Nike", "Pegasus", "Vomero", "Invincible"],
        industry="Running shoes",
        buyer="Recreational and serious runners choosing their next pair of running shoes",
        summary="Nike makes running shoes from daily trainers like the Pegasus to cushioned and racing "
                "models, sold in its own stores, online and through retailers.",
        comps=[("Adidas", ["Adidas", "Adizero"]), ("ASICS", ["ASICS", "Gel-Kayano", "Novablast"]),
               ("Hoka", ["Hoka", "Clifton"])],
    ),
]


def _combos() -> list[tuple[dict, str]]:
    return [(b, m) for b in BRANDS for m in MARKETS]


def _comps(b: dict) -> list[Competitor]:
    return [Competitor(name=n, match_names=m) for n, m in b["comps"]]


async def _prompts_for(api_key: str, b: dict, market: str) -> dict:
    buyer = f"{b['buyer']} in {market}"
    personas = await generate_personas(
        api_key=api_key, industry=b["industry"], competitors=_comps(b), buyer_context=buyer,
        brand_summary=b["summary"], market=market, persona_count=4,
    )
    persona_in = [
        PersonaIn(id=p.id, title=p.title, role=p.role, pains=p.pains, criteria=p.criteria, desc=p.desc)
        for p in personas
    ]
    prompts = await generate_prompts(
        api_key=api_key, brand=b["brand"], industry=b["industry"], buyer_context=buyer,
        brand_summary=b["summary"], market=market, personas=persona_in, prompts_per_persona=3,
        competitors=_comps(b), brand_match_names=b["aliases"],
    )
    return dict(case=b["name"], market=market, personas=[p.model_dump() for p in persona_in], prompts=prompts)


async def _ground(api_key: str, row: dict) -> dict:
    b = next(x for x in BRANDS if x["name"] == row["case"])
    sources, _ = await run_grounding(
        api_key=api_key, brand=b["brand"], industry=b["industry"], competitors=_comps(b),
        brand_match_names=b["aliases"], personas=[PersonaIn(**p) for p in row["personas"]],
        prompts=row["prompts"], market=row["market"],
    )
    return dict(
        case=row["case"], market=row["market"],
        queries=[q.model_dump() for q in sources.queries],
        shown=[dict(url=s.url, domain=s.domain, category=s.category) for s in sources.source_intel],
        coverage_note=sources.coverage_note,
    )


_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "label": {"type": "string", "enum": ["local", "global", "wrong_market", "store"]},
                },
                "required": ["url", "label"],
            },
        }
    },
    "required": ["labels"],
}
_JUDGE_SYSTEM = (
    "You label web pages shown to a client as sources that buyers in a given market read. For each URL: "
    "'local' = written for buyers in that market (local publication, local forum, or a page specifically "
    "about that country/city). 'global' = not tied to any country and genuinely useful to a buyer there "
    "(international reviews, global forums). 'wrong_market' = written for buyers in a different country "
    "(another country's prices, stores, regulations, tax rules, events or availability). 'store' = a "
    "retailer, marketplace, booking or brand/vendor site, including its blog. Judge from the URL and domain."
)


async def _judge(api_key: str, row: dict) -> list[str]:
    if not row["shown"]:
        return []
    result = await call_structured(
        api_key=api_key, model=get_settings().anthropic_model, system=_JUDGE_SYSTEM,
        user=f"Market: {row['market']}\nCategory: {row['case']}\n\n" + "\n".join(s["url"] for s in row["shown"]),
        tool_name="label_sources", tool_description="Label every URL.", input_schema=_JUDGE_SCHEMA,
        max_tokens=2048, retry_malformed=True,
    )
    by_url = {x.get("url"): x.get("label") for x in result["labels"] if isinstance(x, dict)}
    return [by_url.get(s["url"], "unlabelled") for s in row["shown"]]


async def main(cmd: str, args: list[str]) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or get_settings().anthropic_api_key
    if cmd == "prompts":
        rows = await asyncio.gather(*(_prompts_for(api_key, b, m) for b, m in _combos()))
        _write(args[0], rows)
    elif cmd == "ground":
        rows = json.load(open(args[0]))
        out = await asyncio.gather(*(_ground(api_key, r) for r in rows), return_exceptions=True)
        _write(args[1], [o if isinstance(o, dict) else dict(case=r["case"], market=r["market"], error=repr(o),
                                                          queries=[], shown=[]) for r, o in zip(rows, out)])
    elif cmd == "judge":
        for path in args:
            rows = json.load(open(path))
            labels = await asyncio.gather(*(_judge(api_key, r) for r in rows))
            print(f"\n{path}")
            print(f"{'case':13} {'market':28} shown local global wrong store  queries naming market")
            total = Counter()
            for r, ls in zip(rows, labels):
                c = Counter(ls)
                total.update(c)
                places = [_plain(p) for p in r["market"].split(",") if p.strip()]
                named = sum(any(p in _plain(q.get("query") or "") for p in places) for q in r["queries"])
                print(f"{r['case']:13} {r['market']:28} {len(ls):5} {c['local']:5} {c['global']:6} "
                      f"{c['wrong_market']:5} {c['store']:5}  {named}/{len(r['queries'])}"
                      + (f"  ERROR {r['error'][:60]}" if r.get("error") else ""))
                for s, label in zip(r["shown"], ls):
                    s["label"] = label
            print(f"{'TOTAL':42} {sum(total.values()):5} {total['local']:5} {total['global']:6} "
                  f"{total['wrong_market']:5} {total['store']:5}")
            _write(path.replace(".json", ".judged.json"), rows)
    else:
        raise SystemExit(__doc__)


def _plain(text: str) -> str:
    """Lowercase without accents, so "Perú" counts as naming Peru."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower().strip()


def _write(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(path)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2:]))
