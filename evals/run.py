"""Live eval: personas -> prompts -> sources for each gold case, written to JSON.

Costs real API calls (a few dozen model calls + 3 web searches per run) and
needs ANTHROPIC_API_KEY in the environment. Per-prompt answer calls are skipped
on purpose: they aren't affected by prompt/persona/source changes.

    python -m evals.run out/new.json
    python -m evals.metrics out/new.json            # compare several files side by side
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from app.models import Competitor, PersonaIn
from app.services.grounding import run_grounding
from app.services.persona_gen import generate_personas
from app.services.prompt_gen import generate_prompts
from evals.cases import CASES


async def _run_case(api_key: str, c: dict) -> dict:
    comps = [Competitor(name=n, match_names=m) for n, m in c["comps"]]
    t0 = time.time()
    personas = await generate_personas(
        api_key=api_key, industry=c["industry"], competitors=comps, buyer_context=c["buyer"],
        brand_summary=c["summary"], market=c["market"], persona_count=4,
    )
    t1 = time.time()
    persona_in = [
        PersonaIn(id=p.id, title=p.title, role=p.role, pains=p.pains, criteria=p.criteria, desc=p.desc)
        for p in personas
    ]
    prompts = await generate_prompts(
        api_key=api_key, brand=c["brand"], industry=c["industry"], buyer_context=c["buyer"],
        brand_summary=c["summary"], market=c["market"], personas=persona_in, prompts_per_persona=3,
    )
    t2 = time.time()
    sources, _ = await run_grounding(
        api_key=api_key, brand=c["brand"], industry=c["industry"], competitors=comps,
        brand_match_names=c["aliases"], personas=persona_in, prompts=prompts, market=c["market"],
    )
    t3 = time.time()
    return dict(
        case=c["name"],
        personas=[p.model_dump() for p in personas],
        prompts=prompts,
        sources=sources.model_dump(),
        secs=dict(personas=round(t1 - t0, 1), prompts=round(t2 - t1, 1), grounding=round(t3 - t2, 1)),
    )


async def main(out_path: str) -> None:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    results = await asyncio.gather(*(_run_case(api_key, c) for c in CASES), return_exceptions=True)
    out = [r if isinstance(r, dict) else dict(case=c["name"], error=repr(r)) for c, r in zip(CASES, results)]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(out_path, "->", [o.get("error", "ok") for o in out])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
