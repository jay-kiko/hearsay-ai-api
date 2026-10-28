"""Scores evals/run.py output. These are screening metrics, not a verdict: they
catch length creep, repeated openers, stock AI phrasing and brand leakage, but
only reading the prompts tells you whether they sound human. Always read a few.

    python -m evals.metrics out/base.json out/new.json
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import Counter

from evals.cases import CASES

STOCK = [
    "i'm looking for", "i am looking for", "i want to find", "what are the best options", "that prioritize",
    "that offer", "ideal", "seamless", "top-rated", "comprehensive", "tailored", "robust", "leverage",
    "navigate", "ensure", "key factors", "options that", "which brands", "can you recommend",
    "could you recommend", "please recommend", "that meet", "that balance", "while also", "that combine",
]
_NAMES = {c["name"]: c["names"] for c in CASES}


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def report(path: str) -> None:
    for case in json.load(open(path)):
        if "error" in case:
            print(f"  [{case['case']}] ERROR {case['error'][:120]}")
            continue
        prompts = [p for ps in case["prompts"].values() for p in ps]
        personas = case["personas"]
        lengths = [_words(p) for p in prompts]
        names = _NAMES[case["case"]]
        openers = Counter(" ".join(p.lower().split()[:3]) for p in prompts)
        field_words = {k: st.mean(_words(p[k]) for p in personas) for k in ("desc", "role", "pains", "criteria")}
        intel = case["sources"].get("sourceIntel") or case["sources"].get("source_intel") or []
        print(
            f"  [{case['case']}] prompts: words mean={st.mean(lengths):.1f} max={max(lengths)} | "
            f"stock-phrase={sum(any(s in p.lower() for s in STOCK) for p in prompts)}/{len(prompts)} | "
            f"repeated openers={sum(c for c in openers.values() if c > 1)} | "
            f"brand/competitor named: prompts={sum(any(n in p.lower() for n in names) for p in prompts)}, "
            f"personas={sum(any(n in ' '.join(map(str, p.values())).lower() for n in names) for p in personas)}"
        )
        print("      persona field words: " + ", ".join(f"{k}={v:.0f}" for k, v in field_words.items()))
        if case["sources"].get("coverageNote") or case["sources"].get("coverage_note"):
            print("      coverage note: " + (case["sources"].get("coverageNote") or case["sources"].get("coverage_note")))
        print(f"      sources={len(intel)}: " + ", ".join(f"{s['domain']}({s.get('visibility', '?')[0]})" for s in intel))


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path)
        report(path)
