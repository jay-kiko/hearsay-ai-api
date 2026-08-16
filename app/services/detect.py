"""Home-screen query detection — turns the free-text brand description into
a structured brand/industry/competitor set, the input the Wizard needs to
generate personas and prompts against.

Two stages, not one blind structured call: a lone forced tool call answering
straight from the model's own memory confidently misidentifies smaller or
ambiguous brands (confirmed live — "Careline" got mapped to an unrelated US
medical-alert company, "Ever Belena" [a typo] got mapped to a wine brand,
both with zero warning signal since the output still validated cleanly).
Grounding first in a real web_search pass — same tool grounding.py already
uses — lets the model read actual search results distinguishing candidates
instead of guessing from parametric memory, then a cheap second call just
formats those findings into the response shape.
"""
from __future__ import annotations

from app.config import get_settings
from app.models import Competitor, DetectResponse
from app.services.anthropic_client import call_structured, call_web_search

_TOOL_NAME = "detect_brand"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "brand": {"type": "string", "description": "The brand or product name."},
        "industry": {
            "type": "string",
            "description": "A specific category name, e.g. 'Project Management Software', not a vague label.",
        },
        "competitors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Real, specific competitor name — no generic placeholders. Can include "
                            "parenthetical context for a holding company, e.g. 'PVH (parent of Tommy "
                            "Hilfiger and Calvin Klein)'."
                        ),
                    },
                    "matchNames": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Every real-world name variant that should count as a mention of this "
                            "competitor — the company name itself, common short forms, AND any "
                            "sub-brands. An AI answer is far more likely to name a sub-brand (e.g. "
                            "'Tommy Hilfiger') than a parent holding company (e.g. 'PVH') — include "
                            "both, and every distinct sub-brand, so none of them get missed."
                        ),
                    },
                },
                "required": ["name", "matchNames"],
            },
            "description": "3-5 real, specific competitors in that same category.",
        },
        "buyerContext": {
            "type": "string",
            "description": (
                "One sentence describing what real-world choice buyers in this category actually "
                "make — grounded in the true nature of this specific brand's industry, not assumed "
                "to be software. Examples of the range this must cover: 'Businesses and teams "
                "evaluating and adopting project management software' (B2B SaaS), 'Travelers and "
                "event planners choosing where to stay for leisure, business, or celebrations' "
                "(hospitality), 'Shoppers picking a snack brand off the shelf or in a delivery app' "
                "(consumer packaged goods), 'Marketing teams choosing an agency or platform to run "
                "campaigns' (services/agencies), 'Everyday consumers choosing a payment app for "
                "daily transactions and remittances' (consumer fintech). Never default to a "
                "software/vendor framing unless the brand genuinely is software."
            ),
        },
        "brandSummary": {
            "type": "string",
            "description": (
                "A fuller, 2-4 sentence plain-language summary of what this brand actually is and "
                "does — its product/service mix, market position, and what distinguishes it from "
                "competitors. This is shown directly to the user to confirm or correct before "
                "anything downstream is generated, so it must be accurate and specific to the "
                "research findings, not generic boilerplate."
            ),
        },
    },
    "required": ["brand", "industry", "competitors", "buyerContext", "brandSummary"],
}

_RESEARCH_SYSTEM = (
    "Use the web_search tool to research the real company or brand described in the user's "
    "message. Find out: what it actually sells or does, its true industry/category, its real, "
    "specific direct competitors, and who actually buys from it — including the country/region "
    "if it's a regional brand rather than global. The name may be ambiguous, misspelled, or "
    "shared with an unrelated company — search specifically to confirm which real company is "
    "meant using every context clue available, rather than assuming the most famous match. "
    "Write a factual summary of what you found in 3-5 sentences: name the exact industry, name "
    "real competitors by name, and note the target market/region if relevant. If you genuinely "
    "cannot find reliable information, say so plainly rather than guessing."
)

_EXTRACT_SYSTEM = (
    "You extract a brand, its industry category, its real competitors, its buyer context, and a "
    "brand summary from a factual research summary below, already grounded in real web search "
    "results — trust it over any assumption. Identify the brand name and a specific "
    "industry/category label, then name 3-5 real, specific companies from the research that "
    "actually compete in that category — never generic placeholders and never the brand itself. "
    "For each competitor, also list every real name variant an AI might use to refer to it — "
    "the company name, common short forms, and any sub-brands (a holding company is usually "
    "mentioned by its sub-brand's name, not its own). Identify what kind of real-world choice "
    "this actually is: software/vendor procurement, a consumer product purchase, a "
    "hospitality/travel booking, a service or agency hire, or something else — ground this in "
    "the true nature of the brand's industry from the research, not a generic 'evaluating tools' "
    "frame. Also write a brandSummary: a fuller, factual paragraph describing what the brand "
    "actually is and does, for a user to review and correct before anything else gets generated "
    "from it — specific to the research, not boilerplate. If the research found nothing useful, "
    "fall back to your own best judgment from the original query, and say so plainly in the "
    "brandSummary rather than inventing confident-sounding detail."
)


def _normalize_competitors(value: object) -> list[Competitor]:
    """Tool-forced output isn't a hard type guarantee — fields have come back
    in an unexpected shape in practice despite the schema declaring one.
    Recover a usable list of Competitors either way rather than 500."""
    if isinstance(value, str):
        # Whole competitors field collapsed to a comma-joined string — no
        # matchNames available at all, fall back to name-only matching.
        return [Competitor(name=n.strip(), match_names=[n.strip()]) for n in value.split(",") if n.strip()]

    if not isinstance(value, list):
        return []

    competitors: list[Competitor] = []
    for entry in value:
        if isinstance(entry, str):
            name = entry.strip()
            if name:
                competitors.append(Competitor(name=name, match_names=[name]))
            continue
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        match_names_raw = entry.get("matchNames", [])
        if isinstance(match_names_raw, str):
            match_names = [m.strip() for m in match_names_raw.split(",") if m.strip()]
        elif isinstance(match_names_raw, list):
            match_names = [str(m).strip() for m in match_names_raw if str(m).strip()]
        else:
            match_names = []
        if name not in match_names:
            match_names.append(name)
        competitors.append(Competitor(name=name, match_names=match_names))
    return competitors


async def _research_brand(*, api_key: str, model: str, query: str) -> str:
    response = await call_web_search(
        api_key=api_key,
        model=model,
        system=_RESEARCH_SYSTEM,
        user=query,
        # Several search rounds plus extended thinking easily exceed a small
        # cap, truncating mid-synthesis before the summary is even written
        # (same failure mode fixed in grounding.py).
        max_tokens=4096,
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()


async def detect_brand(*, api_key: str, query: str) -> DetectResponse:
    settings = get_settings()

    # Stage 1: ground in real search results, on the quality model — this is
    # where disambiguation actually happens (reading real snippets instead
    # of guessing from memory), so it's not worth cheapening.
    research = await _research_brand(api_key=api_key, model=settings.anthropic_model, query=query)

    # Stage 2: format already-grounded findings into the response shape —
    # genuinely mechanical now that the hard part is done, so the fast model
    # is fine here.
    result = await call_structured(
        api_key=api_key,
        model=settings.anthropic_fast_model,
        system=_EXTRACT_SYSTEM,
        user=(
            f"Original query: {query}\n\n"
            f"Research findings:\n{research or '(search returned nothing useful — use best judgment from the query alone)'}"
        ),
        tool_name=_TOOL_NAME,
        tool_description="Return the detected brand, industry, competitors, buyer context, and brand summary.",
        input_schema=_INPUT_SCHEMA,
        max_tokens=768,
    )

    return DetectResponse(
        brand=result["brand"],
        industry=result["industry"],
        competitors=_normalize_competitors(result.get("competitors", [])),
        buyer_context=result["buyerContext"],
        brand_summary=result["brandSummary"],
    )
