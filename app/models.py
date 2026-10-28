from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def to_camel(field_name: str) -> str:
    head, *tail = field_name.split("_")
    return head + "".join(word.capitalize() for word in tail)


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Sentiment(str, Enum):
    positive = "Positive"
    neutral = "Neutral"
    negative = "Negative"


# ── Shared input shapes ──────────────────────────────────────────────

class PersonaIn(CamelModel):
    id: str
    title: str
    role: str
    pains: str
    criteria: str
    # Optional so existing callers keep working; when sent, it carries the
    # persona's one-line "what they're trying to decide" framing into prompt
    # generation, which is otherwise only given pains/criteria lists.
    desc: str = ""


class Competitor(CamelModel):
    name: str
    # Every real-world variant that should count as a mention of this
    # competitor — the company name itself, common short forms, and any
    # sub-brands. A holding company like "PVH (parent of Tommy Hilfiger and
    # Calvin Klein)" needs both "Tommy Hilfiger" and "Calvin Klein" here,
    # since an AI answer is far more likely to name the sub-brand than the
    # parent company — matching only the literal display name misses almost
    # every real mention (confirmed live: zero matches against an answer
    # that named both sub-brands explicitly).
    match_names: list[str]


# ── /api/access ──────────────────────────────────────────────────────

AccessStatus = Literal["unknown", "revoked", "exhausted", "valid"]


class AccessStatusResponse(CamelModel):
    status: AccessStatus


# ── /api/detect ──────────────────────────────────────────────────────

class DetectRequest(CamelModel):
    query: str
    access_code: str


class DetectResponse(CamelModel):
    brand: str
    # Every real-world name variant that should count as a mention of the
    # brand itself — same rationale as Competitor.match_names above. An AI
    # answer is far more likely to name a specific product line (e.g.
    # "Redmi Note 13") than the parent brand (e.g. "Xiaomi"), or a short
    # form ("Samsung") than a fuller entered name ("Samsung Mobile") —
    # matching only the literal brand string misses those real mentions.
    brand_match_names: list[str]
    industry: str
    competitors: list[Competitor]
    # Free text, not a fixed enum — describes what kind of real-world choice
    # this actually is (buying software, booking a hotel, choosing a snack
    # brand, hiring an agency...) so persona/prompt generation stop assuming
    # every brand is B2B software being "evaluated" like a vendor.
    buyer_context: str
    # Fuller, user-facing paragraph shown for confirm-or-correct before
    # anything downstream is generated — see the Wizard's "Confirm your
    # brand details" step. Editable; whatever the user submits back flows
    # into persona/prompt generation as brandSummary below.
    brand_summary: str


# ── /api/categories ───────────────────────────────────────────────────

class CategoriesRequest(CamelModel):
    brand: str
    industry: str
    competitors: list[Competitor] = Field(default_factory=list)
    buyer_context: str | None = None
    brand_summary: str | None = None
    access_code: str


class CategorySuggestion(CamelModel):
    name: str
    # Scoped to this specific facet, not a copy of the brand's overall
    # buyerContext — a brand can genuinely have multiple distinct buyer
    # types (e.g. Hotel101 sells to both real-estate investors AND
    # short-stay guests), and selecting one facet must narrow to just its
    # relevant buyer type, not leak the other one back in downstream.
    buyer_context: str


class CategoriesResponse(CamelModel):
    categories: list[CategorySuggestion]


# ── /api/generate-personas ───────────────────────────────────────────

class GeneratePersonasRequest(CamelModel):
    brand: str
    industry: str
    competitors: list[Competitor] = Field(default_factory=list)
    buyer_context: str | None = None
    brand_summary: str | None = None
    market: str | None = None
    access_code: str
    persona_count: int | None = None


class GeneratedPersona(CamelModel):
    id: str
    title: str
    initials: str
    desc: str
    role: str
    pains: str
    criteria: str


class GeneratePersonasResponse(CamelModel):
    personas: list[GeneratedPersona]


# ── /api/personas/expand ─────────────────────────────────────────────

class ExpandPersonaRequest(CamelModel):
    # The one field a user types for a custom persona — everything else
    # below is the same category context /api/generate-personas already
    # takes, used to ground the fill-in fields rather than inventing them
    # from the description alone.
    description: str = Field(min_length=10, max_length=500)
    industry: str
    competitors: list[Competitor] = Field(default_factory=list)
    buyer_context: str | None = None
    brand_summary: str | None = None
    market: str | None = None
    access_code: str


class ExpandPersonaResponse(CamelModel):
    persona: GeneratedPersona


# ── /api/prompts ─────────────────────────────────────────────────────

class PromptsRequest(CamelModel):
    brand: str
    industry: str
    personas: list[PersonaIn]
    # Optional: when sent, generated prompts that name the brand (any alias) or a
    # competitor are dropped. Without them only the bare brand name is screened.
    competitors: list[Competitor] = Field(default_factory=list)
    brand_match_names: list[str] = Field(default_factory=list)
    buyer_context: str | None = None
    brand_summary: str | None = None
    market: str | None = None
    access_code: str
    prompts_per_persona: int | None = None
    # When non-empty, generate prompts_per_persona prompts PER category (each
    # scoped to that category's own name/buyerContext) and concatenate them
    # per persona, instead of one generation pass over the top-level
    # industry/buyerContext — lets a user run personas across every category
    # facet they pick, not just one.
    categories: list[CategorySuggestion] = Field(default_factory=list)


class PromptsResponse(CamelModel):
    prompts: dict[str, list[str]]


class AdaptSeedPromptRequest(CamelModel):
    brand: str
    industry: str
    personas: list[PersonaIn]
    # Optional: when sent, generated prompts that name the brand (any alias) or a
    # competitor are dropped. Without them only the bare brand name is screened.
    competitors: list[Competitor] = Field(default_factory=list)
    brand_match_names: list[str] = Field(default_factory=list)
    seed_prompt: str
    buyer_context: str | None = None
    brand_summary: str | None = None
    market: str | None = None
    access_code: str


class AdaptSeedPromptResponse(CamelModel):
    # One adapted prompt per persona — a single seed question in, one
    # persona-voiced variant out, not a list like PromptsResponse.
    prompts: dict[str, str]


# ── /api/analysis ────────────────────────────────────────────────────

class AnalysisRequest(CamelModel):
    brand: str
    # Optional/defaults empty for callers that don't send it yet — falls
    # back to matching on the bare brand string, today's behavior.
    brand_match_names: list[str] = Field(default_factory=list)
    industry: str
    competitors: list[Competitor] = Field(default_factory=list)
    buyer_context: str | None = None
    brand_summary: str | None = None
    market: str | None = None
    # Bounded: every persona and prompt is a billed call (and a search), and one
    # access-code use pays for the whole job. Generous next to the UI's 4 x 3.
    personas: list[PersonaIn] = Field(min_length=1, max_length=10)
    prompts: dict[str, list[str]]
    access_code: str

    @field_validator("prompts")
    @classmethod
    def _cap_prompts(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        if any(len(p) > 10 for p in value.values()):
            raise ValueError("at most 10 prompts per persona")
        return value


class AnalysisStartResponse(CamelModel):
    job_id: str


# ── Per-persona pipeline result ──────────────────────────────────────

class ResponsePart(CamelModel):
    text: str
    kind: Literal["brand", "competitor", "normal"]
    # Canonical entity name for brand/competitor parts (e.g. "Xiaomi" even
    # when text is "Redmi Note 13") — lets downstream aggregation attribute
    # a mention to the right entity directly, instead of re-matching text
    # against an alias list that can go stale the moment mention detection
    # picks up an alias (LLM-recognized sub-brand, etc.) the aggregation
    # layer doesn't independently know about. None for kind="normal".
    name: str | None = None


class PersonaExchange(CamelModel):
    prompt: str
    mentioned: bool
    sentiment: Sentiment
    rank: int | None
    vis: int
    quote: str
    parts: list[ResponsePart]


PersonaOpportunity = Literal["Defend", "Grow", "High", "Critical Gap"]


class PersonaResult(CamelModel):
    prompt: str
    mentioned: bool
    sentiment: Sentiment
    vis: int
    rank: int | None
    quote: str
    parts: list[ResponsePart]
    exchanges: list[PersonaExchange]  # one entry per prompt asked, in the order they were sent
    # Deterministic read on this persona's competitive position, from vis/mentioned
    # alone (see pipeline._classify_opportunity) — not mentioned at all is always a
    # "Critical Gap" regardless of how the other personas are doing.
    opportunity: PersonaOpportunity


PersonaStatus = Literal["waiting", "running", "done", "error"]


class PersonaEvent(CamelModel):
    persona_id: str
    status: PersonaStatus
    result: PersonaResult | None = None
    error: str | None = None


# ── Aggregation / sources / sitelist ─────────────────────────────────

class Overview(CamelModel):
    visibility_score: int
    mentioned: int
    total: int
    mention_rate: float
    avg_sentiment: Sentiment
    top_competitor: str | None
    failed_count: int = 0  # personas whose pipeline errored out entirely, excluded from the stats above


class Product(CamelModel):
    name: str
    count: int
    share: float
    is_brand: bool = False


class ScoreComponent(CamelModel):
    name: str
    score: int
    note: str


class CompetitorDiagnosis(CamelModel):
    rival_wins: list[str]
    brand_wins: list[str]
    gaps: list[str]


class Opportunity(CamelModel):
    title: str
    type: str  # "Critical gap" | "Source gap" | "Competitive" | "Keyword gap" | "Community" | ...
    impact: Literal["High", "Medium", "Low"]
    effort: Literal["High", "Medium", "Low"]
    detail: str
    action: str


class RadarCategory(CamelModel):
    name: str
    score: int


# In Citation.count, Publisher.mentions, Community.mentions and SourceIntel.cited
# the number is distinct pages found on that domain by the buyer-question
# search — not brand mentions. Brand evidence is SourceIntel.mentions.
class Citation(CamelModel):
    domain: str
    title: str
    count: int


class Publisher(CamelModel):
    name: str
    type: str
    mentions: int


class Community(CamelModel):
    name: str
    platform: str
    mentions: int


class SourceIntel(CamelModel):
    domain: str
    url: str
    category: str
    # Reach label, e.g. "High reach · 3 of 4 buyer questions": how many of the
    # brand-free buyer questions surfaced this domain.
    influence: str
    cited: int
    # `keywords` is a best-effort model inference from the domain's title.
    # `personas` (whose buyer question surfaced this source) and `competitors`
    # (named on its pages, most first) are measured — see grounding.py.
    keywords: list[str]
    personas: list[str]
    competitors: list[str]
    # `visibility` is NOT inferred: it's derived from `mentions`, the brand's
    # evidence on this domain's pages (page text, title, quoted passages, URL
    # slug) — see source_quality.py. "Not visible" means no evidence in what we
    # could read, not proof of absence.
    visibility: Literal["Visible", "Weak visibility", "Not visible"]
    mentions: int = 0
    # The title/passage backing `visibility` (empty when mentions == 0). `url`
    # above is the specific page this evidence came from, link-checked.
    evidence: str = ""
    # Same measure as `mentions`, per competitor named on these pages.
    competitor_mentions: dict[str, int] = Field(default_factory=dict)
    # Only when not visible: "competitor-owned" (rivals named here) or "open".
    gap: Literal["competitor-owned", "open"] | None = None


class SourceQuery(CamelModel):
    """One search the source step actually ran, for transparency."""

    persona: str | None
    question: str | None
    query: str
    # True when the model put a brand/competitor name in the query anyway;
    # its results were discarded.
    dropped: bool = False
    # True when the search returned an error instead of results.
    failed: bool = False


class Sources(CamelModel):
    citations: list[Citation]
    publishers: list[Publisher]
    communities: list[Community]
    source_intel: list[SourceIntel] = Field(default_factory=list)
    queries: list[SourceQuery] = Field(default_factory=list)
    # Set when a persona's question went unsearched or few sources came back,
    # so a short list reads as thin coverage rather than "nobody discusses this".
    coverage_note: str | None = None


class SitelistEntry(CamelModel):
    name: str
    category: str
    score: int
    inventory: list[str]
    availability: str
    cpm: str
    buyable: bool


class AnalysisComplete(CamelModel):
    overview: Overview
    products: list[Product]
    sources: Sources
    sitelist: list[SitelistEntry]
    score_breakdown: list[ScoreComponent]
    competitor_diagnosis: CompetitorDiagnosis
    opportunities: list[Opportunity]
    radar: list[RadarCategory]


class JobStatus(str, Enum):
    running = "running"
    complete = "complete"
    error = "error"


class JobSnapshot(CamelModel):
    job_id: str
    status: JobStatus
    personas: dict[str, PersonaEvent]
    result: AnalysisComplete | None = None
    error: str | None = None
