from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
    industry: str
    competitors: list[str]
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
    competitors: list[str] = Field(default_factory=list)
    buyer_context: str | None = None
    brand_summary: str | None = None
    access_code: str


class CategoriesResponse(CamelModel):
    categories: list[str]


# ── /api/generate-personas ───────────────────────────────────────────

class GeneratePersonasRequest(CamelModel):
    brand: str
    industry: str
    competitors: list[str] = Field(default_factory=list)
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


# ── /api/prompts ─────────────────────────────────────────────────────

class PromptsRequest(CamelModel):
    brand: str
    industry: str
    personas: list[PersonaIn]
    buyer_context: str | None = None
    brand_summary: str | None = None
    market: str | None = None
    access_code: str
    prompts_per_persona: int | None = None


class PromptsResponse(CamelModel):
    prompts: dict[str, list[str]]


# ── /api/analysis ────────────────────────────────────────────────────

class AnalysisRequest(CamelModel):
    brand: str
    industry: str
    competitors: list[str] = Field(default_factory=list)
    buyer_context: str | None = None
    brand_summary: str | None = None
    market: str | None = None
    personas: list[PersonaIn]
    prompts: dict[str, list[str]]
    access_code: str


class AnalysisStartResponse(CamelModel):
    job_id: str


# ── Per-persona pipeline result ──────────────────────────────────────

class ResponsePart(CamelModel):
    text: str
    kind: Literal["brand", "competitor", "normal"]


class PersonaResult(CamelModel):
    prompt: str
    mentioned: bool
    sentiment: Sentiment
    vis: int
    rank: int | None
    quote: str
    parts: list[ResponsePart]


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


class Sources(CamelModel):
    citations: list[Citation]
    publishers: list[Publisher]
    communities: list[Community]


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
