# hearsay.ai backend

FastAPI backend that measures whether a brand actually gets mentioned when
AI models answer real buyer questions — "AI visibility" research. No
accounts: access is gated by single/multi-use codes instead, and one
server-held Anthropic key funds every request.

## Architecture at a glance

- **No accounts, no persistent user data.** Access codes (`app/code_store.py`,
  SQLite) gate usage instead of logins. Analysis jobs live in an in-memory
  store (`app/job_store.py`) with a TTL — nothing is kept once a job expires.
- **Two-model split** (`app/config.py`): `ANTHROPIC_MODEL` (quality) is
  reserved for the one call that's the actual signal being measured — the
  buyer-persona "answer." Every other call (sentiment classification,
  prompt/persona generation, brand detection, source grounding) runs on
  `ANTHROPIC_FAST_MODEL` — same quality where it matters, a fraction of the
  cost everywhere else.
- **Brand detection is grounded, not guessed.** `/api/detect` runs a real
  `web_search` pass before extracting structured output — a lone
  memory-only call confidently misidentifies smaller or ambiguous brands
  (confirmed live: a real brand name got mapped to a same-named but
  unrelated company with zero warning signal).
- **The "answer" call never sees the brand.** It's deliberately blind to
  brand identity and to the fact that this is a test — it only sees the
  buyer question plus brand-agnostic context (`buyerContext`, `market`).
  That's what makes a mention "organic" rather than prompted.

## Quick start

```bash
cp .env.example .env        # fill in ANTHROPIC_API_KEY
docker compose up --build
curl http://localhost:8001/health
```

Mint yourself an access code before calling anything else:

```bash
docker compose exec api python -m app.scripts.mint_codes 1 5   # 1 code, 5 uses
```

## API reference

All POST bodies are camelCase JSON. Endpoints marked **(throttled)** share
one pre-spend call budget per access code (`MAX_PROMPT_CALLS_PER_CODE`,
default 10) — see [Access codes](#access-codes) below.

| Method & path | Purpose |
|---|---|
| `GET /health` | Liveness check. |
| `GET /api/access/status?code=` | Cheap, unthrottled check of whether a code is `valid` / `exhausted` / `revoked` / `unknown` — no AI call, no auth needed. |
| `POST /api/detect` **(throttled)** | Free-text query → `{ brand, industry, competitors, buyerContext, brandSummary }`. Two-stage: real web search, then structured extraction. Competitors come back as `{ name, matchNames }` — see [Competitor matching](#competitor-matching). |
| `POST /api/categories` **(throttled)** | Given a confirmed brand, suggests 4-6 specific category facets (e.g. "Sensitive Skin" within "Skincare"), each with its own correctly-scoped `buyerContext` — picking one narrows generation instead of testing the whole broad industry. |
| `POST /api/generate-personas` **(throttled)** | Generates buyer personas for a brand/industry/competitor set. Not hardcoded to B2B software — framing follows whatever `buyerContext` actually describes. |
| `POST /api/prompts` **(throttled, not consumed)** | Writes realistic buyer-research questions per persona. Every prompt is required to be phrased so a helpful answer would *name* a brand — purely educational questions with no recommendation angle are rejected by the generation prompt itself. |
| `POST /api/analysis` | Spends one use of the access code, kicks off a job, returns `{ jobId }` immediately — the actual work happens after the response is sent. |
| `GET /api/analysis/{jobId}/stream` | SSE: one `persona` event per completed persona, then one `complete` event with the full result. |
| `GET /api/analysis/{jobId}` | Polling fallback — same shape as the SSE `complete` payload, for reconnects. |
| `POST /admin/codes?count=&uses=` | Mint codes over HTTP. Disabled (503) unless `ADMIN_SECRET` is set. |
| `GET /admin/codes` | List all codes and their state. |
| `DELETE /admin/codes/{code}` | Soft-revoke — code stays in the table (audit trail), becomes permanently unusable. |

## Access codes

Gate usage without accounts (`app/code_store.py`, SQLite-backed):

- **Multi-use**: minted with `uses = N`; each `/api/analysis` call spends
  exactly one use, atomically (`UPDATE ... WHERE uses_remaining > 0`).
- **Shared pre-spend throttle**: `/api/detect`, `/api/categories`,
  `/api/generate-personas`, and `/api/prompts` all draw from one counter per
  code (`MAX_PROMPT_CALLS_PER_CODE`) — they're billed AI calls that happen
  *before* a code is actually redeemed, so without a cap a code could be
  hammered for free generation forever.
- **The throttle resets on every successful redemption**, not just once per
  code's lifetime — otherwise heavy iteration during one real use could
  starve every other use of the same multi-use code before they even start.
- **Revoke is soft-delete**: `DELETE /admin/codes/{code}` marks a code
  unusable everywhere but keeps its row and redemption history.

Mint via CLI (no `ADMIN_SECRET` needed):

```bash
docker compose exec api python -m app.scripts.mint_codes <count> <uses>
```

## Competitor matching

Competitors aren't plain strings — `{ name, matchNames }`. `name` is what's
shown to the user (can stay descriptive, e.g. `"PVH (parent of Tommy
Hilfiger and Calvin Klein)"`); `matchNames` is every real-world alias that
should count as a mention of that competitor. A holding company is almost
always mentioned by a sub-brand's name, not its own — matching only the
literal display name misses nearly every real mention (confirmed live: zero
matches against an answer that named both sub-brands explicitly).

Mention detection (`app/services/scoring.py`) also strips common corporate
suffixes (Inc., Corp., Ltd., LLC, Group, Holdings, ...) before matching, so
`"Tapestry, Inc."` still matches a bare `"Tapestry"` mention with no alias
configured at all.

Every prompt actually run for a persona (not just the highest-scoring
"representative" one) is checked for mentions — `PersonaResult.exchanges`
carries every individual prompt/answer pair, and Share-of-Voice counting
(`app/services/aggregation.py`) looks across all of them, not just the one
that happens to be shown by default.

## Visibility scoring

Fully deterministic, no extra AI call (`app/services/scoring.py`):

```
rank_score = max(0, 100 - (rank - 1) × 20)     # rank 1→100, 2→80, 3→60, ...
multiplier = {Positive: 1.0, Neutral: 0.7, Negative: 0.35}
score       = round(rank_score × multiplier), clamped 0-100
not mentioned → always 0
```

A persona's own score is the average across all its prompts; the overview's
`visibilityScore` is the average across all personas' (already-averaged)
scores — not a flat average across every raw prompt.

## Environment variables

See [`.env.example`](.env.example) for the full list with inline comments —
`ANTHROPIC_API_KEY` is the only one you must set yourself.

## Project structure

```
app/
  main.py              FastAPI app, router registration, lifespan
  config.py            Settings (env-driven)
  models.py            Pydantic request/response shapes
  code_store.py         Access codes (SQLite)
  job_store.py           In-memory analysis job store + SSE fan-out
  runner.py               Orchestrates one full analysis job
  routes/                 One file per endpoint group
  services/
    detect.py              Brand/industry/competitor detection (grounded)
    category_gen.py        Product-category facet suggestions
    persona_gen.py          Buyer persona generation
    prompt_gen.py           Buyer-question generation
    pipeline.py             Per-persona answer + sentiment pipeline
    scoring.py              Deterministic mention/rank/visibility scoring
    aggregation.py          Overview + Share-of-Voice aggregation
    grounding.py            Real citations via web_search
    anthropic_client.py     Thin Anthropic SDK wrapper
  scripts/
    mint_codes.py            CLI code minting
```
