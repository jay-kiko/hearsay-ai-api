# hearsay-ai-api

FastAPI backend for AI-visibility research: synthetic buyer personas ask AI
assistants real buying questions, and we measure whether/how a brand comes up.
See README.md for endpoints and env vars; this file is what to check *before*
changing anything.

## How to work here

You are a principal/senior engineer with 10+ years building high-scale,
resilient web applications. Prioritize maintainability, clean architecture,
security, accessibility and performance over quick fixes. Write code other
engineers enjoy reviewing: modular, well-typed, defensively handled, easy to test.

- For any change that touches more than one module or alters behaviour, explain
  design decisions and trade-offs first, and name edge cases, failure modes and
  security risks. A one-line fix doesn't need an essay.
- Ask a clarifying question only when the answer changes the design. Otherwise
  state the assumption and proceed.
- Match surrounding code: comment density, naming, idiom. Comments here explain
  *why* (often a confirmed-live failure), not what.
- Python 3.12 (Dockerfile). Run tests with `python -m pytest` from the repo root.

## Pipeline (one analysis job — `app/runner.py`)

1. **Detect** (`detect.py`) — grounded web search → brand, aliases, industry, competitors, buyer context.
2. **Categories / personas / prompts** (`category_gen.py`, `persona_gen.py`, `prompt_gen.py`) — generative; the *wording is the product*.
3. **Per-persona** (`pipeline.py`) — for each prompt: one "answer" call (no brand, no web search) → one sentiment call → deterministic mention/rank/score (`scoring.py`).
4. **Grounding** (`grounding.py` + `source_quality.py`) — ONE web-search call per job that searches one brand-free buyer question per persona (each persona's first prompt) → third-party sources, then measures brand + competitors on each page. The Singapore inventory (`inventory.py`) only labels sources as buyable in the sitelist. Separate from step 3: these are pages buyers' questions surface, not pages that shaped a specific answer.
5. **Aggregation** (`aggregation.py`) — deterministic: Overview, Share of Voice, score breakdown, composite.
6. **Insights** (`insights_gen.py`) — one generative call over everything above.

## User-facing numbers and what backs them

| Shown to the client | Computed from | Where |
|---|---|---|
| Presence / mention rate | prompt-level `exchange.mentioned` (regex on the answer, alias-aware) | `aggregation._prompt_mention_counts` |
| Share of Voice | share of all product mentions | `aggregation.build_products` |
| Ranking, Sentiment | first-mention position; sentiment call | `scoring.py`, `pipeline.py` |
| Source "Visible" badge, `mentions`, `evidence` | brand aliases on the **linked page only**: its text where readable, plus title / quoted passages / URL slug | `grounding.py` via `source_quality` + `scoring.count_brand_mentions` |
| Source Authority score | the same `mentions` as the badge | `aggregation.build_score_breakdown` |
| Source `url` | the page with brand evidence, link-checked (dead, homepage-redirect and internal-address links dropped) | `grounding.py` |
| Source `influence` (reach), sitelist `score` | how many of the brand-free buyer questions surfaced the domain | `grounding._reach_label` |
| Source `personas` (whose question surfaced it), `competitors`, `competitorMentions`, `gap` | measured from the search attribution and page text | `grounding.py` |
| Source `keywords`, `Publisher.type` | **model inference** (plausibility, not measurement) | `grounding._enrich_sources` |

## Invariants — a change that breaks one is a bug

1. **A source is a specific third-party page.** Never the brand's own site, a competitor's, or a booking/affiliate site. Forums, independent blogs/reviews and editorial are the point. Clicking a source lands on the page that mentions the brand, never a dead link or a homepage.
2. **Measured vs inferred is never blurred.** A value shown to a client as fact (a badge, a score, a count) must be computed from evidence. If it is a model's guess, it is labelled as such or not shown. Never let a model fill a "visibility"/"mentioned"/"cited" style field.
3. **One definition per number.** A figure that appears in two places comes from one function. "Does this text mention the brand" is `scoring.count_brand_mentions` / `find_mentions`; don't re-derive it with substring checks.
4. **Never leak the brand into the answer call** (`pipeline._answer_system` deliberately omits brand, summary and industry) — it measures whether the brand comes up organically.
5. **Generated prompts never name the brand or reveal a test**, and must still make a helpful answer name real products (otherwise the measurement is empty), while reading like a real person typed them: varied shape, one concern each, no stock AI phrasing.
6. **Fail open on enrichment, never on trust.** A failed optional call degrades (empty list, fallback) instead of failing a job whose persona results succeeded — but a source we could not classify as independent is not shown at all, and a missing measurement (no sources) is left out of the headline score, never scored as 0.
7. **Sources are found like a buyer, never by brand lookup.** The source search and its relevance filter never see the brand or competitor names; a search that names one is dropped. Thin results get a `coverageNote`, never a silent fallback.
8. **Counts agree.** Overview, Presence and the breakdown all use prompt-level counts (see commits a2c37a0, 3c57800).

## Model tiers (`app/config.py`)

Choose per call by: *does its output directly shape what a client sees?* and *how often does it run?*

- **quality** (`anthropic_quality_model`): generative steps whose content is the result and that run once or a few times per job — persona gen/expand, prompt gen/seed-adapt, categories, insights.
- **standard** (`anthropic_model`): the per-prompt "answer" call (the signal being measured; fans out per prompt, so cost-sensitive), plus judgement calls that decide what the report shows — detect research, source `origin` classification.
- **fast** (`anthropic_fast_model`): mechanical work only — sentiment classification, domain relevance filter, detect formatting, and the source search (it only runs the given buyer questions; the search engine decides what comes back).

A new LLM call must be assigned a tier deliberately and say why in a comment.
Promoting a high-fan-out call needs a cost estimate first.

## Testing

- `python -m pytest` — unit tests, no network, no API key. `test_grounding.py`
  fakes the model and page-fetch edges and encodes the client-reported source
  bugs; extend it when you touch grounding or scoring.
- `python -m evals.run out/x.json` then `python -m evals.metrics out/x.json` —
  the live gold-case eval (`evals/cases.py`: Disney Cruise, Love Bonito, Xero;
  Singapore). Costs real API calls. Run it before and after any change to prompt,
  persona or grounding logic and compare. Metrics screen for length, repeated
  openers, stock AI phrasing, brand/competitor leakage and source lists; they
  cannot tell you a prompt sounds human or a source is useful, so read a sample.
  Add a case when a client reports a new failure.

## Gotchas confirmed live

- **One worker process only.** Jobs, their progress streams and the rate limiter live in memory, and
  `code_store.refund_unfinished()` treats every unfinished redemption at startup as a job the last process
  lost. A second worker would 404 streams for jobs started in the other one and refund its running jobs.
  Scaling out needs a shared job store first.
- An access-code use is spent before the job starts and refunded by `runner._fail` when the job delivers
  nothing; a delivered job is marked `complete` so it never is.
- The quality-tier model rejects a forced `tool_choice` (400). `call_structured`
  falls back to `tool_choice=auto` and remembers the model; don't remove that.
- Search snippets alone are too thin to judge brand visibility (an obviously
  on-topic review read "Not visible"), so grounding reads the page text during
  the link check. Blocked / script-rendered pages fall back to snippets + URL slug.
- Translate/cache/archive hosts proxy other sites' pages; `source_quality` checks
  hosts embedded in the URL so an owned site can't enter through one.
- `web_search` `user_location.country` 400s for unsupported codes (SG is one — Singapore is localised by city + timezone instead, see `grounding._location_for`); `call_web_search` drops it and retries. (`allowed_domains` also 400s the whole request if any domain is one the crawler refuses — straitstimes.com, channelnewsasia.com, reuters.com — relevant if a domain allowlist is ever reintroduced.)
- The page fetcher (`source_quality.check_pages`) fetches URLs from search results, i.e. attacker-influenced input: every hop is checked syntactically and by DNS resolution for non-public addresses. Keep both checks.
- The model trims long questions into shorter searches, sometimes to bare keywords; `attribute_queries` matches on the share of the search's words drawn from each question, falling back to order. Keyword-trimmed searches return quite different pages, which is most of the run-to-run variance.
- Apps in the DSP CSV are mapped to publisher websites (`tag_inventory`), but the `editorial` flag is deliberately strict, so most app-publisher sites (MyFitnessPal, Moovit, Carousell) are inventory-only, not sources. Loosening it is a product call.
- The DSP CSV mixes real sites with Android package names that look like domains (`com.livescore`, `tunein.player`); `inventory._clean_domain` validates the TLD and package prefixes. Re-check after importing a new export.
- Persona text feeds prompt generation, so a brand or competitor named in a
  persona leaks into the prompts. Generated personas/prompts must stay neutral.
