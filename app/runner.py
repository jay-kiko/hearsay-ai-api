"""Orchestrates one analysis job end to end: per-persona pipeline (concurrency
capped, retried once, one bad persona doesn't fail the job), then the single
whole-job grounding call, then deterministic aggregation. Runs as a detached
asyncio task started right after POST /api/analysis responds — this is the
"actual persona calls happen after the response is sent" behavior from the
design doc.
"""
from __future__ import annotations

import asyncio
import logging

from app.code_store import get_code_store
from app.config import get_settings
from app.errors import user_facing_error
from app.job_store import JobStore
from app.models import AnalysisComplete, AnalysisRequest, Competitor, PersonaEvent, PersonaIn, PersonaResult
from app.services import aggregation
from app.services.grounding import run_grounding
from app.services.insights_gen import generate_insights
from app.services.pipeline import run_persona

logger = logging.getLogger("hearsay.runner")


async def _run_one_persona(
    *,
    semaphore: asyncio.Semaphore,
    job_store: JobStore,
    job_id: str,
    persona: PersonaIn,
    prompts: list[str],
    brand: str,
    brand_match_names: list[str],
    competitors: list[Competitor],
    buyer_context: str | None,
    market: str | None,
    api_key: str,
) -> tuple[str, PersonaResult | None]:
    job_store.set_persona_status(job_id, persona.id, "running")
    async with semaphore:
        try:
            result = await run_persona(
                persona=persona,
                prompts=prompts,
                brand=brand,
                brand_match_names=brand_match_names,
                competitors=competitors,
                buyer_context=buyer_context,
                market=market,
                api_key=api_key,
            )
            job_store.update_persona(job_id, PersonaEvent(persona_id=persona.id, status="done", result=result))
            return persona.id, result
        except Exception as exc:  # noqa: BLE001 - one persona's failure must not sink the job
            logger.exception("persona %s failed", persona.id)
            job_store.update_persona(
                job_id, PersonaEvent(persona_id=persona.id, status="error", error=user_facing_error(exc))
            )
            return persona.id, None


async def _fail(job_store: JobStore, job_id: str, message: str) -> None:
    """Every failure path delivers nothing, so the access-code use is handed back."""
    job_store.fail(job_id, message)
    try:
        await get_code_store().refund(job_id)
    except Exception:  # noqa: BLE001 - a refund problem must not mask the job's own failure
        logger.exception("refund failed for job %s", job_id)


async def _stop(task: asyncio.Task) -> None:
    """Cancels and waits, so the task's in-flight calls are released and an
    exception it already raised is retrieved rather than logged as lost."""
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def run_job(job_id: str, request: AnalysisRequest, job_store: JobStore, anthropic_api_key: str) -> None:
    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.persona_concurrency)
    # Falls back to the bare brand name for callers that don't send aliases yet.
    brand_match_names = request.brand_match_names or [request.brand]

    try:
        tasks = [
            _run_one_persona(
                semaphore=semaphore,
                job_store=job_store,
                job_id=job_id,
                persona=persona,
                prompts=request.prompts.get(persona.id) or [],
                brand=request.brand,
                brand_match_names=brand_match_names,
                competitors=request.competitors,
                buyer_context=request.buyer_context,
                market=request.market,
                api_key=anthropic_api_key,
            )
            for persona in request.personas
            if request.prompts.get(persona.id)
        ]

        if not tasks:
            await _fail(job_store, job_id, "No personas had prompts to analyze")
            return

        # Grounding needs only the prompts, not the persona answers, so it runs
        # alongside them instead of adding its whole duration (search, filter,
        # classification, page fetches) after them.
        grounding = asyncio.create_task(
            run_grounding(
                api_key=anthropic_api_key,
                brand=request.brand,
                industry=request.industry,
                competitors=request.competitors,
                brand_match_names=brand_match_names,
                personas=request.personas,
                prompts=request.prompts,
                market=request.market,
            )
        )
        try:
            pairs = await asyncio.gather(*tasks)
        except BaseException:
            await _stop(grounding)
            raise
        results = {pid: result for pid, result in pairs if result is not None}

        if not results:
            await _stop(grounding)  # nothing to attach sources to; stop spending on it
            await _fail(job_store, job_id, "All persona calls failed — see per-persona errors for details")
            return

        sources, sitelist = await grounding

        products = aggregation.build_products(results, request.brand, request.competitors)
        score_breakdown = aggregation.build_score_breakdown(results, products, request.brand, sources)
        visibility_score = aggregation.compute_composite_score(
            score_breakdown, has_sources=bool(sources.source_intel)
        )

        overview = aggregation.build_overview(results, visibility_score)
        overview.failed_count = len(tasks) - len(results)
        overview.top_competitor = aggregation.top_competitor(products, request.brand)

        competitor_diagnosis, opportunities, radar = await generate_insights(
            api_key=anthropic_api_key,
            brand=request.brand,
            industry=request.industry,
            buyer_context=request.buyer_context,
            brand_summary=request.brand_summary,
            market=request.market,
            competitors=request.competitors,
            personas=request.personas,
            results=results,
            products=products,
            sources=sources,
        )

        complete = AnalysisComplete(
            overview=overview,
            products=products,
            sources=sources,
            sitelist=sitelist,
            score_breakdown=score_breakdown,
            competitor_diagnosis=competitor_diagnosis,
            opportunities=opportunities,
            radar=radar,
        )
        job_store.complete(job_id, complete)
        try:
            await get_code_store().finish(job_id)
        except Exception:  # noqa: BLE001 - the client already has the result
            logger.exception("could not mark job %s finished", job_id)
    except Exception as exc:  # noqa: BLE001 - last-resort guard so the SSE stream always terminates
        logger.exception("job %s failed", job_id)
        await _fail(job_store, job_id, user_facing_error(exc))
