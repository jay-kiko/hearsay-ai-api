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

from app.config import get_settings
from app.job_store import JobStore
from app.models import AnalysisComplete, AnalysisRequest, PersonaEvent
from app.services import aggregation
from app.services.grounding import run_grounding
from app.services.pipeline import run_persona

logger = logging.getLogger("hearsay.runner")


async def _run_one_persona(
    *,
    semaphore: asyncio.Semaphore,
    job_store: JobStore,
    job_id: str,
    persona,
    prompts,
    brand,
    competitors,
    buyer_context,
    market,
    api_key,
):
    job_store.set_persona_status(job_id, persona.id, "running")
    async with semaphore:
        try:
            result = await run_persona(
                persona=persona,
                prompts=prompts,
                brand=brand,
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
                job_id, PersonaEvent(persona_id=persona.id, status="error", error=str(exc))
            )
            return persona.id, None


async def run_job(job_id: str, request: AnalysisRequest, job_store: JobStore, anthropic_api_key: str) -> None:
    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.persona_concurrency)

    try:
        tasks = [
            _run_one_persona(
                semaphore=semaphore,
                job_store=job_store,
                job_id=job_id,
                persona=persona,
                prompts=request.prompts.get(persona.id) or [],
                brand=request.brand,
                competitors=request.competitors,
                buyer_context=request.buyer_context,
                market=request.market,
                api_key=anthropic_api_key,
            )
            for persona in request.personas
            if request.prompts.get(persona.id)
        ]

        pairs = await asyncio.gather(*tasks) if tasks else []
        results = {pid: result for pid, result in pairs if result is not None}

        if not tasks:
            job_store.fail(job_id, "No personas had prompts to analyze")
            return
        if not results:
            job_store.fail(job_id, "All persona calls failed — see per-persona errors for details")
            return

        sources, sitelist = await run_grounding(
            api_key=anthropic_api_key,
            brand=request.brand,
            industry=request.industry,
            competitors=request.competitors,
            market=request.market,
        )

        overview = aggregation.build_overview(results)
        overview.failed_count = len(tasks) - len(results)
        products = aggregation.build_products(results, request.brand, request.competitors)
        overview.top_competitor = aggregation.top_competitor(products, request.brand)

        complete = AnalysisComplete(overview=overview, products=products, sources=sources, sitelist=sitelist)
        job_store.complete(job_id, complete)
    except Exception as exc:  # noqa: BLE001 - last-resort guard so the SSE stream always terminates
        logger.exception("job %s failed", job_id)
        job_store.fail(job_id, str(exc))
