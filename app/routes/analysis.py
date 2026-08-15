import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.code_store import get_code_store
from app.config import get_settings
from app.job_store import get_job_store, new_job_id
from app.models import AnalysisRequest, AnalysisStartResponse, JobSnapshot
from app.runner import run_job

router = APIRouter()


@router.post("/api/analysis", response_model=AnalysisStartResponse)
async def start_analysis(body: AnalysisRequest) -> AnalysisStartResponse:
    if not body.personas:
        raise HTTPException(status_code=400, detail="At least one persona is required")

    code_store = get_code_store()
    status = await code_store.status(body.access_code)
    if status == "unknown":
        raise HTTPException(status_code=404, detail="Access code not found")
    if status == "revoked":
        raise HTTPException(status_code=403, detail="This access code has been revoked")
    if status == "exhausted":
        raise HTTPException(status_code=403, detail="This access code has no uses remaining")

    # Reserve the job id and spend one use against it atomically before
    # anything else exists, so two concurrent requests can't both win the
    # same use.
    job_id = new_job_id()
    if not await code_store.redeem(body.access_code, job_id):
        raise HTTPException(status_code=403, detail="This access code has no uses remaining")

    job_store = get_job_store()
    job = job_store.create_job([p.id for p in body.personas], job_id=job_id)

    task = asyncio.create_task(run_job(job.job_id, body, job_store, get_settings().anthropic_api_key))
    job_store.attach_task(job.job_id, task)

    return AnalysisStartResponse(job_id=job.job_id)


@router.get("/api/analysis/{job_id}/stream")
async def stream_analysis(job_id: str, request: Request) -> StreamingResponse:
    job_store = get_job_store()
    queue = job_store.subscribe(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    break
                yield item
        finally:
            job_store.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/api/analysis/{job_id}", response_model=JobSnapshot)
async def get_analysis(job_id: str) -> JobSnapshot:
    job_store = get_job_store()
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobSnapshot(
        job_id=job.job_id,
        status=job.status,
        personas=job.personas,
        result=job.result,
        error=job.error,
    )
