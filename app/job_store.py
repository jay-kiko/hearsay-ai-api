"""Map<jobId, Job> · in memory, TTL-evicted, gone on restart.

This intentionally is not a database (see design doc §07 "In-memory job
store"): nothing to provision or migrate for an MVP, and it matches the "no
persistence" story honestly. The Anthropic API key is forwarded per-call and
is never written into a Job.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from app.models import AnalysisComplete, JobStatus, PersonaEvent, PersonaStatus

# A None on a subscriber queue signals "stream is done, close the connection".
_SENTINEL = None


def new_job_id() -> str:
    return f"j_{uuid.uuid4().hex[:10]}"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.running
    personas: dict[str, PersonaEvent] = field(default_factory=dict)
    result: AnalysisComplete | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    task: asyncio.Task | None = None


class JobStore:
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._jobs: dict[str, Job] = {}

    def create_job(self, persona_ids: list[str], job_id: str | None = None) -> Job:
        job_id = job_id or new_job_id()
        job = Job(job_id=job_id, personas={pid: PersonaEvent(persona_id=pid, status="waiting") for pid in persona_ids})
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def attach_task(self, job_id: str, task: asyncio.Task) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.task = task

    def _broadcast(self, job: Job, event_name: str, data: str) -> None:
        payload = f"event: {event_name}\ndata: {data}\n\n"
        for queue in job.subscribers:
            queue.put_nowait(payload)

    def update_persona(self, job_id: str, event: PersonaEvent) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.personas[event.persona_id] = event
        self._broadcast(job, "persona", event.model_dump_json(by_alias=True))

    def set_persona_status(self, job_id: str, persona_id: str, status: PersonaStatus) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        existing = job.personas.get(persona_id)
        event = PersonaEvent(persona_id=persona_id, status=status, result=existing.result if existing else None)
        self.update_persona(job_id, event)

    def complete(self, job_id: str, result: AnalysisComplete) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.status = JobStatus.complete
        job.result = result
        self._broadcast(job, "complete", result.model_dump_json(by_alias=True))
        for queue in job.subscribers:
            queue.put_nowait(_SENTINEL)

    def fail(self, job_id: str, message: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.status = JobStatus.error
        job.error = message
        self._broadcast(job, "error", f'{{"message": {message!r}}}')
        for queue in job.subscribers:
            queue.put_nowait(_SENTINEL)

    def subscribe(self, job_id: str) -> asyncio.Queue | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        queue: asyncio.Queue = asyncio.Queue()
        job.subscribers.append(queue)

        # Replay whatever's already known so a subscriber that connects a beat
        # late (or reconnects) still sees the full picture.
        for event in job.personas.values():
            if event.status in ("done", "error"):
                queue.put_nowait(f"event: persona\ndata: {event.model_dump_json(by_alias=True)}\n\n")
        if job.status == JobStatus.complete and job.result is not None:
            queue.put_nowait(f"event: complete\ndata: {job.result.model_dump_json(by_alias=True)}\n\n")
            queue.put_nowait(_SENTINEL)
        elif job.status == JobStatus.error:
            queue.put_nowait(f'event: error\ndata: {{"message": {job.error!r}}}\n\n')
            queue.put_nowait(_SENTINEL)

        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        job = self._jobs.get(job_id)
        if job and queue in job.subscribers:
            job.subscribers.remove(queue)

    async def evict_loop(self, interval_seconds: int = 60) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            now = time.monotonic()
            # A job whose task is still running is never evicted, even past
            # its TTL — deleting it mid-flight would leave its SSE stream
            # hanging forever, since complete()/fail() silently no-op on a
            # missing job. It's swept on the next pass once the task finishes.
            expired = [
                jid
                for jid, job in self._jobs.items()
                if now - job.created_at > self._ttl_seconds and (job.task is None or job.task.done())
            ]
            for jid in expired:
                del self._jobs[jid]


_store: JobStore | None = None


def get_job_store() -> JobStore:
    global _store
    if _store is None:
        from app.config import get_settings

        _store = JobStore(ttl_seconds=get_settings().job_ttl_seconds)
    return _store
