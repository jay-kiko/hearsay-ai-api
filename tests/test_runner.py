"""Job orchestration: grounding overlaps the persona phase, and a job that
delivers nothing stops its grounding and refunds the use."""
import asyncio

import pytest

from app import runner
from app.job_store import JobStore
from app.models import AnalysisRequest, PersonaResult, Sentiment, Sources


def _request():
    persona = {"id": "p1", "title": "T", "role": "r", "pains": "p", "criteria": "c"}
    return AnalysisRequest(brand="B", industry="i", accessCode="AAAA-BBBB", personas=[persona], prompts={"p1": ["q"]})


class _Refunds:
    def __init__(self):
        self.refunded, self.finished = [], []

    async def refund(self, job_id):
        self.refunded.append(job_id)

    async def finish(self, job_id):
        self.finished.append(job_id)


@pytest.fixture
def store(monkeypatch):
    codes = _Refunds()
    monkeypatch.setattr(runner, "get_code_store", lambda: codes)
    return codes


async def test_grounding_starts_before_personas_finish(monkeypatch, store):
    order = []
    persona_gate = asyncio.Event()

    async def slow_persona(**_):
        order.append("persona start")
        await persona_gate.wait()
        return PersonaResult(prompt="q", mentioned=False, sentiment=Sentiment.neutral, vis=0, rank=None, quote="",
                             parts=[], exchanges=[], opportunity="Critical Gap")

    async def grounding(**_):
        order.append("grounding start")
        persona_gate.set()
        return Sources(citations=[], publishers=[], communities=[]), []

    async def insights(**_):
        from app.models import CompetitorDiagnosis
        return CompetitorDiagnosis(rival_wins=[], brand_wins=[], gaps=[]), [], []

    monkeypatch.setattr(runner, "run_persona", slow_persona)
    monkeypatch.setattr(runner, "run_grounding", grounding)
    monkeypatch.setattr(runner, "generate_insights", insights)
    jobs = JobStore(ttl_seconds=60)
    job = jobs.create_job(["p1"])
    await asyncio.wait_for(runner.run_job(job.job_id, _request(), jobs, "k"), timeout=2)
    assert "grounding start" in order and jobs.get(job.job_id).status.value == "complete"
    assert store.finished == [job.job_id] and not store.refunded


async def test_all_personas_failing_cancels_grounding_and_refunds(monkeypatch, store):
    cancelled = asyncio.Event()

    async def failing_persona(**_):
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    async def grounding(**_):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(runner, "run_persona", failing_persona)
    monkeypatch.setattr(runner, "run_grounding", grounding)
    jobs = JobStore(ttl_seconds=60)
    job = jobs.create_job(["p1"])
    await runner.run_job(job.job_id, _request(), jobs, "k")
    await asyncio.sleep(0)
    assert cancelled.is_set()
    assert jobs.get(job.job_id).status.value == "error" and store.refunded == [job.job_id]


async def test_a_grounding_error_still_delivers_the_persona_results(monkeypatch, store):
    async def ok_persona(**_):
        return PersonaResult(prompt="q", mentioned=False, sentiment=Sentiment.neutral, vis=0, rank=None, quote="",
                             parts=[], exchanges=[], opportunity="Critical Gap")

    async def broken_grounding(**_):
        raise KeyError("malformed search block")

    async def insights(**_):
        from app.models import CompetitorDiagnosis
        return CompetitorDiagnosis(rival_wins=[], brand_wins=[], gaps=[]), [], []

    monkeypatch.setattr(runner, "run_persona", ok_persona)
    monkeypatch.setattr(runner, "run_grounding", broken_grounding)
    monkeypatch.setattr(runner, "generate_insights", insights)
    jobs = JobStore(ttl_seconds=60)
    job = jobs.create_job(["p1"])
    await runner.run_job(job.job_id, _request(), jobs, "k")
    done = jobs.get(job.job_id)
    assert done.status.value == "complete"
    assert done.result.sources.coverage_note.startswith("Sources couldn't be researched")
    assert store.finished == [job.job_id] and not store.refunded
