import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.code_store import get_code_store
from app.config import get_settings
from app.job_store import get_job_store
from app.routes.access import router as access_router
from app.routes.admin import router as admin_router
from app.routes.analysis import router as analysis_router
from app.routes.categories import router as categories_router
from app.routes.detect import router as detect_router
from app.routes.personas import router as personas_router
from app.routes.prompts import router as prompts_router
from app.services.anthropic_client import close_clients

logging.basicConfig(level=logging.INFO)


async def _refund_lost_jobs_loop(interval_seconds: int = 300) -> None:
    """Jobs live in memory, so one lost to a restart or crash never refunds
    itself. Swept periodically (not just at startup) by age, so a rolling
    deploy's overlap never refunds jobs the other process is still running."""
    logger = logging.getLogger("hearsay")
    while True:
        try:
            refunded = await get_code_store().refund_stale(
                get_settings().stale_job_seconds, get_job_store().running_job_ids()
            )
            if refunded:
                logger.warning("refunded %d use(s) for jobs that were lost", refunded)
        except Exception:  # noqa: BLE001 - keep sweeping
            logger.exception("refund sweep failed")
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_code_store().connect()
    eviction_task = asyncio.create_task(get_job_store().evict_loop())
    refund_task = asyncio.create_task(_refund_lost_jobs_loop())
    try:
        yield
    finally:
        eviction_task.cancel()
        refund_task.cancel()
        await close_clients()
        await get_code_store().close()


settings = get_settings()

app = FastAPI(title="hearsay.ai backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    # No cookies or auth headers are used (codes travel in the body), so
    # credentialed cross-origin requests are not allowed.
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Admin-Secret"],
)

app.include_router(access_router)
app.include_router(categories_router)
app.include_router(detect_router)
app.include_router(personas_router)
app.include_router(prompts_router)
app.include_router(analysis_router)
app.include_router(admin_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
