import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.code_store import get_code_store
from app.config import get_settings
from app.job_store import get_job_store
from app.routes.admin import router as admin_router
from app.routes.analysis import router as analysis_router
from app.routes.prompts import router as prompts_router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_code_store().connect()
    eviction_task = asyncio.create_task(get_job_store().evict_loop())
    try:
        yield
    finally:
        eviction_task.cancel()
        await get_code_store().close()


settings = get_settings()

app = FastAPI(title="hearsay.ai backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prompts_router)
app.include_router(analysis_router)
app.include_router(admin_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
