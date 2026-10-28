from fastapi import APIRouter

from app.config import get_settings
from app.models import DetectRequest, DetectResponse
from app.routes._access import charge_generation_call, upstream_errors
from app.services.detect import detect_brand

router = APIRouter()


@router.post("/api/detect", response_model=DetectResponse)
async def detect(body: DetectRequest) -> DetectResponse:
    await charge_generation_call(body.access_code)
    with upstream_errors():
        return await detect_brand(api_key=get_settings().anthropic_api_key, query=body.query)
