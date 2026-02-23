from __future__ import annotations

from fastapi import APIRouter, Query

from app.routers.ocr import ocr_extract
from app.schemas.agents_ocr import OCRExtractAnyResponse, OCRExtractRequest

router = APIRouter(tags=["agents", "ocr"])


@router.post("/agents/ocr", response_model=OCRExtractAnyResponse)
async def agents_ocr(
    payload: OCRExtractRequest,
    mode: str | None = Query(default=None, pattern="^(full|simple)$"),
) -> OCRExtractAnyResponse:
    return await ocr_extract(payload, mode=mode)


@router.post("/agents/ocr/extract", response_model=OCRExtractAnyResponse)
async def agents_ocr_extract(
    payload: OCRExtractRequest,
    mode: str | None = Query(default=None, pattern="^(full|simple)$"),
) -> OCRExtractAnyResponse:
    return await ocr_extract(payload, mode=mode)
