from __future__ import annotations

from fastapi import APIRouter

from app.agents.evidence_store_agent import create_span, list_spans
from app.schemas.agents_ocr import EvidenceSpanCreateRequest, EvidenceSpanRecord

router = APIRouter(tags=["evidence"])


@router.post("/evidence/spans", response_model=EvidenceSpanRecord)
async def create_evidence_span(payload: EvidenceSpanCreateRequest) -> EvidenceSpanRecord:
    return create_span(payload)


@router.get("/evidence/spans/{page_id}", response_model=list[EvidenceSpanRecord])
async def get_evidence_spans(page_id: str) -> list[EvidenceSpanRecord]:
    return list_spans(page_id)
