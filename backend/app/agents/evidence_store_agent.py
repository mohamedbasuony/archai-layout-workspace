from __future__ import annotations

from datetime import datetime, timezone
import secrets

from app.schemas.agents_ocr import EvidenceSpanCreateRequest, EvidenceSpanRecord

_EVIDENCE_SPANS: dict[str, list[EvidenceSpanRecord]] = {}


def create_span(payload: EvidenceSpanCreateRequest) -> EvidenceSpanRecord:
    record = EvidenceSpanRecord(
        span_id=f"span-{secrets.token_hex(8)}",
        page_id=payload.page_id,
        region_id=payload.region_id,
        text=payload.text,
        bbox_xyxy=payload.bbox_xyxy,
        polygon=payload.polygon,
        model_used=payload.model_used,
        prompt_version=payload.prompt_version,
        crop_sha256=payload.crop_sha256,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _EVIDENCE_SPANS.setdefault(payload.page_id, []).append(record)
    return record


def list_spans(page_id: str) -> list[EvidenceSpanRecord]:
    return list(_EVIDENCE_SPANS.get(page_id, []))
