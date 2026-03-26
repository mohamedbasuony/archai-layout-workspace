from __future__ import annotations

import json
from typing import Any

from app.db import pipeline_db
from app.schemas.agents_ocr import EvidenceSpanCreateRequest, EvidenceSpanRecord


def _record_from_row(row: dict[str, Any]) -> EvidenceSpanRecord:
    bbox_value = row.get("bbox_json")
    polygon_value = row.get("polygon_json")
    meta_value = row.get("meta_json")
    if isinstance(meta_value, str):
        try:
            meta_value = json.loads(meta_value)
        except Exception:
            meta_value = {}
    elif not isinstance(meta_value, dict):
        meta_value = {}
    return EvidenceSpanRecord(
        span_id=str(row.get("span_id") or ""),
        run_id=str(row.get("run_id") or ""),
        asset_ref=row.get("asset_ref"),
        page_id=row.get("page_id"),
        region_id=row.get("region_id") or meta_value.get("region_id"),
        chunk_id=row.get("chunk_id"),
        mention_id=row.get("mention_id"),
        text=str(row.get("raw_text") or row.get("text") or ""),
        raw_text=str(row.get("raw_text") or row.get("text") or ""),
        normalized_text=row.get("normalized_text"),
        start_offset=row.get("start_offset"),
        end_offset=row.get("end_offset"),
        bbox_xyxy=bbox_value if isinstance(bbox_value, list) else None,
        polygon=polygon_value if isinstance(polygon_value, list) else None,
        model_used=row.get("ocr_model") or row.get("model_used"),
        ocr_model=row.get("ocr_model") or row.get("model_used"),
        prompt_version=row.get("prompt_version"),
        crop_sha256=row.get("crop_sha256"),
        meta_json=meta_value or None,
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or row.get("created_at") or ""),
    )


def create_span(payload: EvidenceSpanCreateRequest) -> EvidenceSpanRecord:
    meta_json = dict(payload.meta_json or {})
    if payload.region_id and "region_id" not in meta_json:
        meta_json["region_id"] = payload.region_id
    row = pipeline_db.upsert_evidence_span(
        {
            "run_id": payload.run_id,
            "asset_ref": payload.asset_ref,
            "page_id": payload.page_id,
            "region_id": payload.region_id,
            "chunk_id": payload.chunk_id,
            "mention_id": payload.mention_id,
            "raw_text": payload.raw_text or payload.text,
            "normalized_text": payload.normalized_text,
            "start_offset": payload.start_offset,
            "end_offset": payload.end_offset,
            "bbox_xyxy": payload.bbox_xyxy,
            "polygon": payload.polygon,
            "ocr_model": payload.ocr_model or payload.model_used,
            "prompt_version": payload.prompt_version,
            "crop_sha256": payload.crop_sha256,
            "meta_json": meta_json or None,
        }
    )
    return _record_from_row(row)


def list_spans(page_id: str) -> list[EvidenceSpanRecord]:
    rows = pipeline_db.list_evidence_spans(page_id=page_id)
    return [_record_from_row(row) for row in rows]
