from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class OcrEvidenceRecord:
    created_at: str
    image_sha256: str
    image_ref: str
    model: str
    decoding: dict
    prompt_version: str
    pipeline_version: str
    script_hint: str
    confidence: float
    warnings: list[str]
    raw_ocr_text: str
    final_text: str | None = None
    final_changes: list[str] | None = None


def write_ocr_evidence_jsonl(
    record: OcrEvidenceRecord,
    out_path: str = "data/evidence/ocr_evidence.jsonl",
) -> None:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
