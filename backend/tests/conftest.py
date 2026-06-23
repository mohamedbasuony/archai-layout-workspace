from __future__ import annotations

import sys
from pathlib import Path

import pytest

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents import ocr_agent  # type: ignore[import-untyped]
from app.services.ocr_evidence import OcrEvidenceRecord, write_ocr_evidence_jsonl  # type: ignore[import-untyped]


@pytest.fixture(autouse=True)
def isolate_ocr_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep OCR test evidence out of the tracked demonstration dataset."""
    evidence_path = tmp_path / "ocr_evidence.jsonl"

    def write_test_evidence(record: OcrEvidenceRecord, *_args: object, **_kwargs: object) -> None:
        write_ocr_evidence_jsonl(record, out_path=str(evidence_path))

    monkeypatch.setattr(ocr_agent, "write_ocr_evidence_jsonl", write_test_evidence)
