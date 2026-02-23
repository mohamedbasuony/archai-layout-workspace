from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.routers import ocr as ocr_router  # type: ignore[import-untyped]
from app.schemas.agents_ocr import (  # type: ignore[import-untyped]
    SaiaFullPageExtractRequest,
    SaiaOCRLocationSuggestion,
    SaiaOCRResponse,
)

SAMPLE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mNk+M/wHwAE/wJ/lxNn4QAAAABJRU5ErkJggg=="


class _FakeSaiaAgent:
    def extract(self, payload: Any) -> SaiaOCRResponse:
        _ = payload
        return SaiaOCRResponse(
            status="FULL",
            model_used="qwen3-vl-30b-a3b-instruct",
            fallbacks=[],
            fallbacks_used=[],
            warnings=[],
            lines=["Reuerendi", "patri"],
            text="Reuerendi\npatri",
            script_hint="latin",
            detected_language="latin",
            confidence=0.88,
            raw_json={
                "lines": ["Reuerendi", "patri"],
                "text": "Reuerendi\npatri",
                "script_hint": "latin",
                "detected_language": "latin",
                "confidence": 0.88,
                "warnings": [],
            },
        )


def test_extract_full_page_returns_required_keys(monkeypatch: Any) -> None:
    monkeypatch.setattr(ocr_router, "_get_saia_ocr_agent", lambda: _FakeSaiaAgent())
    monkeypatch.setattr(
        ocr_router,
        "_run_segmentation_for_suggestions",
        lambda _image_bytes: [
            SaiaOCRLocationSuggestion(
                region_id="r1",
                category="Main script black",
                bbox_xywh=[10.0, 20.0, 100.0, 80.0],
            )
        ],
    )

    payload = SaiaFullPageExtractRequest(
        document_id="doc-1",
        page_id="page-1",
        image_b64=SAMPLE_B64,
        apply_proofread=True,
    )
    result = asyncio.run(ocr_router.ocr_extract_full_page(payload))

    dumped = result.model_dump()
    for key in (
        "status",
        "model_used",
        "fallbacks_used",
        "detected_language",
        "language_confidence",
        "script_hint",
        "confidence",
        "warnings",
        "lines",
        "text",
    ):
        assert key in dumped

    assert result.status in {"FULL", "PARTIAL", "EMPTY"}


def test_workspace_contains_detected_language_line() -> None:
    workspace = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "src"
        / "components"
        / "workspace"
        / "DocumentChatWorkspace.tsx"
    )
    source = workspace.read_text(encoding="utf-8")
    assert "Detected language:" in source


def test_runtime_ocr_router_imports_without_extra_ocr_dependencies() -> None:
    from app.routers import ocr as runtime_ocr_router  # type: ignore[import-untyped]

    assert hasattr(runtime_ocr_router, "ocr_extract_full_page")
