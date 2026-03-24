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
    OCRComparisonResult,
    OCRExtractResponse,
    OCRProvenance,
    OCRRawOCRPayload,
    OCRRegionInput,
    OCRRegionResult,
    SaiaFullPageExtractRequest,
    SaiaOCRLocationSuggestion,
    SaiaOCRResponse,
)
from app.db import pipeline_db  # type: ignore[import-untyped]

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


class _FakeStructuredOcrAgent:
    def run(self, payload: Any) -> OCRExtractResponse:
        regions = list(getattr(payload, "regions", []) or [])
        region_id = str(regions[0].region_id) if regions else "r1"
        return OCRExtractResponse(
            status="FULL",
            model="kraken_catmus",
            ocr_backend="kraken_catmus",
            fallbacksUsed=[],
            warnings=[],
            text="Reuerendi\npatri",
            script_hint="latin_medieval",
            final_text="Reuerendi\npatri",
            page_id=getattr(payload, "page_id", None),
            image_id=getattr(payload, "image_id", None),
            fallbacks=[],
            regions=[
                OCRRegionResult(
                    region_id=region_id,
                    text="Reuerendi\npatri",
                    quality=0.88,
                    confidence=0.88,
                    flags=[],
                    bbox_xyxy=[10.0, 20.0, 110.0, 100.0],
                    backend_name="kraken_catmus",
                    model_name="CATMuS Medieval",
                )
            ],
            provenance=OCRProvenance(
                crop_sha256="sha",
                prompt_version="test",
                agent_version="test",
                timestamp="2026-01-01T00:00:00+00:00",
            ),
            raw_ocr=OCRRawOCRPayload(
                lines=["Reuerendi", "patri"],
                text="Reuerendi\npatri",
                script_hint="latin_medieval",
                confidence=0.88,
                warnings=[],
            ),
            comparison_results=[
                OCRComparisonResult(
                    page_id=getattr(payload, "page_id", None),
                    region_id=region_id,
                    backend_name="kraken_catmus",
                    model_name="CATMuS Medieval",
                    text="Reuerendi\npatri",
                    confidence=0.88,
                    selected=True,
                    raw_metadata={"engine": "kraken"},
                ),
                OCRComparisonResult(
                    page_id=getattr(payload, "page_id", None),
                    region_id=region_id,
                    backend_name="saia",
                    model_name="internvl3.5-30b-a3b",
                    text="Reuerendi\npatri",
                    confidence=0.84,
                    selected=False,
                    raw_metadata={"engine": "saia"},
                ),
            ],
            evidence_id="evidence",
            is_evidence=True,
            is_verified=False,
        )


class _FakeStructuredEmptyOcrAgent:
    def run(self, payload: Any) -> OCRExtractResponse:
        regions = list(getattr(payload, "regions", []) or [])
        region_id = str(regions[0].region_id) if regions else "r1"
        return OCRExtractResponse(
            status="FAILED",
            model="kraken_catmus",
            ocr_backend="kraken_catmus",
            fallbacksUsed=["kraken_cremma_lat"],
            warnings=["EMPTY_TEXT:r1"],
            text="",
            script_hint="unknown",
            final_text="",
            page_id=getattr(payload, "page_id", None),
            image_id=getattr(payload, "image_id", None),
            fallbacks=[],
            regions=[
                OCRRegionResult(
                    region_id=region_id,
                    text="",
                    quality=0.0,
                    confidence=0.0,
                    flags=["EMPTY_TEXT"],
                    bbox_xyxy=[10.0, 20.0, 110.0, 100.0],
                    backend_name=None,
                    model_name=None,
                )
            ],
            provenance=OCRProvenance(
                crop_sha256="sha",
                prompt_version="test",
                agent_version="test",
                timestamp="2026-01-01T00:00:00+00:00",
            ),
            raw_ocr=OCRRawOCRPayload(
                lines=[],
                text="",
                script_hint="unknown",
                confidence=0.0,
                warnings=["EMPTY_TEXT:r1"],
            ),
            comparison_results=[
                OCRComparisonResult(
                    page_id=getattr(payload, "page_id", None),
                    region_id=region_id,
                    backend_name="kraken_catmus",
                    model_name="CATMuS Medieval",
                    text="",
                    confidence=0.0,
                    selected=False,
                    raw_metadata={"error": "missing model"},
                ),
                OCRComparisonResult(
                    page_id=getattr(payload, "page_id", None),
                    region_id=region_id,
                    backend_name="saia",
                    model_name="internvl3.5-30b-a3b",
                    text="",
                    confidence=0.0,
                    selected=False,
                    raw_metadata={"error": "SAIA unavailable"},
                ),
            ],
            evidence_id="evidence",
            is_evidence=True,
            is_verified=False,
        )


class _FakeStructuredCompareAgent:
    def run(self, payload: Any) -> OCRExtractResponse:
        regions = list(getattr(payload, "regions", []) or [])
        region_id = str(regions[0].region_id) if regions else "r1"
        return OCRExtractResponse(
            status="FULL",
            model="CATMuS Medieval",
            ocr_backend="kraken_catmus",
            fallbacksUsed=[],
            warnings=[],
            text="kraken line",
            script_hint="latin_medieval",
            final_text="kraken line",
            page_id=getattr(payload, "page_id", None),
            image_id=getattr(payload, "image_id", None),
            fallbacks=[],
            regions=[
                OCRRegionResult(
                    region_id=region_id,
                    text="kraken line",
                    quality=0.82,
                    confidence=0.82,
                    flags=[],
                    bbox_xyxy=[10.0, 20.0, 110.0, 100.0],
                    backend_name="kraken_catmus",
                    model_name="CATMuS Medieval",
                    raw_metadata={"language_hint": "old_french", "script_family": "gothic textualis", "notes": ["recommended manuscript OCR path using segmentation-driven line crops"]},
                )
            ],
            provenance=OCRProvenance(
                crop_sha256="sha",
                prompt_version="test",
                agent_version="test",
                timestamp="2026-01-01T00:00:00+00:00",
            ),
            raw_ocr=OCRRawOCRPayload(
                lines=["kraken line"],
                text="kraken line",
                script_hint="latin_medieval",
                confidence=0.82,
                warnings=[],
            ),
            comparison_results=[
                OCRComparisonResult(
                    page_id=getattr(payload, "page_id", None),
                    region_id=region_id,
                    backend_name="calamari",
                    model_name="Calamari historical_french",
                    text="calamari line",
                    confidence=0.51,
                    selected=False,
                    raw_metadata={"language_hint": "old_french", "script_family": "gothic textualis", "notes": ["historical print baseline"]},
                ),
            ],
            evidence_id="evidence",
            is_evidence=True,
            is_verified=False,
        )


def test_extract_full_page_returns_required_keys(monkeypatch: Any) -> None:
    monkeypatch.setattr(ocr_router, "_get_ocr_agent", lambda: _FakeStructuredOcrAgent())

    payload = SaiaFullPageExtractRequest(
        document_id="doc-1",
        page_id="page-1",
        image_b64=SAMPLE_B64,
        apply_proofread=True,
        location_suggestions=[
            SaiaOCRLocationSuggestion(
                region_id="r1",
                category="Main script black",
                bbox_xywh=[10.0, 20.0, 100.0, 80.0],
            )
        ],
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
    assert "Ask ArchAI about this page..." in source


def test_runtime_ocr_router_imports_without_extra_ocr_dependencies() -> None:
    from app.routers import ocr as runtime_ocr_router  # type: ignore[import-untyped]

    assert hasattr(runtime_ocr_router, "ocr_extract_full_page")


def test_extract_full_page_compare_all_returns_three_backends(monkeypatch: Any) -> None:
    monkeypatch.setattr(ocr_router, "_get_ocr_agent", lambda: _FakeStructuredCompareAgent())
    monkeypatch.setattr(
        ocr_router,
        "_run_glmocr_full_page_summary",
        lambda payload, image_bytes, regions, selected=False: ocr_router.OCRComparisonSummary(
            backend_name="glmocr",
            model_name="GLM-OCR-0.9B",
            selected=selected,
            text="glm page text",
            lines=["glm page text"],
            confidence=None,
            warnings=[],
            language_hint="old_french",
            script_family="gothic textualis",
            notes=["experimental document-level OCR"],
        ),
    )

    payload = SaiaFullPageExtractRequest(
        document_id="doc-1",
        page_id="page-1",
        image_b64=SAMPLE_B64,
        apply_proofread=False,
        ocr_backend="auto",
        compare_backends=["calamari", "glmocr"],
        language_hint="old_french",
        regions=[OCRRegionInput(region_id="r1", bbox_xyxy=[10.0, 20.0, 110.0, 100.0], label="Main script black", reading_order=0)],
    )

    result = asyncio.run(ocr_router.ocr_extract_full_page(payload))

    assert len(result.comparison_runs) == 3
    assert {item.backend_name for item in result.comparison_runs} == {"kraken_family", "calamari", "glmocr"}
    assert any(item.selected for item in result.comparison_runs)


def test_page_with_trace_segmented_path_persists_backend_comparisons(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(ocr_router, "_get_ocr_agent", lambda: _FakeStructuredOcrAgent())
    monkeypatch.setattr(ocr_router, "_run_trace_analysis", lambda run_id, text: ([], [], [], {"skipped": True}))
    monkeypatch.setattr(
        ocr_router,
        "check_mention_recall",
        lambda *args, **kwargs: {"mention_recall_ok": True, "reason": "ok", "trigger_high_recall": False},
    )
    monkeypatch.setattr(ocr_router, "_run_authority_linking_stage", lambda run_id: None)
    monkeypatch.setattr(ocr_router, "_auto_index_run", lambda run_id: None)
    monkeypatch.setattr(ocr_router, "_build_mention_extraction_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(ocr_router, "_build_consolidated_report", lambda *args, **kwargs: None)

    payload = SaiaFullPageExtractRequest(
        document_id="doc-1",
        page_id="page-1",
        image_b64=SAMPLE_B64,
        apply_proofread=True,
        location_suggestions=[
            SaiaOCRLocationSuggestion(
                region_id="r1",
                category="Main script black",
                bbox_xywh=[10.0, 20.0, 100.0, 80.0],
            )
        ],
        compare_backends=["saia"],
    )

    result = asyncio.run(ocr_router.ocr_page_with_trace(payload))

    assert result["status"] == "COMPLETED"
    assert len(result["ocr_backend_results"]) == 2
    stored = pipeline_db.list_ocr_backend_results(str(result["run_id"]))
    assert len(stored) == 2
    assert {item["backend_name"] for item in stored} == {"kraken_catmus", "saia"}


def test_workspace_popup_copy_documents_engine_roles() -> None:
    workspace = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "src"
        / "components"
        / "workspace"
        / "DocumentChatWorkspace.tsx"
    )
    source = workspace.read_text(encoding="utf-8")
    assert "Kraken family" in source
    assert "Best manuscript OCR path in this repo using segmented line crops and medieval-trained models." in source
    assert "Calamari" in source
    assert "Historical print baseline" in source
    assert "Not the best default for medieval handwritten manuscripts." in source
    assert "GLM-OCR" in source
    assert "General multimodal OCR for full-page or column-level document extraction." in source
    assert "Compare all three" in source
    assert "Auto recommendation:" in source


def test_page_with_trace_blocks_low_trust_segmented_ocr(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(ocr_router, "_get_ocr_agent", lambda: _FakeStructuredEmptyOcrAgent())

    payload = SaiaFullPageExtractRequest(
        document_id="doc-1",
        page_id="page-1",
        image_b64=SAMPLE_B64,
        apply_proofread=False,
        language_hint="latin",
        location_suggestions=[
            SaiaOCRLocationSuggestion(
                region_id="r1",
                category="Main script black",
                bbox_xywh=[10.0, 20.0, 100.0, 80.0],
            )
        ],
        compare_backends=["saia"],
    )

    result = asyncio.run(ocr_router.ocr_page_with_trace(payload))

    assert result["status"] == "FAILED_QUALITY"
    assert set(result["quality_gates"]["blocked_stages"]) >= {
        "translation",
        "paraphrase",
        "entity_claims",
        "no_entity_claims",
    }
    assert len(result["ocr_backend_results"]) == 2
