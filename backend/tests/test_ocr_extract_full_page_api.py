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
    OCRDocumentMetadata,
    OCRExtractResponse,
    OCRProvenance,
    OCRRawOCRPayload,
    OCRRegionInput,
    OCRRegionResult,
    SaiaFullPageExtractRequest,
    SaiaFullPageExtractResponse,
    SaiaOCRLocationSuggestion,
    SaiaOCRResponse,
)
from app.db import pipeline_db  # type: ignore[import-untyped]
from app.services.glm_ollama_ocr import GlmOllamaOcrResult  # type: ignore[import-untyped]

SAMPLE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mNk+M/wHwAE/wJ/lxNn4QAAAABJRU5ErkJggg=="


def _fake_glm_result(
    text: str = "Reuerendi\npatri",
    warnings: list[str] | None = None,
    *,
    processed_variant_name: str = "rgb_jpeg_1280",
    attempts_used: int = 1,
) -> GlmOllamaOcrResult:
    lines = [line for line in text.splitlines() if line.strip()]
    return GlmOllamaOcrResult(
        text=text,
        lines=lines,
        model_used="glm-ocr:latest",
        warnings=list(warnings or []),
        original_size_bytes=912345,
        original_width=2400,
        original_height=1600,
        processed_width=1280,
        processed_height=853,
        processed_size_bytes=345678,
        preprocessing_applied=True,
        processed_variant_name=processed_variant_name,
        attempts_used=attempts_used,
        duration_seconds=1.25,
    )


def test_extract_salvage_mentions_rejects_weak_lexical_place_candidates() -> None:
    mentions, debug = ocr_router._extract_salvage_mentions("Li parol furent dites de vilanie et en enfant.")

    surfaces = {str(item.get("surface") or "").lower() for item in mentions}
    assert "vilanie" not in surfaces
    assert "enfant" not in surfaces
    rejected = [str(item.get("reason") or "") for item in debug.get("rejected", [])]
    assert any("weak_place_lexical_blacklist" in reason for reason in rejected)


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
    monkeypatch.setattr(ocr_router, "run_glm_ollama_ocr", lambda *args, **kwargs: _fake_glm_result())
    monkeypatch.setattr(ocr_router, "_run_post_ocr_pipeline_for_glm", lambda *args, **kwargs: {})

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
        "original_image_size_bytes",
        "processed_image_size_bytes",
        "processed_image_width",
        "processed_image_height",
        "processed_variant_name",
        "ocr_attempts_used",
    ):
        assert key in dumped

    assert result.status in {"FULL", "PARTIAL", "EMPTY"}
    assert result.model_used == "glm-ocr:latest"
    assert result.original_image_size_bytes == 912345
    assert result.processed_image_size_bytes == 345678
    assert result.processed_variant_name == "rgb_jpeg_1280"
    assert result.ocr_attempts_used == 1
    assert len(result.comparison_runs) == 1
    assert result.comparison_runs[0].backend_name == "glmocr"


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


def test_extract_full_page_ignores_compare_backends_and_returns_glm_run(monkeypatch: Any) -> None:
    monkeypatch.setattr(ocr_router, "run_glm_ollama_ocr", lambda *args, **kwargs: _fake_glm_result("glm page text"))
    monkeypatch.setattr(ocr_router, "_run_post_ocr_pipeline_for_glm", lambda *args, **kwargs: {})

    payload = SaiaFullPageExtractRequest(
        document_id="doc-1",
        page_id="page-1",
        image_b64=SAMPLE_B64,
        apply_proofread=False,
        ocr_backend="glmocr",
        compare_backends=["calamari", "glmocr"],
        language_hint="old_french",
    )

    result = asyncio.run(ocr_router.ocr_extract_full_page(payload))

    assert len(result.comparison_runs) == 1
    assert result.comparison_runs[0].backend_name == "glmocr"
    assert result.comparison_runs[0].selected is True
    assert "COMPARE_BACKENDS_IGNORED" in result.warnings


def test_extract_full_page_surfaces_run_and_knowledge_pipeline_fields(monkeypatch: Any) -> None:
    monkeypatch.setattr(ocr_router, "run_glm_ollama_ocr", lambda *args, **kwargs: _fake_glm_result("glm page text"))
    monkeypatch.setattr(
        ocr_router,
        "_run_post_ocr_pipeline_for_glm",
        lambda *args, **kwargs: {
            "run_id": "run-123",
            "quality_label": "OK",
            "downstream_mode": "full",
            "chunks_count": 4,
            "mentions_count": 2,
            "authority_report": "=== ENTITY LINKING REPORT ===",
            "mention_report": "=== MENTION EXTRACTION REPORT ===",
            "consolidated_report": "=== CONSOLIDATED REPORT ===",
            "warnings": [],
        },
    )

    payload = SaiaFullPageExtractRequest(
        document_id="doc-1",
        page_id="page-1",
        image_b64=SAMPLE_B64,
        apply_proofread=False,
        ocr_backend="glmocr",
    )

    result = asyncio.run(ocr_router.ocr_extract_full_page(payload))

    assert result.run_id == "run-123"
    assert result.quality_label == "OK"
    assert result.downstream_mode == "full"
    assert result.chunks_count == 4
    assert result.mentions_count == 2
    assert result.authority_report == "=== ENTITY LINKING REPORT ==="
    assert result.consolidated_report == "=== CONSOLIDATED REPORT ==="


def test_extract_full_page_builds_glm_prompt_from_metadata(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_glm_ollama_ocr(*args: Any, **kwargs: Any) -> GlmOllamaOcrResult:
        captured["prompt"] = kwargs.get("prompt")
        captured["image_ref"] = kwargs.get("image_ref")
        return _fake_glm_result("glm page text")

    monkeypatch.setattr(ocr_router, "run_glm_ollama_ocr", _fake_run_glm_ollama_ocr)
    monkeypatch.setattr(ocr_router, "_run_post_ocr_pipeline_for_glm", lambda *args, **kwargs: {})

    payload = SaiaFullPageExtractRequest(
        document_id="doc-1",
        page_id="page-1",
        image_b64=SAMPLE_B64,
        apply_proofread=False,
        ocr_backend="glmocr",
        script_hint_seed="latin",
        language_hint="middle_french",
        metadata=OCRDocumentMetadata(
            language="Middle French",
            year="1450",
            place_or_origin="Rouen",
            script_family="Gothic cursiva",
            document_type="Charter",
            notes="Abbreviations are frequent.",
        ),
    )

    result = asyncio.run(ocr_router.ocr_extract_full_page(payload))

    assert result.model_used == "glm-ocr:latest"
    assert captured["image_ref"] == "page-1"
    assert "Likely manuscript language: Middle French; middle_french" in str(captured["prompt"])
    assert "Approximate manuscript date or year: 1450" in str(captured["prompt"])
    assert "Script hint: latin" in str(captured["prompt"])
    assert "Place or origin: Rouen" in str(captured["prompt"])
    assert "Document type: Charter" in str(captured["prompt"])


def test_extract_full_page_surfaces_variant_retry_details(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        ocr_router,
        "run_glm_ollama_ocr",
        lambda *args, **kwargs: _fake_glm_result(
            "glm page text",
            warnings=["OCR_PAYLOAD_RESIZED", "OCR_VARIANT_FALLBACK:rgb_autocontrast_jpeg_1280", "OCR_RETRY_ATTEMPTS_USED:2"],
            processed_variant_name="rgb_autocontrast_jpeg_1280",
            attempts_used=2,
        ),
    )
    monkeypatch.setattr(ocr_router, "_run_post_ocr_pipeline_for_glm", lambda *args, **kwargs: {})

    payload = SaiaFullPageExtractRequest(
        document_id="doc-1",
        page_id="page-1",
        image_b64=SAMPLE_B64,
        apply_proofread=False,
        ocr_backend="glmocr",
    )

    result = asyncio.run(ocr_router.ocr_extract_full_page(payload))

    assert result.fallbacks_used == ["rgb_autocontrast_jpeg_1280", "retry_attempt:2"]
    assert result.processed_variant_name == "rgb_autocontrast_jpeg_1280"
    assert result.ocr_attempts_used == 2


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


def test_workspace_extract_copy_uses_single_glm_flow() -> None:
    workspace = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "src"
        / "components"
        / "workspace"
        / "DocumentChatWorkspace.tsx"
    )
    source = workspace.read_text(encoding="utf-8")
    assert 'Extraction status: running GLM OCR...' in source
    assert 'await handleExtractTextInChat({ userPrompt: text });' in source
    assert "Select OCR Engine" not in source
    assert "Compare all three" not in source


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


def test_glm_post_pipeline_persists_unresolved_mentions_when_linking_is_deferred(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(
        ocr_router,
        "enforce_quality_gates",
        lambda *args, **kwargs: {
            "ner_allowed": False,
            "token_search_allowed": False,
            "downstream_mode": "degraded",
            "blocked_stages": [],
        },
    )
    monkeypatch.setattr(ocr_router, "_auto_index_run", lambda run_id: None)
    monkeypatch.setattr(ocr_router, "_build_consolidated_report", lambda *args, **kwargs: "=== CONSOLIDATED REPORT ===")

    def _fake_trace_analysis(run_id: str, text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        chunks = pipeline_db.insert_chunks(
            run_id,
            [
                {
                    "chunk_id": "chunk-1",
                    "idx": 0,
                    "start_offset": 0,
                    "end_offset": len(text),
                    "text": text,
                }
            ],
        )
        mentions = pipeline_db.insert_entity_mentions(
            run_id,
            [
                {
                    "mention_id": "mention-1",
                    "chunk_id": "chunk-1",
                    "start_offset": 0,
                    "end_offset": 6,
                    "surface": "Arthur",
                    "norm": "arthur",
                    "label": "PERSON",
                    "ent_type": "person",
                    "confidence": 0.71,
                    "method": "test",
                    "notes": "degraded capture",
                }
            ],
        )
        return chunks, mentions, [], {"skipped": False, "test_mode": True}

    monkeypatch.setattr(ocr_router, "_run_trace_analysis", _fake_trace_analysis)

    response = SaiaFullPageExtractResponse(
        status="FULL",
        model_used="glm-ocr:latest",
        detected_language="latin",
        script_hint="latin",
        confidence=0.74,
        warnings=[],
        lines=["Arthur"],
        text="Arthur appears here",
        fallbacks=[],
        comparison_runs=[],
    )
    payload = SaiaFullPageExtractRequest(
        document_id="doc-1",
        page_id="page-1",
        image_b64=SAMPLE_B64,
        apply_proofread=False,
    )

    result = ocr_router._run_post_ocr_pipeline_for_glm(payload, response, b"image-bytes")

    assert result["mentions_count"] == 1
    assert result["quality_label"] is not None
    assert result["authority_report"] is not None

    links = pipeline_db.list_mention_links_for_run(str(result["run_id"]))
    assert len(links) == 1
    assert links[0]["link_status"] == "unresolved_low_quality"
    assert links[0]["surface"] == "Arthur"
    assert links[0]["evidence_raw_text"] == "Arthur"
