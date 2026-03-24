from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents import ocr_agent  # type: ignore[import-untyped]
from app.db import pipeline_db  # type: ignore[import-untyped]
from app.schemas.agents_ocr import (  # type: ignore[import-untyped]
    OCRExtractOptions,
    OCRExtractRequest,
    OCRRegionInput,
)
from app.services.ocr_backends import (  # type: ignore[import-untyped]
    OCRBackendResult,
    _resolve_model_path,
    select_backend_plan,
)


SAMPLE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mNk+M/wHwAE/wJ/lxNn4QAAAABJRU5ErkJggg=="


class _DummyClient:
    def list_models(self, force_refresh: bool = False) -> list[str]:
        _ = force_refresh
        return []


class _FakeBackend:
    def __init__(self, backend_name: str, model_name: str, outputs: dict[str, tuple[str, float | None, str | None]]) -> None:
        self.backend_name = backend_name
        self.model_name = model_name
        self.outputs = outputs
        self.calls: list[str] = []

    def recognize(self, crop_b64: str, metadata: Any) -> OCRBackendResult:
        _ = crop_b64
        self.calls.append(str(metadata.region_id))
        text, confidence, script_hint = self.outputs.get(str(metadata.region_id), ("", 0.0, None))
        return OCRBackendResult(
            text=text,
            confidence=confidence,
            backend_name=self.backend_name,
            model_name=self.model_name,
            raw_metadata={"warnings": [], "flags": []},
            region_id=str(metadata.region_id),
            page_id=metadata.page_id,
            script_hint=script_hint,
        )


class _FailingBackend:
    def __init__(self, backend_name: str, message: str) -> None:
        self.backend_name = backend_name
        self.message = message

    def recognize(self, crop_b64: str, metadata: Any) -> OCRBackendResult:
        _ = crop_b64
        _ = metadata
        raise ocr_agent.OCRBackendError(self.message)


def _request(options: OCRExtractOptions) -> OCRExtractRequest:
    return OCRExtractRequest(
        page_id="page-1",
        image_b64=SAMPLE_B64,
        regions=[
            OCRRegionInput(region_id="line-1", bbox_xyxy=[0, 0, 1, 1], reading_order=0, label="Main script black")
        ],
        options=options,
    )


def _multi_region_request(options: OCRExtractOptions) -> OCRExtractRequest:
    return OCRExtractRequest(
        page_id="page-1",
        image_b64=SAMPLE_B64,
        regions=[
            OCRRegionInput(region_id="line-1", bbox_xyxy=[0, 0, 1, 1], reading_order=0, label="Main script black"),
            OCRRegionInput(region_id="line-2", bbox_xyxy=[0, 2, 1, 3], reading_order=1, label="Main script black"),
            OCRRegionInput(region_id="line-3", bbox_xyxy=[0, 4, 1, 5], reading_order=2, label="Main script black"),
            OCRRegionInput(region_id="line-4", bbox_xyxy=[0, 6, 1, 7], reading_order=3, label="Main script black"),
        ],
        options=options,
    )


def test_backend_plan_routes_by_language_hint() -> None:
    latin = select_backend_plan(explicit_backend="auto", language_hint="latin", compare_backends=[])
    french = select_backend_plan(explicit_backend="auto", language_hint="middle_french", compare_backends=[])
    unknown = select_backend_plan(explicit_backend="auto", language_hint="unknown", compare_backends=[])

    assert latin.attempt_backends == ("kraken_cremma_lat", "kraken_catmus", "kraken_mccatmus")
    assert french.attempt_backends == ("kraken_cremma_medieval", "kraken_catmus", "kraken_mccatmus")
    assert unknown.attempt_backends == ("kraken_catmus", "kraken_mccatmus")


def test_backend_plan_routes_german_family_to_mccatmus_first() -> None:
    german = select_backend_plan(explicit_backend="auto", language_hint="middle_high_german", compare_backends=[])

    assert german.attempt_backends == ("kraken_mccatmus", "kraken_catmus")


def test_explicit_calamari_and_glm_backends_win() -> None:
    calamari = select_backend_plan(explicit_backend="calamari", language_hint="latin", compare_backends=["glmocr"])
    glm = select_backend_plan(explicit_backend="glmocr", language_hint="old_french", compare_backends=["calamari"])

    assert calamari.primary_backend == "calamari"
    assert calamari.attempt_backends == ("calamari",)
    assert calamari.comparison_backends == ("glmocr",)
    assert glm.primary_backend == "glmocr"
    assert glm.attempt_backends == ("glmocr",)
    assert glm.comparison_backends == ("calamari",)


def test_explicit_saia_backend_preserves_region_geometry(monkeypatch: Any) -> None:
    fake_runtime = {
        "saia": _FakeBackend("saia", "internvl3.5-30b-a3b", {"line-1": ("Linea una", 0.91, "latin_medieval")})
    }
    monkeypatch.setattr(ocr_agent, "build_backend_runtime", lambda *args, **kwargs: fake_runtime)

    result = ocr_agent.run_ocr_extraction(
        _request(OCRExtractOptions(backend="saia", apply_proofread=False)),
        client=_DummyClient(),
    )

    assert result.ocr_backend == "saia"
    assert result.regions[0].region_id == "line-1"
    assert result.regions[0].bbox_xyxy == [0.0, 0.0, 1.0, 1.0]
    assert result.regions[0].backend_name == "saia"
    assert result.text == "Linea una"


def test_latin_auto_route_falls_back_to_catmus_when_primary_is_weak(monkeypatch: Any) -> None:
    fake_runtime = {
        "kraken_cremma_lat": _FakeBackend("kraken_cremma_lat", "CREMMA-Medieval-LAT", {"line-1": ("", 0.05, None)}),
        "kraken_catmus": _FakeBackend("kraken_catmus", "CATMuS Medieval", {"line-1": ("Linea catmus", 0.88, None)}),
    }
    monkeypatch.setattr(ocr_agent, "build_backend_runtime", lambda *args, **kwargs: fake_runtime)

    result = ocr_agent.run_ocr_extraction(
        _request(OCRExtractOptions(language_hint="latin", quality_floor=0.5, apply_proofread=False)),
        client=_DummyClient(),
    )

    assert result.ocr_backend == "kraken_catmus"
    assert result.text == "Linea catmus"
    assert "kraken_cremma_lat" in result.fallbacksUsed


def test_comparison_mode_returns_multiple_backend_outputs(monkeypatch: Any) -> None:
    fake_runtime = {
        "kraken_catmus": _FakeBackend("kraken_catmus", "CATMuS Medieval", {"line-1": ("Linea catmus", 0.87, None)}),
        "saia": _FakeBackend("saia", "internvl3.5-30b-a3b", {"line-1": ("Linea saia", 0.90, "latin_medieval")}),
    }
    monkeypatch.setattr(ocr_agent, "build_backend_runtime", lambda *args, **kwargs: fake_runtime)

    result = ocr_agent.run_ocr_extraction(
        _request(
            OCRExtractOptions(
                backend="kraken_catmus",
                compare_backends=["saia"],
                apply_proofread=False,
            )
        ),
        client=_DummyClient(),
    )

    assert result.ocr_backend == "kraken_catmus"
    assert len(result.comparison_results) == 2
    assert sum(1 for item in result.comparison_results if item.selected) == 1
    assert {item.backend_name for item in result.comparison_results} == {"kraken_catmus", "saia"}


def test_comparison_results_are_persisted_side_by_side(tmp_path: Path, monkeypatch: Any) -> None:
    db_path = tmp_path / "archai.sqlite"
    monkeypatch.setenv("ARCHAI_DB_PATH", str(db_path))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)

    run_id = pipeline_db.create_run(asset_ref="page-1", asset_sha256="sha")
    pipeline_db.insert_ocr_backend_results(
        run_id,
        [
            {
                "page_id": "page-1",
                "region_id": "line-1",
                "backend_name": "kraken_catmus",
                "model_name": "CATMuS Medieval",
                "confidence": 0.8,
                "selected": True,
                "text": "Linea catmus",
                "raw_json": {"engine": "kraken"},
            },
            {
                "page_id": "page-1",
                "region_id": "line-1",
                "backend_name": "saia",
                "model_name": "internvl3.5-30b-a3b",
                "confidence": 0.9,
                "selected": False,
                "text": "Linea saia",
                "raw_json": {"engine": "saia"},
            },
        ],
    )

    results = pipeline_db.list_ocr_backend_results(run_id)
    assert len(results) == 2
    assert {item["backend_name"] for item in results} == {"kraken_catmus", "saia"}
    assert any(item["selected"] for item in results)


def test_benchmark_reference_hook_persists_reference_text(tmp_path: Path, monkeypatch: Any) -> None:
    db_path = tmp_path / "archai.sqlite"
    monkeypatch.setenv("ARCHAI_DB_PATH", str(db_path))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)

    run_id = pipeline_db.create_run(asset_ref="page-1", asset_sha256="sha")
    benchmark_id = pipeline_db.insert_ocr_benchmark_reference(
        run_id,
        page_id="page-1",
        source_label="transkribus",
        reference_text="Reference line one\nReference line two",
    )

    rows = pipeline_db.list_ocr_benchmark_references(run_id)
    assert benchmark_id
    assert len(rows) == 1
    assert rows[0]["page_id"] == "page-1"
    assert rows[0]["source_label"] == "transkribus"


def test_failed_backends_still_produce_comparison_rows(monkeypatch: Any) -> None:
    fake_runtime = {
        "kraken_cremma_lat": _FailingBackend("kraken_cremma_lat", "missing CREMMA lat model"),
        "kraken_catmus": _FailingBackend("kraken_catmus", "missing CATMuS model"),
        "kraken_mccatmus": _FailingBackend("kraken_mccatmus", "missing McCATMuS model"),
        "saia": _FailingBackend("saia", "SAIA unavailable"),
    }
    monkeypatch.setattr(ocr_agent, "build_backend_runtime", lambda *args, **kwargs: fake_runtime)

    result = ocr_agent.run_ocr_extraction(
        _request(
            OCRExtractOptions(
                language_hint="latin",
                compare_backends=["saia"],
                apply_proofread=False,
            )
        ),
        client=_DummyClient(),
    )

    assert result.status == "FAILED"
    assert result.text == ""
    assert len(result.comparison_results) == 4
    assert {item.backend_name for item in result.comparison_results} == {
        "kraken_cremma_lat",
        "kraken_catmus",
        "kraken_mccatmus",
        "saia",
    }
    assert all(not item.selected for item in result.comparison_results)
    assert all("error" in (item.raw_metadata or {}) for item in result.comparison_results)


def test_degenerate_repeat_output_falls_through_to_fallback_backend(monkeypatch: Any) -> None:
    fake_runtime = {
        "kraken_cremma_lat": _FakeBackend(
            "kraken_cremma_lat",
            "CREMMA-Medieval-LAT",
            {
                "line-1": ("In primis autem", 0.91, None),
                "line-2": ("In primis autem", 0.91, None),
                "line-3": ("In primis autem", 0.91, None),
                "line-4": ("In primis autem", 0.91, None),
            },
        ),
        "kraken_catmus": _FakeBackend(
            "kraken_catmus",
            "CATMuS Medieval",
            {
                "line-1": ("Prima linea", 0.82, None),
                "line-2": ("Secunda linea", 0.84, None),
                "line-3": ("Tertia linea", 0.83, None),
                "line-4": ("Quarta linea", 0.81, None),
            },
        ),
        "kraken_mccatmus": _FailingBackend("kraken_mccatmus", "missing McCATMuS model"),
    }
    monkeypatch.setattr(ocr_agent, "build_backend_runtime", lambda *args, **kwargs: fake_runtime)

    result = ocr_agent.run_ocr_extraction(
        _multi_region_request(
            OCRExtractOptions(language_hint="latin", quality_floor=0.5, apply_proofread=False)
        ),
        client=_DummyClient(),
    )

    assert result.ocr_backend in {"kraken_cremma_lat", "kraken_catmus"}
    assert result.text.count("In primis autem") <= 1
    assert "Secunda linea" in result.text
    assert "kraken_cremma_lat" in result.fallbacksUsed
    assert any(
        "DEGENERATE_REPEAT" in flag
        for region in result.regions
        for flag in (region.flags or [])
    )


def test_resolve_model_path_finds_project_root_weights() -> None:
    path = _resolve_model_path("weights/kraken_models/catmus_medieval.mlmodel")
    assert "catmus" in path.name
    assert path.exists()
