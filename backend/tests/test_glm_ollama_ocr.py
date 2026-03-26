from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

from PIL import Image

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.services import glm_ollama_ocr  # type: ignore[import-untyped]
from app.schemas.agents_ocr import OCRDocumentMetadata  # type: ignore[import-untyped]


def _png_bytes(mode: str = "RGBA", size: tuple[int, int] = (2400, 1800)) -> bytes:
    image = Image.new(mode, size, (20, 40, 60, 0 if "A" in mode else 255))
    if "A" in mode:
        overlay = Image.new("RGBA", size, (255, 255, 255, 180))
        image.alpha_composite(overlay)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any], url: str) -> None:
        self.status_code = status_code
        self._payload = payload
        self.url = url
        self.ok = 200 <= status_code < 300
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


def test_preprocess_variants_produces_safe_jpeg_payloads() -> None:
    original_size, variants = glm_ollama_ocr.preprocess_variants(
        _png_bytes(),
        max_payload_bytes=2_000_000,
    )

    assert original_size == (2400, 1800)
    assert len(variants) == 6
    assert all(variant.size_bytes < 2_000_000 for variant in variants)
    assert all(variant.image_bytes[:2] == b"\xff\xd8" for variant in variants)

    with Image.open(io.BytesIO(variants[0].image_bytes)) as processed:
        assert processed.mode == "RGB"


def test_build_glm_ocr_prompt_includes_metadata_hints() -> None:
    prompt = glm_ollama_ocr.build_glm_ocr_prompt(
        language_hint="old_french",
        script_hint_seed="latin",
        metadata=OCRDocumentMetadata(
            language="Old French",
            year="c. 1275",
            place_or_origin="Paris",
            script_family="Gothic textualis",
            document_type="Liturgical manuscript",
            notes="Expect abbreviations and decorated initials.",
        ),
    )

    assert "Strict diplomatic transcription task." in prompt
    assert "Likely manuscript language: Old French; old_french" in prompt
    assert "Approximate manuscript date or year: c. 1275" in prompt
    assert "Script hint: latin" in prompt
    assert "Script family metadata: Gothic textualis" in prompt
    assert "Place or origin: Paris" in prompt
    assert "Document type: Liturgical manuscript" in prompt
    assert "Additional manuscript notes: Expect abbreviations and decorated initials." in prompt
    assert "Do not normalize, translate, summarize, or invent text based on the hints." in prompt


def test_run_glm_ollama_ocr_falls_back_to_generate_and_cleans_output(monkeypatch: Any) -> None:
    calls: list[str] = []

    def _fake_post(url: str, json: dict[str, Any], timeout: int) -> _FakeResponse:
        _ = json
        _ = timeout
        calls.append(url)
        if url.endswith("/api/chat"):
            return _FakeResponse(404, {"error": "not found"}, url)
        return _FakeResponse(200, {"response": "```text\nExtracted text: Linea una\n```"}, url)

    monkeypatch.setattr(glm_ollama_ocr.requests, "post", _fake_post)

    result = glm_ollama_ocr.run_glm_ollama_ocr(
        _png_bytes(mode="RGB", size=(1200, 900)),
        image_ref="page-1",
        model="glm-ocr:latest",
        host="http://localhost:11434",
        timeout=30,
        temperature=0.0,
        retries=1,
        max_payload_bytes=2_000_000,
    )

    assert result.text == "Linea una"
    assert result.lines == ["Linea una"]
    assert result.model_used == "glm-ocr:latest"
    assert result.original_size_bytes == len(_png_bytes(mode="RGB", size=(1200, 900)))
    assert result.attempts_used == 1
    assert calls[0].endswith("/api/chat")
    assert calls[1].endswith("/api/generate")


def test_run_glm_ollama_ocr_reports_variant_fallback_and_retry_metadata(monkeypatch: Any) -> None:
    variants = [
        glm_ollama_ocr.PreparedOcrImage(
            name="rgb_jpeg_1536",
            image_bytes=b"variant-a",
            width=1536,
            height=1024,
            size_bytes=120000,
            resized=True,
            compressed=False,
            applied_autocontrast=False,
            jpeg_quality=90,
        ),
        glm_ollama_ocr.PreparedOcrImage(
            name="rgb_autocontrast_jpeg_1280",
            image_bytes=b"variant-b",
            width=1280,
            height=853,
            size_bytes=88000,
            resized=True,
            compressed=True,
            applied_autocontrast=True,
            jpeg_quality=78,
        ),
    ]

    attempts_by_variant: dict[bytes, int] = {b"variant-a": 0, b"variant-b": 0}

    monkeypatch.setattr(
        glm_ollama_ocr,
        "preprocess_variants",
        lambda image_bytes, *, max_payload_bytes: ((2400, 1800), variants),
    )

    def _fake_ollama_extract(*, image_bytes: bytes, **kwargs: Any) -> str:
        _ = kwargs
        attempts_by_variant[image_bytes] += 1
        if image_bytes == b"variant-a":
            raise RuntimeError("Ollama HTTP 500 at /api/chat: shape failure")
        if attempts_by_variant[image_bytes] == 1:
            raise RuntimeError("temporary upstream failure")
        return "Linea due"

    monkeypatch.setattr(glm_ollama_ocr, "ollama_extract", _fake_ollama_extract)

    result = glm_ollama_ocr.run_glm_ollama_ocr(
        b"original-image-bytes",
        image_ref="page-2",
        model="glm-ocr:latest",
        host="http://localhost:11434",
        timeout=30,
        temperature=0.0,
        retries=2,
        max_payload_bytes=2_000_000,
    )

    assert result.text == "Linea due"
    assert result.original_size_bytes == len(b"original-image-bytes")
    assert result.processed_variant_name == "rgb_autocontrast_jpeg_1280"
    assert result.attempts_used == 2
    assert "OCR_VARIANT_FALLBACK:rgb_autocontrast_jpeg_1280" in result.warnings
    assert "OCR_RETRY_ATTEMPTS_USED:2" in result.warnings
