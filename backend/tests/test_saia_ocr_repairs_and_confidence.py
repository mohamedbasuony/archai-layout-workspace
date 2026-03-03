from __future__ import annotations

import sys
from pathlib import Path

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents.ocr_proofreader_agent import OcrProofreaderAgent  # type: ignore[import-untyped]
from app.agents.saia_ocr_agent import (  # type: ignore[import-untyped]
    SaiaOCRAgent,
    _apply_ocr_confidence_caps,
    _parse_ocr_payload,
)


class _DummySaiaClient:
    def list_models(self, force_refresh: bool = False) -> list[str]:
        _ = force_refresh
        return ["internvl3.5-30b-a3b"]

    def chat_completion(self, **kwargs):  # pragma: no cover - not used in these tests
        _ = kwargs
        return {"text": ""}


def test_repair_splits_embedded_newlines_and_rejoins_text() -> None:
    raw = (
        "{"
        "\"lines\":[\"a\\nb\\nc\"],"
        "\"text\":\"a\\nb\\nc\","
        "\"script_hint\":\"latin\","
        "\"detected_language\":\"latin\","
        "\"confidence\":0.9,"
        "\"warnings\":[]"
        "}"
    )
    parsed = _parse_ocr_payload(raw)
    assert parsed is not None
    assert parsed["lines"] == ["a", "b", "c"]
    assert parsed["text"] == "a\nb\nc"
    assert "repair:lines_split_embedded_newlines" in parsed["warnings"]


def test_confidence_cap_applies_for_high_junk_ratio() -> None:
    noisy = "x1! z2! q3! b4? c5? d6?"
    confidence, warnings = _apply_ocr_confidence_caps(noisy, 0.95, [])
    assert confidence <= 0.75
    assert "ocr:confidence_capped" in warnings
    assert any(item.startswith("ocr:junk_ratio=") for item in warnings)


def test_proofread_no_change_adds_warning(monkeypatch) -> None:
    def _same_text(self, ocr_text: str, script_hint: str | None, detected_language: str | None = None) -> str:
        _ = (self, script_hint, detected_language)
        return ocr_text

    monkeypatch.setattr(OcrProofreaderAgent, "proofread", _same_text)

    agent = SaiaOCRAgent(client=_DummySaiaClient(), model_prefs=["internvl3.5-30b-a3b"])
    lines = ["linea prima", "linea secunda", "linea tertia"]
    text = "\n".join(lines)

    out_lines, out_text, warnings = agent._maybe_proofread(
        model="internvl3.5-30b-a3b",
        script_hint="latin",
        detected_language="latin",
        lines=lines,
        text=text,
        enabled=True,
    )

    assert out_lines == lines
    assert out_text == text
    assert "proofread:no_changes" in warnings
