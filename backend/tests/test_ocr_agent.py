from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents import ocr_agent  # type: ignore[import-untyped]
from app.schemas.agents_ocr import (  # type: ignore[import-untyped]
    OCRExtractOptions,
    OCRExtractRequest,
    OCRExtractSimpleResponse,
    OCRRegionInput,
)


class _FakeSaiaClient:
    def __init__(
        self,
        *,
        models: list[str],
        responses_by_model: dict[str, list[str]] | None = None,
        failures_by_model: dict[str, Exception] | None = None,
    ) -> None:
        self._models = models
        self._responses_by_model = responses_by_model or {}
        self._failures_by_model = failures_by_model or {}
        self.calls: list[dict[str, Any]] = []

    def list_models(self, force_refresh: bool = False) -> list[str]:
        return list(self._models)

    def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        model = str(kwargs.get("model") or "")
        failure = self._failures_by_model.get(model)
        if failure is not None:
            raise failure

        queue = self._responses_by_model.get(model, [])
        if not queue:
            return {"text": ""}
        return {"text": queue.pop(0)}


def _sample_request(options: OCRExtractOptions | None = None) -> OCRExtractRequest:
    return OCRExtractRequest(
        page_id="page-1",
        image_b64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mNk+M/wHwAE/wJ/lxNn4QAAAABJRU5ErkJggg==",
        regions=[OCRRegionInput(region_id="r1", bbox_xyxy=[0, 0, 1, 1])],
        options=options or OCRExtractOptions(),
    )


def test_choose_models_prefers_declared_order() -> None:
    available = [
        "gemma-3-27b-it",
        "internvl3.5-30b-a3b",
        "qwen3-vl-30b-a3b-instruct",
    ]
    selected = ocr_agent.choose_models(
        available,
        ["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b", "gemma-3-27b-it"],
    )

    assert selected == ["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b", "gemma-3-27b-it"]


def test_sanitize_ocr_text_strips_json_and_code_fences() -> None:
    assert ocr_agent.sanitize_ocr_text('```json\n{"text":"Linea prima\\nLinea secunda"}\n```') == (
        "Linea prima\nLinea secunda"
    )
    assert ocr_agent.sanitize_ocr_text('"Linea tertia"') == "Linea tertia"


def test_ocr_user_prompt_declares_schema_and_reading_order_rules() -> None:
    assert "Keys must be exactly: lines, text, script_hint, confidence, warnings" in ocr_agent.PALEO_OCR_USER_PROMPT
    assert "If page is two-column: output left column fully first, then right column." in ocr_agent.PALEO_OCR_USER_PROMPT


def test_build_ocr_messages_is_single_turn_system_and_user_with_image() -> None:
    messages = ocr_agent.build_ocr_messages("ZmFrZV9iNjQ=")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    assert len(user_content) == 2
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "image_url"
    assert "history" not in str(messages).lower()


def test_ocr_agent_enforces_locked_model(monkeypatch: Any) -> None:
    fake = _FakeSaiaClient(
        models=["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b"],
        responses_by_model={"internvl3.5-30b-a3b": ["Linea una"]},
        failures_by_model={"qwen3-vl-30b-a3b-instruct": RuntimeError("Model Not Found")},
    )
    monkeypatch.setattr(ocr_agent, "SaiaClient", lambda: fake)

    result = ocr_agent.run_ocr_extraction(
        _sample_request(
            OCRExtractOptions(
                model_preference=["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b"],
                max_fallbacks=2,
                quality_floor=0.0,
            )
        )
    )

    assert result.model == "internvl3.5-30b-a3b"
    assert result.text == "Linea una"
    assert result.fallbacksUsed == []
    assert fake.calls
    sent_messages = fake.calls[0]["messages"]
    assert isinstance(sent_messages, list)
    assert len(sent_messages) == 2
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[1]["role"] == "user"


def test_ocr_agent_returns_empty_string_for_no_readable_response(monkeypatch: Any) -> None:
    fake = _FakeSaiaClient(
        models=["internvl3.5-30b-a3b"],
        responses_by_model={"internvl3.5-30b-a3b": ["No readable text is visible in this image."]},
    )
    monkeypatch.setattr(ocr_agent, "SaiaClient", lambda: fake)

    result = ocr_agent.run_ocr_extraction(_sample_request(OCRExtractOptions(max_fallbacks=0, quality_floor=0.0)))

    assert result.model == "internvl3.5-30b-a3b"
    assert result.text == ""
    assert result.regions[0].text == ""
    assert "EMPTY_TEXT" in result.regions[0].flags


def test_parse_ocr_json_selects_expected_region_id() -> None:
    raw = (
        '{"document_script_hint":"latin","regions":['
        '{"region_id":"r0","category":"text","bbox_xywh":[0,0,10,10],"lines":["Alpha"],"text":"Alpha","confidence":0.7,"warnings":[]},'
        '{"region_id":"r1","category":"text","bbox_xywh":[0,0,10,10],"lines":["Reuerendi"],"text":"Reuerendi","confidence":0.9,"warnings":[]}'
        '],"full_text":"Alpha\\n\\nReuerendi"}'
    )
    parsed = ocr_agent.parse_ocr_json(raw, expected_region_id="r1", latin_lock=True)

    assert parsed["text"] == "Reuerendi"
    assert parsed["lines"] == ["Reuerendi"]


def test_parse_ocr_json_replaces_non_latin_when_latin_locked() -> None:
    raw = '{"lines":["Αρχαίος"],"text":"Αρχαίος","script_hint":"latin","confidence":0.8,"warnings":[]}'
    parsed = ocr_agent.parse_ocr_json(raw, latin_lock=True)

    assert parsed["text"] == ""
    assert "non_latin_chars_replaced" in parsed["warnings"]
    assert "pattern_junk_removed" in parsed["warnings"]


def test_ocr_agent_drops_non_latin_hallucinated_region(monkeypatch: Any) -> None:
    fake = _FakeSaiaClient(
        models=["internvl3.5-30b-a3b"],
        responses_by_model={
            "internvl3.5-30b-a3b": [
                '{"lines":["Αρχαίος"],"text":"Αρχαίος","script_hint":"latin","confidence":0.9,"warnings":[]}'
            ]
        },
    )
    monkeypatch.setattr(ocr_agent, "SaiaClient", lambda: fake)

    result = ocr_agent.run_ocr_extraction(_sample_request(OCRExtractOptions(max_fallbacks=0, quality_floor=0.0)))

    assert result.text == ""
    assert any(flag == "REGION_TEXT_DROPPED" for flag in result.regions[0].flags)
    assert any(msg.startswith("NON_LATIN_CHARS_DETECTED:") for msg in result.warnings)


def test_ocr_agent_simple_mode_returns_only_final_text(monkeypatch: Any) -> None:
    fake = _FakeSaiaClient(
        models=["internvl3.5-30b-a3b"],
        responses_by_model={
            "internvl3.5-30b-a3b": [
                '{"lines":["faunente"],"text":"faunente","script_hint":"latin","confidence":0.9,"warnings":[]}',
                "fauente",
            ]
        },
    )
    monkeypatch.setattr(ocr_agent, "SaiaClient", lambda: fake)

    request = _sample_request(OCRExtractOptions(max_fallbacks=0, quality_floor=0.0)).model_copy(
        update={"mode": "simple"}
    )
    result = ocr_agent.run_ocr_extraction(request)

    assert isinstance(result, OCRExtractSimpleResponse)
    assert result.text == "fauente"
    assert result.script_hint == "latin_medieval"
