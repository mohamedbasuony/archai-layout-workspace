from __future__ import annotations

import sys
import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents.saia_ocr_agent import (  # type: ignore[import-untyped]
    SAIA_OCR_USER_PROMPT,
    SaiaOCRAgent,
    build_saia_ocr_messages,
)
from app.schemas.agents_ocr import SaiaOCRLocationSuggestion, SaiaOCRRequest  # type: ignore[import-untyped]
from app.services import saia_client  # type: ignore[import-untyped]

SAMPLE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADElEQVR42mNk+M/wHwAE/wJ/lxNn4QAAAABJRU5ErkJggg=="


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
        _ = force_refresh
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


def _valid_json(text: str = "Linea una", confidence: float = 0.9) -> str:
    return (
        "{"
        f"\"lines\":[\"{text}\"],"
        f"\"text\":\"{text}\","
        "\"script_hint\":\"latin\","
        "\"detected_language\":\"latin\","
        f"\"confidence\":{confidence},"
        "\"warnings\":[]"
        "}"
    )


def _png_b64(width: int, height: int) -> str:
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_prompt_declares_json_schema_and_strict_rules() -> None:
    assert "keys must be exactly: lines, text, script_hint, detected_language, confidence, warnings." in SAIA_OCR_USER_PROMPT
    assert "If nothing readable is visible, return lines=[] and text=\"\"." in SAIA_OCR_USER_PROMPT


def test_build_messages_is_two_message_stateless_payload() -> None:
    messages = build_saia_ocr_messages(SAMPLE_B64)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    content = messages[1]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


def test_build_messages_includes_location_suggestions_as_text_only() -> None:
    suggestions = [
        SaiaOCRLocationSuggestion(
            region_id="r1",
            category="Main script black",
            bbox_xywh=[10, 20, 300, 80],
        ),
    ]
    messages = build_saia_ocr_messages(SAMPLE_B64, location_suggestions=suggestions)
    assert len(messages) == 2
    content = messages[1]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert "Location suggestions" in str(content[0]["text"])
    assert "(10.0, 20.0, 300.0, 80.0)" in str(content[0]["text"])
    assert content[1]["type"] == "image_url"


def test_json_repair_retry_on_same_model() -> None:
    fake = _FakeSaiaClient(
        models=["internvl3.5-30b-a3b"],
        responses_by_model={"internvl3.5-30b-a3b": ["not-json", _valid_json()]},
    )
    agent = SaiaOCRAgent(client=fake, model_prefs=["internvl3.5-30b-a3b"])
    result = agent.extract(SaiaOCRRequest(image_b64=SAMPLE_B64, apply_proofread=False))

    assert result.model_used == "internvl3.5-30b-a3b"
    assert result.text == "Linea una"
    assert result.detected_language == "latin"
    assert len(fake.calls) == 2
    for call in fake.calls:
        messages = call.get("messages")
        assert isinstance(messages, list)
        assert len(messages) == 2


def test_model_fallback_when_first_model_not_found() -> None:
    fake = _FakeSaiaClient(
        models=["internvl2.5-8b-mpo", "internvl3.5-30b-a3b"],
        responses_by_model={"internvl3.5-30b-a3b": [_valid_json("Linea secunda")]},
        failures_by_model={"internvl2.5-8b-mpo": RuntimeError("404 model not found")},
    )
    agent = SaiaOCRAgent(
        client=fake,
        model_prefs=["internvl2.5-8b-mpo", "internvl3.5-30b-a3b"],
    )
    result = agent.extract(SaiaOCRRequest(image_b64=SAMPLE_B64, apply_proofread=False))

    assert result.model_used == "internvl3.5-30b-a3b"
    assert result.text == "Linea secunda"
    assert result.detected_language == "latin"
    assert len(result.fallbacks) == 1
    assert result.fallbacks[0].model == "internvl2.5-8b-mpo"


def test_invalid_detected_language_is_coerced_to_unknown() -> None:
    fake = _FakeSaiaClient(
        models=["internvl3.5-30b-a3b"],
        responses_by_model={
            "internvl3.5-30b-a3b": [
                (
                    "{"
                    "\"lines\":[\"Linea una\"],"
                    "\"text\":\"Linea una\","
                    "\"script_hint\":\"latin\","
                    "\"detected_language\":\"klingon\","
                    "\"confidence\":0.7,"
                    "\"warnings\":[]"
                    "}"
                )
            ]
        },
    )
    agent = SaiaOCRAgent(client=fake, model_prefs=["internvl3.5-30b-a3b"])
    result = agent.extract(SaiaOCRRequest(image_b64=SAMPLE_B64, apply_proofread=False))

    assert result.detected_language == "latin"


def test_frontend_bundle_has_no_backend_api_key_reference() -> None:
    frontend_src = Path(__file__).resolve().parents[2] / "frontend" / "src"
    target_files = list(frontend_src.rglob("*.ts")) + list(frontend_src.rglob("*.tsx"))
    assert target_files
    blocked_tokens = ("SAIA_API_KEY", "CHAT_AI_API_KEY", "ARCHAI_CHAT_AI_API_KEY")

    for file_path in target_files:
        source = file_path.read_text(encoding="utf-8")
        for token in blocked_tokens:
            assert token not in source, f"Found backend credential token {token} in {file_path}"


def test_saia_api_key_is_read_from_env(monkeypatch: Any) -> None:
    monkeypatch.setattr(saia_client.settings, "chat_ai_api_key", "", raising=False)
    monkeypatch.setattr(saia_client.settings, "saia_api_key", "", raising=False)
    monkeypatch.setattr(saia_client.settings, "archai_chat_ai_api_key", "", raising=False)
    monkeypatch.setattr(saia_client.settings, "archai_saia_api_key", "", raising=False)
    monkeypatch.setenv("SAIA_API_KEY", "env-key-for-test")

    assert saia_client.SaiaClient._resolve_api_key() == "env-key-for-test"


def test_extract_auto_resizes_image_before_call(monkeypatch: Any) -> None:
    fake = _FakeSaiaClient(
        models=["internvl3.5-30b-a3b"],
        responses_by_model={"internvl3.5-30b-a3b": [_valid_json("Linea una")]},
    )
    monkeypatch.setattr("app.agents.saia_ocr_agent.SAIA_OCR_MAX_PIXELS", 50)
    monkeypatch.setattr("app.agents.saia_ocr_agent.SAIA_OCR_MAX_LONG_EDGE", 20)

    original = _png_b64(30, 30)
    agent = SaiaOCRAgent(client=fake, model_prefs=["internvl3.5-30b-a3b"])
    result = agent.extract(SaiaOCRRequest(image_b64=original, apply_proofread=False))

    assert result.model_used == "internvl3.5-30b-a3b"
    assert any(item.startswith("AUTO_RESIZED_FOR_LIMIT:") for item in result.warnings)

    call_messages = fake.calls[0]["messages"]
    image_url = call_messages[1]["content"][1]["image_url"]["url"]
    sent_b64 = image_url.split(",", 1)[1]
    assert sent_b64 != original


def test_fail_all_models_include_model_error_details() -> None:
    fake = _FakeSaiaClient(
        models=["internvl3.5-30b-a3b"],
        failures_by_model={"internvl3.5-30b-a3b": RuntimeError("simulated upstream failure")},
    )
    agent = SaiaOCRAgent(client=fake, model_prefs=["internvl3.5-30b-a3b"])
    result = agent.extract(SaiaOCRRequest(image_b64=SAMPLE_B64, apply_proofread=False))

    assert result.status == "FAIL"
    assert "OCR_FAILED_ALL_MODELS" in result.warnings
    assert any(item.startswith("MODEL_ERROR:internvl3.5-30b-a3b:") for item in result.warnings)


def test_non_internvl_models_are_ignored() -> None:
    fake = _FakeSaiaClient(
        models=["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b"],
        responses_by_model={"internvl3.5-30b-a3b": [_valid_json("Linea tertia")]},
    )
    agent = SaiaOCRAgent(client=fake, model_prefs=["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b"])
    result = agent.extract(SaiaOCRRequest(image_b64=SAMPLE_B64, apply_proofread=False))

    assert result.model_used == "internvl3.5-30b-a3b"
    assert result.text == "Linea tertia"
    assert all(call.get("model") == "internvl3.5-30b-a3b" for call in fake.calls)
