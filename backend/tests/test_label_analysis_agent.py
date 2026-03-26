from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents.label_analysis_agent import LabelAnalysisAgent  # type: ignore[import-untyped]


def _png_b64(size: tuple[int, int] = (120, 90)) -> str:
    image = Image.new("RGB", size, (255, 255, 255))
    for x in range(20, 51):
        for y in range(15, 46):
            image.putpixel((x, y), (180, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class _FakeSaiaClient:
    def __init__(self, text: str = "This appears to be a decorated initial with red infill.") -> None:
        self.calls: list[dict[str, Any]] = []
        self.text = text

    def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"text": self.text}


class _TransientSaiaClient(_FakeSaiaClient):
    def __init__(self, text: str = "A", failures_before_success: int = 1) -> None:
        super().__init__(text=text)
        self.failures_before_success = failures_before_success

    def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise RuntimeError("502 Proxy Error: Error reading from remote server")
        return {"text": self.text}


class _SequenceSaiaClient(_FakeSaiaClient):
    def __init__(self, texts: list[str]) -> None:
        super().__init__(text=texts[0] if texts else "UNKNOWN")
        self._texts = list(texts)

    def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        text = self._texts.pop(0) if self._texts else self.text
        return {"text": text}


def test_label_analysis_agent_crops_requested_regions_and_calls_saia() -> None:
    fake_client = _FakeSaiaClient()
    agent = LabelAnalysisAgent(client=fake_client)
    payload = SimpleNamespace(
        question="What is this embellished initial?",
        label_name="Embellished",
        image_b64=_png_b64(),
        filename="folio_001.png",
        regions=[
            SimpleNamespace(
                region_id="ann-1",
                bbox_xyxy=[20, 15, 50, 45],
                polygons=[],
            )
        ],
    )

    result = agent.run(payload)

    assert result.status == "ok"
    assert result.label_name == "Embellished"
    assert result.region_count == 1
    assert result.text.startswith("This appears to be")
    assert fake_client.calls, "Expected SAIA client to receive a request."

    sent = fake_client.calls[0]
    assert sent["model"] == "qwen3-vl-30b-a3b-instruct"
    prompt = sent["messages"][1]["content"][0]["text"]
    assert prompt.startswith("Requested segmentation label: Embellished")
    assert "This is a decorated-initial analysis task." in prompt
    assert "This is an illustration analysis task." not in prompt
    image_url = sent["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    assert result.stage_metadata["image_format_sent"] == "jpeg"
    assert result.stage_metadata["region_count_used"] == 1

    crop_b64 = image_url.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(crop_b64))) as crop:
        assert crop.width >= 30
        assert crop.height >= 30
        assert crop.mode == "RGB"


def test_label_analysis_agent_uses_glm_ocr_for_textual_labels(monkeypatch: Any) -> None:
    fake_client = _FakeSaiaClient()
    agent = LabelAnalysisAgent(client=fake_client)
    payload = SimpleNamespace(
        question="What does this page number say?",
        label_name="Page Number",
        image_b64=_png_b64(),
        filename="folio_001.png",
        regions=[
            SimpleNamespace(
                region_id="ann-1",
                bbox_xyxy=[20, 15, 50, 45],
                polygons=[],
            )
        ],
    )

    monkeypatch.setattr(
        "app.agents.label_analysis_agent.run_glm_ollama_ocr",
        lambda *args, **kwargs: SimpleNamespace(text="fol. xii", warnings=[]),
    )

    result = agent.run(payload)

    assert result.status == "ok"
    assert result.analysis_mode == "textual"
    assert result.text == "fol. xii"
    assert result.ocr_text == "fol. xii"
    assert result.model_used == "glm-ocr:latest"
    assert result.stage_metadata["selected_region_strategy"] == "single_region_only"
    assert fake_client.calls == []


def test_label_analysis_agent_sanitizes_prompt_echo_for_page_number(monkeypatch: Any) -> None:
    fake_client = _FakeSaiaClient()
    agent = LabelAnalysisAgent(client=fake_client)
    payload = SimpleNamespace(
        question="What does this page number say?",
        label_name="Page Number",
        image_b64=_png_b64(),
        filename="folio_001.png",
        regions=[
            SimpleNamespace(
                region_id="ann-1",
                bbox_xyxy=[20, 15, 50, 45],
                polygons=[],
            )
        ],
    )

    monkeypatch.setattr(
        "app.agents.label_analysis_agent.run_glm_ollama_ocr",
        lambda *args, **kwargs: SimpleNamespace(
            text=(
                "The transcription is:\n"
                "Strict diplomatic transcription task.\n\n"
                "The text is:\n"
                "Preserve reading order exactly.\n"
                "Preserve line breaks exactly.\n"
                "Output one manuscript line per output line."
            ),
            warnings=[],
        ),
    )

    result = agent.run(payload)

    assert result.status == "ok"
    assert result.analysis_mode == "textual"
    assert result.text == "The page number is unreadable on the selected crop."
    assert result.ocr_text == ""
    assert fake_client.calls == []


def test_label_analysis_agent_uses_text_model_for_textual_explanation(monkeypatch: Any) -> None:
    fake_client = _FakeSaiaClient()
    agent = LabelAnalysisAgent(client=fake_client)
    payload = SimpleNamespace(
        question="What is the function of this gloss?",
        label_name="Gloss",
        image_b64=_png_b64(),
        filename="folio_001.png",
        regions=[
            SimpleNamespace(
                region_id="ann-1",
                bbox_xyxy=[20, 15, 50, 45],
                polygons=[],
            )
        ],
    )

    monkeypatch.setattr(
        "app.agents.label_analysis_agent.run_glm_ollama_ocr",
        lambda *args, **kwargs: SimpleNamespace(text="nota bene", warnings=[]),
    )

    result = agent.run(payload)

    assert result.status == "ok"
    assert result.analysis_mode == "textual"
    assert fake_client.calls
    assert fake_client.calls[0]["model"] == "qwen3-30b-a3b-instruct-2507"
    assert isinstance(fake_client.calls[0]["messages"][1]["content"], str)
    assert "This is a textual label explanation task." in fake_client.calls[0]["messages"][1]["content"]


def test_label_analysis_agent_uses_initial_letter_identification_submode() -> None:
    fake_client = _FakeSaiaClient(text="Letter: A\nConfidence: medium")
    agent = LabelAnalysisAgent(client=fake_client)
    payload = SimpleNamespace(
        question="What is the embellished letter?",
        label_name="Embellished",
        image_b64=_png_b64(),
        filename="folio_001.png",
        regions=[
            SimpleNamespace(
                region_id="ann-1",
                bbox_xyxy=[20, 15, 50, 45],
                polygons=[],
            )
        ],
    )

    result = agent.run(payload)

    assert result.status == "ok"
    assert result.analysis_mode == "visual"
    assert result.text == "A"
    assert result.model_used == "qwen3-vl-30b-a3b-instruct"
    assert result.stage_metadata["stage_name"] == "label_analysis"
    assert result.stage_metadata["mode_used"] == "initial_letter_identification"
    assert result.stage_metadata["region_count_used"] == 1
    assert result.stage_metadata["selected_region_strategy"] == "single_region_only"
    assert result.inspection["confidence_or_assessment"] == "medium"
    prompt = fake_client.calls[0]["messages"][1]["content"][0]["text"]
    assert "This is an initial-letter identification task." in prompt
    assert "Letter: <single uppercase Latin letter or UNKNOWN>" in prompt
    assert "Do not describe ornament, iconography, or decoration before deciding whether the letterform itself is identifiable." in prompt


def test_label_analysis_agent_routes_plural_initial_query_to_multi_region_letter_identification() -> None:
    fake_client = _SequenceSaiaClient(["Letter: R", "Letter: C"])
    agent = LabelAnalysisAgent(client=fake_client)
    payload = SimpleNamespace(
        question="What are the embellished letters?",
        label_name="Embellished",
        image_b64=_png_b64(size=(220, 180)),
        filename="folio_001.png",
        regions=[
            SimpleNamespace(region_id="ann-1", bbox_xyxy=[10, 15, 40, 55], polygons=[]),
            SimpleNamespace(region_id="ann-2", bbox_xyxy=[80, 20, 130, 90], polygons=[]),
        ],
    )

    result = agent.run(payload)

    assert result.status == "ok"
    assert result.analysis_mode == "visual"
    assert result.text == "Region 1: R\nRegion 2: C"
    assert result.stage_metadata["mode_used"] == "multi_initial_letter_identification"
    assert result.stage_metadata["region_count_used"] == 2
    assert result.stage_metadata["selected_region_strategy"] == "all_initial_regions_reading_order"
    assert len(fake_client.calls) == 2
    assert all(call["deterministic"] is True for call in fake_client.calls)
    assert all(call["max_tokens"] == 24 for call in fake_client.calls)
    assert "decorated-initial analysis task" not in fake_client.calls[0]["messages"][1]["content"][0]["text"].lower()


def test_label_analysis_agent_retries_transient_visual_error_and_uses_tighter_initial_letter_limits(monkeypatch: Any) -> None:
    fake_client = _TransientSaiaClient(text="A", failures_before_success=1)
    monkeypatch.setattr("app.agents.label_analysis_agent.time.sleep", lambda seconds: None)
    agent = LabelAnalysisAgent(client=fake_client)
    payload = SimpleNamespace(
        question="Which letter is this embellished initial?",
        label_name="Embellished",
        image_b64=_png_b64(),
        filename="folio_001.png",
        regions=[
            SimpleNamespace(
                region_id="ann-1",
                bbox_xyxy=[20, 15, 50, 45],
                polygons=[],
            )
        ],
    )

    result = agent.run(payload)

    assert result.status == "ok"
    assert result.text == "A"
    assert len(fake_client.calls) == 2
    assert fake_client.calls[0]["max_tokens"] == 32
    assert fake_client.calls[0]["timeout_seconds"] == 12.0
    assert fake_client.calls[0]["deterministic"] is True
    assert result.stage_metadata["retry_count"] == 1


def test_label_analysis_agent_falls_back_after_repeated_transient_initial_letter_failure(monkeypatch: Any) -> None:
    fake_client = _TransientSaiaClient(text="A", failures_before_success=2)
    fake_client.list_models = lambda: ["qwen3-vl-30b-a3b-instruct", "internvl3.5-30b-a3b"]  # type: ignore[attr-defined]
    monkeypatch.setattr("app.agents.label_analysis_agent.time.sleep", lambda seconds: None)
    agent = LabelAnalysisAgent(client=fake_client)
    monkeypatch.setattr(agent, "_choose_retry_model", lambda requested_model, prefer_vision: "internvl3.5-30b-a3b")
    payload = SimpleNamespace(
        question="Which letter is this embellished initial?",
        label_name="Embellished",
        image_b64=_png_b64(),
        filename="folio_001.png",
        regions=[
            SimpleNamespace(
                region_id="ann-1",
                bbox_xyxy=[20, 15, 50, 45],
                polygons=[],
            )
        ],
    )

    result = agent.run(payload)

    assert result.status == "ok"
    assert result.text == "A"
    assert result.model_used == "internvl3.5-30b-a3b"
    assert any("LABEL_TRANSIENT_FALLBACK:internvl3.5-30b-a3b" in warning for warning in result.warnings)
    assert result.stage_metadata["retry_count"] == 2


def test_label_analysis_agent_uses_illustration_specific_prompt() -> None:
    fake_client = _FakeSaiaClient(text="The crop shows a standing figure holding a book.")
    agent = LabelAnalysisAgent(client=fake_client)
    payload = SimpleNamespace(
        question="What is shown in this illustration?",
        label_name="Illustrations",
        image_b64=_png_b64(),
        filename="folio_001.png",
        regions=[
            SimpleNamespace(
                region_id="ann-1",
                bbox_xyxy=[20, 15, 50, 45],
                polygons=[],
            )
        ],
    )

    result = agent.run(payload)

    assert result.status == "ok"
    assert result.analysis_mode == "visual"
    prompt = fake_client.calls[0]["messages"][1]["content"][0]["text"]
    assert "This is an illustration analysis task." in prompt
    assert "This is a decorated-initial analysis task." not in prompt


def test_label_analysis_agent_uses_structural_prompt_for_column() -> None:
    fake_client = _FakeSaiaClient(text="This appears to be the main writing block of a column.")
    agent = LabelAnalysisAgent(client=fake_client)
    payload = SimpleNamespace(
        question="What is the function of this column?",
        label_name="Column",
        image_b64=_png_b64(),
        filename="folio_001.png",
        regions=[
            SimpleNamespace(
                region_id="ann-1",
                bbox_xyxy=[20, 15, 50, 45],
                polygons=[],
            )
        ],
    )

    result = agent.run(payload)

    assert result.status == "ok"
    assert result.analysis_mode == "structural"
    assert result.model_used == "metadata-only"
    assert "structural" in result.text.lower()
    assert fake_client.calls == []


def test_label_analysis_agent_selects_single_best_region_by_default() -> None:
    fake_client = _FakeSaiaClient()
    agent = LabelAnalysisAgent(client=fake_client)
    payload = SimpleNamespace(
        question="What is this embellished initial?",
        label_name="Embellished",
        image_b64=_png_b64(size=(220, 180)),
        filename="folio_001.png",
        regions=[
            SimpleNamespace(region_id="ann-1", bbox_xyxy=[10, 10, 30, 30], polygons=[]),
            SimpleNamespace(region_id="ann-2", bbox_xyxy=[60, 20, 150, 140], polygons=[]),
        ],
    )

    result = agent.run(payload)

    assert result.status == "ok"
    assert result.region_count == 1
    assert result.stage_metadata["region_count_input"] == 2
    assert result.stage_metadata["region_count_used"] == 1
    assert result.stage_metadata["selected_region_strategy"] == "largest_decorative_region"
