from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.config import settings  # type: ignore[import-untyped]
from app.services.chat_ai import create_chat_completion, list_available_models  # type: ignore[import-untyped]


class _FakeClient:
    def __init__(self, model_ids: list[str], completion_text: str = "ok") -> None:
        self._model_ids = model_ids
        self._completion_text = completion_text
        self.calls: list[dict[str, Any]] = []

        self.models = SimpleNamespace(list=self._list_models)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_completion))

    def _list_models(self) -> Any:
        data = [SimpleNamespace(id=model_id) for model_id in self._model_ids]
        return SimpleNamespace(data=data)

    def _create_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._completion_text))]
        )


class _FakeRetryClient(_FakeClient):
    def __init__(self, model_ids: list[str], failing_model: str, completion_text: str = "ok") -> None:
        super().__init__(model_ids, completion_text=completion_text)
        self.failing_model = failing_model

    def _create_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("model") == self.failing_model:
            raise RuntimeError("Model Not Found")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._completion_text))]
        )


def test_list_available_models_marks_vision_models(monkeypatch: Any) -> None:
    fake = _FakeClient(["internvl3.5-30b-a3b", "meta-llama-3.1-8b-instruct"])
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "archai_chat_ai_model", "meta-llama-3.1-8b-instruct")

    payload = list_available_models()

    assert "models" in payload
    assert "internvl3.5-30b-a3b" in payload["vision_models"]
    assert payload["default_model"] == "meta-llama-3.1-8b-instruct"


def test_create_chat_completion_injects_archai_context(monkeypatch: Any) -> None:
    fake = _FakeClient(["meta-llama-3.1-8b-instruct"], completion_text="assistant output")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "archai_chat_ai_model", "meta-llama-3.1-8b-instruct")

    result = create_chat_completion(
        messages=[{"role": "user", "content": "Summarize this page."}],
        model=None,
        temperature=0.3,
        context={"document_id": "doc-1", "filename": "folio_001.png", "current_page_index": 0},
    )

    assert result["text"] == "assistant output"
    assert fake.calls, "Expected completion API call to be issued"

    sent = fake.calls[0]
    assert sent["model"] == "meta-llama-3.1-8b-instruct"
    assert sent["temperature"] == 0.3

    first_message = sent["messages"][0]
    assert first_message["role"] == "system"
    assert "document_id: doc-1" in first_message["content"]
    assert "filename: folio_001.png" in first_message["content"]


def test_create_chat_completion_retries_on_model_not_found(monkeypatch: Any) -> None:
    fake = _FakeRetryClient(
        ["internvl3.5-30b-a3b", "qwen3-vl-30b-a3b-instruct"],
        failing_model="internvl3.5-30b-a3b",
        completion_text="retry-output",
    )
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "archai_chat_ai_model", "internvl3.5-30b-a3b")

    result = create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract this."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            }
        ],
        model="internvl3.5-30b-a3b",
    )

    assert result["text"] == "retry-output"
    assert len(fake.calls) == 2
    assert fake.calls[0]["model"] == "internvl3.5-30b-a3b"
    assert fake.calls[1]["model"] == "qwen3-vl-30b-a3b-instruct"
