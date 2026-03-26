from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents.paleography_verification_agent import PaleographyVerificationAgent  # type: ignore[import-untyped]
from app.config import settings  # type: ignore[import-untyped]


class _FakeVerifierClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_completion))

    def _create_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.response_text))]
        )


def test_paleography_verification_agent_uses_dedicated_verifier_model(monkeypatch: Any) -> None:
    fake = _FakeVerifierClient(
        '{"assessment":"supported","corrected_answer":"Verified answer.","notes":["Matches transcript."],"citations_checked":["chunk-1"]}'
    )
    monkeypatch.setattr(settings, "paleography_verification_model", "qwen3-235b-a22b")
    monkeypatch.setattr(settings, "chat_rag_model", "qwen3-30b-a3b-instruct-2507")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)

    result = PaleographyVerificationAgent().run(
        SimpleNamespace(
            question="What does the passage say?",
            draft_answer="Draft answer.",
            transcript="Li rois Artus",
            authority_report="Arthur -> Q45720",
            evidence_text="[EVIDENCE]\ntext: Li rois Artus\n[/EVIDENCE]",
            ocr_run_id="run-1",
        )
    )

    assert result.model_used == "qwen3-235b-a22b"
    assert fake.calls[0]["model"] == "qwen3-235b-a22b"
    assert result.stage_metadata["stage_name"] == "verification"
    assert result.stage_metadata["model_used"] == "qwen3-235b-a22b"
    assert result.duration_ms >= 0


def test_paleography_verification_agent_sanitizes_reasoning_leaks(monkeypatch: Any) -> None:
    fake = _FakeVerifierClient(
        """
        {
          "assessment": "supported",
          "corrected_answer": "<think>hidden</think> Verified answer grounded in the text.",
          "notes": ["Reasoning: private chain of thought", "Transcript supports the reading."],
          "citations_checked": ["```chunk-1```", "ENTITY_EVIDENCE Arthur"]
        }
        """
    )
    monkeypatch.setattr(settings, "paleography_verification_model", "qwen3-235b-a22b")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)

    result = PaleographyVerificationAgent().run(
        SimpleNamespace(
            question="Who is mentioned?",
            draft_answer="Draft answer.",
            transcript="Arthur",
            authority_report="Arthur -> Q45720",
            evidence_text="[ENTITY_EVIDENCE]\ncanonical_label: King Arthur\n[/ENTITY_EVIDENCE]",
            ocr_run_id="run-1",
        )
    )

    assert result.corrected_answer == "Verified answer grounded in the text."
    assert result.notes == ["Transcript supports the reading."]
    assert result.citations_checked == ["chunk-1", "ENTITY_EVIDENCE Arthur"]
    assert result.inspection["confidence_or_assessment"] == "supported"


def test_paleography_verification_agent_handles_malformed_output_without_leaking_raw_text(monkeypatch: Any) -> None:
    fake = _FakeVerifierClient("<think>private scratchpad</think> unsupported because maybe something")
    monkeypatch.setattr(settings, "paleography_verification_model", "qwen3-235b-a22b")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)

    result = PaleographyVerificationAgent().run(
        SimpleNamespace(
            question="What does it say?",
            draft_answer="Draft answer retained.",
            transcript="Arthur",
            authority_report="",
            evidence_text="[EVIDENCE]\ntext: Arthur\n[/EVIDENCE]",
            ocr_run_id="run-1",
        )
    )

    assert result.assessment == "unavailable"
    assert result.corrected_answer == "Draft answer retained."
    assert result.notes == ["Verifier response could not be parsed cleanly."]
    assert "scratchpad" not in result.notes[0].lower()
    assert result.inspection["final_output"]["verified_answer"] == "Draft answer retained."
