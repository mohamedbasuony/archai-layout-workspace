from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.config import settings  # type: ignore[import-untyped]
from app.routers import chat as chat_router  # type: ignore[import-untyped]
from app.services.chat_ai import _context_to_system_message, _format_verification_appendix, _retrieve_evidence, build_rag_evidence_for_debug, create_chat_completion, list_available_models, stream_chat_completion  # type: ignore[import-untyped]


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


class _FakeStreamClient(_FakeClient):
    def __init__(self, model_ids: list[str], deltas: list[str]) -> None:
        super().__init__(model_ids, completion_text="")
        self._deltas = deltas

    def _create_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return (
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))]
                )
                for delta in self._deltas
            )
        return super()._create_completion(**kwargs)


class _TransientRetryClient(_FakeClient):
    def __init__(self, model_ids: list[str], failures_before_success: int, completion_text: str = "ok") -> None:
        super().__init__(model_ids, completion_text=completion_text)
        self.failures_before_success = failures_before_success

    def _create_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise RuntimeError("502 Proxy Error: Error reading from remote server")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._completion_text))]
        )


class _TransientStreamRetryClient(_FakeClient):
    def __init__(self, model_ids: list[str], failures_before_success: int, deltas: list[str]) -> None:
        super().__init__(model_ids, completion_text="")
        self.failures_before_success = failures_before_success
        self._deltas = deltas

    def _create_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise RuntimeError("502 Proxy Error: Error reading from remote server")
        if kwargs.get("stream"):
            return (
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))]
                )
                for delta in self._deltas
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="".join(self._deltas)))]
        )


def test_list_available_models_marks_vision_models(monkeypatch: Any) -> None:
    fake = _FakeClient(["internvl3.5-30b-a3b", "meta-llama-3.1-8b-instruct"])
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "chat_rag_model", "qwen3-30b-a3b-instruct-2507")
    monkeypatch.setattr(settings, "translation_model", "llama-3.3-70b-instruct")
    monkeypatch.setattr(settings, "label_visual_model", "qwen3-vl-30b-a3b-instruct")
    monkeypatch.setattr(settings, "label_visual_fallback_model", "internvl3.5-30b-a3b")
    monkeypatch.setattr(settings, "paleography_verification_model", "qwen3-235b-a22b")

    payload = list_available_models()

    assert "models" in payload
    assert "internvl3.5-30b-a3b" in payload["vision_models"]
    assert payload["default_model"] == "qwen3-30b-a3b-instruct-2507"
    assert payload["task_models"]["translation_model"] == "llama-3.3-70b-instruct"


def test_create_chat_completion_injects_archai_context(monkeypatch: Any) -> None:
    fake = _FakeClient(["meta-llama-3.1-8b-instruct"], completion_text="assistant output")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "chat_rag_model", "meta-llama-3.1-8b-instruct")

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


def test_context_message_includes_authority_report_and_ocr_run_id() -> None:
    message = _context_to_system_message({
        "document_id": "doc-1",
        "ocr_run_id": "run-123",
        "authority_report": "Arthur -> Q45720",
    })

    assert "ocr_run_id: run-123" in message
    assert "authority_report_snippet:" in message
    assert "Arthur -> Q45720" in message


def test_context_message_includes_document_metadata() -> None:
    message = _context_to_system_message(
        {
            "document_id": "doc-1",
            "document_language": "Old French",
            "document_year": "1275",
            "place_or_origin": "Northern France",
            "script_family": "Gothic textualis",
            "document_type": "Arthurian romance",
            "document_notes": "Translation should use the uploaded language as the primary hint.",
        }
    )

    assert "document_language: Old French" in message
    assert "document_year: 1275" in message
    assert "place_or_origin: Northern France" in message
    assert "script_family: Gothic textualis" in message
    assert "document_type: Arthurian romance" in message
    assert "document_notes:" in message


def test_retrieve_evidence_includes_structured_entity_and_unresolved_blocks(monkeypatch: Any) -> None:
    monkeypatch.setattr("app.services.rag_store.retrieve_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr("app.services.rag_store.retrieve_entities", lambda *args, **kwargs: [])
    monkeypatch.setattr("app.services.rag_store.format_evidence_blocks", lambda hits: "")
    monkeypatch.setattr(
        "app.db.pipeline_db.list_mention_links_for_run",
        lambda run_id: [
            {
                "run_id": run_id,
                "asset_ref": "page-1",
                "surface": "Arthur",
                "canonical_label": "King Arthur",
                "entity_type": "person",
                "ent_type": "person",
                "reason": "linked via wikidata",
                "link_status": "linked",
                "confidence": 0.91,
                "wikidata_qid": "Q45720",
                "viaf_id": "24604281",
                "geonames_id": "",
                "chunk_id": "chunk-1",
                "evidence_start_offset": 0,
                "evidence_end_offset": 6,
                "bbox_json": [10, 20, 40, 60],
                "evidence_raw_text": "Arthur",
            },
            {
                "run_id": run_id,
                "asset_ref": "page-1",
                "surface": "Camelot",
                "canonical_label": "",
                "entity_type": "",
                "ent_type": "place",
                "reason": "token_search_allowed=False quality=LOW",
                "link_status": "unresolved_low_quality",
                "confidence": 0.22,
                "wikidata_qid": "",
                "viaf_id": "",
                "geonames_id": "",
                "chunk_id": "chunk-1",
                "evidence_start_offset": 15,
                "evidence_end_offset": 22,
                "bbox_json": [50, 20, 90, 60],
                "evidence_raw_text": "Camelot",
            },
        ],
    )

    evidence = _retrieve_evidence(
        [{"role": "user", "content": "Tell me about Arthur and Camelot"}],
        {"ocr_run_id": "run-1"},
    )

    assert "[LINKED_ENTITY_EVIDENCE]" in evidence
    assert "wikidata_qid: Q45720" in evidence
    assert "[UNRESOLVED_MENTION_EVIDENCE]" in evidence
    assert "reason_unresolved: token_search_allowed=False quality=LOW" in evidence


def test_retrieve_evidence_includes_entity_index_hits(monkeypatch: Any) -> None:
    monkeypatch.setattr("app.services.rag_store.retrieve_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr("app.services.rag_store.format_evidence_blocks", lambda hits: "")
    monkeypatch.setattr(
        "app.services.rag_store.retrieve_entities",
        lambda *args, **kwargs: [
            {
                "doc_id": "entity:run-1:wikidata:Q45720",
                "record_type": "entity",
                "run_id": "run-1",
                "asset_ref": "page-1",
                "entity_id": "wikidata:Q45720",
                "mention_id": "",
                "authority_source": "wikidata",
                "authority_id": "Q45720",
                "wikidata_qid": "Q45720",
                "viaf_id": "24604281",
                "geonames_id": "",
                "entity_type": "person",
                "chunk_id": "",
                "start_offset": 0,
                "end_offset": 0,
                "link_status": "linked",
                "confidence": 0.96,
                "text": "canonical_label: King Arthur",
                "distance": 0.01,
            }
        ],
    )
    monkeypatch.setattr("app.db.pipeline_db.list_mention_links_for_run", lambda run_id: [])

    evidence = _retrieve_evidence(
        [{"role": "user", "content": "Which Arthurian figures are present?"}],
        {"ocr_run_id": "run-1"},
    )

    assert "[LINKED_ENTITY_EVIDENCE]" in evidence
    assert "authority_source: wikidata" in evidence
    assert "wikidata_qid: Q45720" in evidence


def test_create_chat_completion_print_the_rag_returns_structured_inspection(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "app.services.chat_ai.build_rag_evidence_for_debug",
        lambda query, run_id, k=8: {
            "query": query,
            "run_id": run_id,
            "ocr_chunk_evidence": [{"chunk_id": "chunk-1"}],
            "linked_entity_evidence": [{"entity_id": "wikidata:Q45720"}],
            "unresolved_mention_evidence": [{"mention_id": "mention-2"}],
            "presentation_markdown": "**RAG Evidence**\n- Query: Arthur\n- OCR chunk evidence: 1",
            "debug_markdown": "**RAG Debug View**\n```json\n{\"ocr_chunk_evidence\": [{\"chunk_id\": \"chunk-1\"}]}\n```",
            "inspection_markdown": "**RAG Evidence**\n- Query: Arthur\n- OCR chunk evidence: 1",
        },
    )
    fake = _FakeClient(["qwen3-30b-a3b-instruct-2507"], completion_text="Should not be used.")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)

    result = create_chat_completion(
        messages=[
            {"role": "user", "content": "Who is mentioned on this page?"},
            {"role": "assistant", "content": "Arthur is mentioned."},
            {"role": "user", "content": "print the RAG"},
        ],
        model=None,
        context={"ocr_run_id": "run-1"},
    )

    assert result["model"] == "rag-inspection"
    assert "**RAG Evidence**" in result["text"]
    assert result["stage_metadata"]["stage_name"] == "rag_inspection"
    assert result["stage_metadata"]["mode_used"] == "presentation"
    assert result["inspection"]["evidence_used"]["ocr_chunk_evidence"] == 1
    assert fake.calls == []


def test_create_chat_completion_print_the_rag_debug_uses_debug_mode(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "app.services.chat_ai.build_rag_evidence_for_debug",
        lambda query, run_id, k=8: {
            "query": query,
            "run_id": run_id,
            "ocr_chunk_evidence": [{"chunk_id": "chunk-1"}],
            "linked_entity_evidence": [],
            "unresolved_mention_evidence": [],
            "presentation_markdown": "**RAG Evidence**\n- Query: Arthur",
            "debug_markdown": "**RAG Debug View**\n```json\n{\"ocr_chunk_evidence\": [{\"chunk_id\": \"chunk-1\"}]}\n```",
            "inspection_markdown": "**RAG Evidence**\n- Query: Arthur",
        },
    )

    result = create_chat_completion(
        messages=[{"role": "user", "content": "print the RAG debug"}],
        model=None,
        context={"ocr_run_id": "run-1"},
    )

    assert result["model"] == "rag-inspection"
    assert "**RAG Debug View**" in result["text"]
    assert result["stage_metadata"]["mode_used"] == "debug"


def test_build_rag_evidence_for_debug_formats_conservative_presentation_and_compact_debug(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "app.services.rag_store.retrieve_chunks",
        lambda query, top_k=8, run_ids=None: [
            {
                "run_id": "run-1",
                "asset_ref": "folio-1",
                "chunk_id": "chunk-1",
                "chunk_idx": 0,
                "start_offset": 0,
                "end_offset": 32,
                "text": "Li rois Artus vint a la cité.",
            },
            {
                "run_id": "run-1",
                "asset_ref": "folio-1",
                "chunk_id": "chunk-2",
                "chunk_idx": 1,
                "start_offset": 33,
                "end_offset": 60,
                "text": "Merlin parla a Artus.",
            },
        ],
    )
    monkeypatch.setattr(
        "app.services.rag_store.retrieve_entities",
        lambda query, top_k=4, run_ids=None: [],
    )
    monkeypatch.setattr(
        "app.db.pipeline_db.get_run",
        lambda run_id: {
            "asset_ref": "folio-1",
            "detected_language": "Old French",
            "proofread_text": "Li rois Artus vint a la cité. Merlin parla a Artus.",
            "ocr_text": "",
        },
    )
    monkeypatch.setattr("app.db.pipeline_db.list_mention_links_for_run", lambda run_id: [])

    payload = build_rag_evidence_for_debug("Arthur", "run-1")

    assert "**Document Metadata**" in payload["presentation_markdown"]
    assert "**Extraction Summary**" in payload["presentation_markdown"]
    assert "**Transcript Snippet**" in payload["presentation_markdown"]
    assert "likely poetic" not in payload["presentation_markdown"].lower()
    assert "moral allegory" not in payload["presentation_markdown"].lower()
    assert "\"authority_summary\"" in payload["debug_markdown"]
    assert "\"ocr_chunk_evidence_sample\"" in payload["debug_markdown"]
    assert "\"omitted_counts\"" in payload["debug_markdown"]


def test_create_chat_completion_retries_on_model_not_found(monkeypatch: Any) -> None:
    fake = _FakeRetryClient(
        ["internvl3.5-30b-a3b", "qwen3-vl-30b-a3b-instruct"],
        failing_model="internvl3.5-30b-a3b",
        completion_text="retry-output",
    )
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "chat_rag_model", "qwen3-30b-a3b-instruct-2507")
    monkeypatch.setattr(settings, "label_visual_model", "qwen3-vl-30b-a3b-instruct")
    monkeypatch.setattr(settings, "label_visual_fallback_model", "internvl3.5-30b-a3b")

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


def test_create_chat_completion_appends_verification(monkeypatch: Any) -> None:
    fake = _FakeClient(["meta-llama-3.1-8b-instruct"], completion_text="Draft answer.")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "chat_rag_model", "meta-llama-3.1-8b-instruct")
    monkeypatch.setattr("app.services.chat_ai._retrieve_evidence", lambda *args, **kwargs: "[EVIDENCE]\ntext: Arthur\n[/EVIDENCE]")
    monkeypatch.setattr(
        "app.services.chat_ai._run_paleography_verification",
        lambda **kwargs: {
            "assessment": "supported",
            "corrected_answer": "Verified answer.",
            "notes": ["Matches OCR evidence."],
            "citations_checked": ["chunk-1"],
            "model_used": "mistral-large-3-675b-instruct-2512",
        },
    )

    result = create_chat_completion(
        messages=[{"role": "user", "content": "What does the passage say about Arthur?"}],
        model=None,
        context={"ocr_run_id": "run-1", "transcript": "Arthur", "authority_report": "Arthur -> Q45720"},
    )

    assert result["verification"]["assessment"] == "supported"
    assert "[Verification]" in result["text"]
    assert "Verified answer: Verified answer." in result["text"]


def test_create_chat_completion_retries_transient_proxy_error(monkeypatch: Any) -> None:
    fake = _TransientRetryClient(["meta-llama-3.1-8b-instruct"], failures_before_success=1, completion_text="Recovered.")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "chat_rag_model", "meta-llama-3.1-8b-instruct")
    monkeypatch.setattr("app.services.chat_ai.time.sleep", lambda seconds: None)

    result = create_chat_completion(
        messages=[{"role": "user", "content": "Translate page to English"}],
        model=None,
    )

    assert result["text"] == "Recovered."
    assert len(fake.calls) == 2


def test_create_chat_completion_returns_clean_degraded_verification_when_verifier_fails(monkeypatch: Any) -> None:
    fake = _FakeClient(["qwen3-30b-a3b-instruct-2507"], completion_text="Draft answer.")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "chat_rag_model", "qwen3-30b-a3b-instruct-2507")
    monkeypatch.setattr("app.services.chat_ai._retrieve_evidence", lambda *args, **kwargs: "[OCR_CHUNK_EVIDENCE]\ntext: Arthur\n[/OCR_CHUNK_EVIDENCE]")
    monkeypatch.setattr(
        "app.agents.paleography_verification_agent.PaleographyVerificationAgent.run",
        lambda self, payload: (_ for _ in ()).throw(RuntimeError("bad verifier output")),
    )

    result = create_chat_completion(
        messages=[{"role": "user", "content": "What does the page say?"}],
        model=None,
        context={"ocr_run_id": "run-1", "transcript": "Arthur"},
    )

    assert result["verification"]["assessment"] == "unavailable"
    assert result["verification"]["corrected_answer"] == "Draft answer."
    assert result["verification"]["notes"] == ["Verifier response could not be parsed cleanly."]
    assert "[Verification]" in result["text"]
    assert "Assessment: unavailable" in result["text"]
    assert "Verified answer: Draft answer." in result["text"]
    assert "Notes: Verifier response could not be parsed cleanly." in result["text"]
    assert "Citations checked: none" in result["text"]


def test_create_chat_completion_uses_translation_stage_model_and_skips_verification(monkeypatch: Any) -> None:
    fake = _FakeClient(["llama-3.3-70b-instruct"], completion_text="English translation.")
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "translation_model", "llama-3.3-70b-instruct")

    result = create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Translate the extracted transcript into English."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,ignored"}},
                ],
            }
        ],
        model="qwen3-vl-30b-a3b-instruct",
        context={
            "chat_stage": "translation",
            "ocr_run_id": "run-1",
            "transcript": "Li rois Artus",
            "document_language": "Old French",
        },
    )

    assert result["text"] == "English translation."
    assert fake.calls[0]["model"] == "llama-3.3-70b-instruct"
    assert fake.calls[0]["temperature"] == 0.0
    assert "Translate the authoritative transcript from Old French into English as a coherent passage." in fake.calls[0]["messages"][0]["content"]
    assert fake.calls[0]["messages"][1]["content"] == "Translate the extracted transcript into English."
    assert result["verification"] is None
    assert result["stage_metadata"]["stage_name"] == "translation"
    assert result["stage_metadata"]["model_used"] == "llama-3.3-70b-instruct"
    assert result["inspection"]["evidence_used"]["page_image_used"] is False
    assert result["inspection"]["input_source_summary"]["source_type"] == "extracted_transcript"
    assert result["inspection"]["input_source_summary"]["source_language"] == "Old French"
    assert result["inspection"]["input_source_summary"]["target_language"] == "English"


def test_create_chat_completion_translation_normalizes_page_description_noise(monkeypatch: Any) -> None:
    fake = _FakeClient(
        ["llama-3.3-70b-instruct"],
        completion_text="The page appears damaged.\nEnglish translation: The king goes to the city.",
    )
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "translation_model", "llama-3.3-70b-instruct")

    result = create_chat_completion(
        messages=[{"role": "user", "content": "Translate this passage into English."}],
        model=None,
        context={
            "chat_stage": "translation",
            "ocr_run_id": "run-1",
            "transcript": "Li rois vait a la cité.",
            "document_language": "Old French",
        },
    )

    assert result["text"] == "The king goes to the city."
    assert "page appears" not in result["text"].lower()


def test_create_chat_completion_translation_strips_trailing_note_commentary(monkeypatch: Any) -> None:
    fake = _FakeClient(
        ["llama-3.3-70b-instruct"],
        completion_text="English translation: The king goes to the city. Note: Some parts of the translation may be uncertain.",
    )
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "translation_model", "llama-3.3-70b-instruct")

    result = create_chat_completion(
        messages=[{"role": "user", "content": "Translate this passage into English."}],
        model=None,
        context={
            "chat_stage": "translation",
            "ocr_run_id": "run-1",
            "transcript": "Li rois vait a la cité.",
            "document_language": "Old French",
        },
    )

    assert result["text"] == "The king goes to the city."
    assert "note:" not in result["text"].lower()


def test_stream_chat_completion_normalizes_cumulative_chunks(monkeypatch: Any) -> None:
    fake = _FakeStreamClient(
        ["meta-llama-3.1-8b-instruct"],
        deltas=["T", "Tr", "Tra", "Trans", "Transl", "Translate"],
    )
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "chat_rag_model", "meta-llama-3.1-8b-instruct")

    events = list(
        stream_chat_completion(
            messages=[{"role": "user", "content": "Translate this."}],
            model=None,
        )
    )

    assert "".join(event["delta"] for event in events if event["type"] == "delta") == "Translate"
    assert events[-1]["type"] == "done"
    assert events[-1]["text"] == "Translate"


def test_stream_chat_completion_preserves_incremental_chunks(monkeypatch: Any) -> None:
    fake = _FakeStreamClient(
        ["meta-llama-3.1-8b-instruct"],
        deltas=["Trans", "late", " this"],
    )
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "chat_rag_model", "meta-llama-3.1-8b-instruct")

    events = list(
        stream_chat_completion(
            messages=[{"role": "user", "content": "Translate this."}],
            model=None,
        )
    )

    assert [event["delta"] for event in events if event["type"] == "delta"] == ["Trans", "late", " this"]
    assert events[-1]["type"] == "done"
    assert events[-1]["text"] == "Translate this"


def test_stream_chat_completion_translation_buffers_and_normalizes_final_output(monkeypatch: Any) -> None:
    fake = _FakeStreamClient(
        ["llama-3.3-70b-instruct"],
        deltas=["The page ", "appears damaged.\n", "English translation: The king goes."],
    )
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "translation_model", "llama-3.3-70b-instruct")

    events = list(
        stream_chat_completion(
            messages=[{"role": "user", "content": "Translate this."}],
            model=None,
            context={"chat_stage": "translation", "transcript": "Li rois vait."},
        )
    )

    deltas = [event["delta"] for event in events if event["type"] == "delta"]
    assert deltas == ["The king goes."]
    assert events[-1]["type"] == "done"
    assert events[-1]["text"] == "The king goes."
    assert events[-1]["verification"] is None


def test_stream_chat_completion_appends_verification_delta(monkeypatch: Any) -> None:
    fake = _FakeStreamClient(
        ["meta-llama-3.1-8b-instruct"],
        deltas=["Arthur", " appears"],
    )
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "chat_rag_model", "meta-llama-3.1-8b-instruct")
    monkeypatch.setattr("app.services.chat_ai._retrieve_evidence", lambda *args, **kwargs: "[EVIDENCE]\ntext: Arthur\n[/EVIDENCE]")
    monkeypatch.setattr(
        "app.services.chat_ai._run_paleography_verification",
        lambda **kwargs: {
            "assessment": "partially_supported",
            "corrected_answer": "Arthur appears in the evidence.",
            "notes": ["Reading is somewhat uncertain."],
            "citations_checked": ["chunk-1"],
            "model_used": "mistral-large-3-675b-instruct-2512",
        },
    )

    events = list(
        stream_chat_completion(
            messages=[{"role": "user", "content": "What does the page say?"}],
            model=None,
            context={"ocr_run_id": "run-1", "transcript": "Arthur"},
        )
    )

    deltas = [event["delta"] for event in events if event["type"] == "delta"]
    assert deltas[0] == "Arthur"
    assert any("[Verification]" in delta for delta in deltas)
    assert events[-1]["verification"]["assessment"] == "partially_supported"
    assert "[Verification]" in events[-1]["text"]
    assert events[-1]["stage_metadata"]["stage_name"] == "rag_chat"
    assert events[-1]["inspection"]["evidence_used"]["ocr_chunk_evidence"] == 1


def test_format_verification_appendix_strips_reasoning_markers() -> None:
    appendix = _format_verification_appendix(
        {
            "assessment": "supported",
            "corrected_answer": "<think>hidden</think> Verified answer.",
            "notes": ["Reasoning: secret", "Transcript supports the answer."],
            "citations_checked": ["```chunk-1```", "ENTITY_EVIDENCE Arthur"],
            "model_used": "qwen3-235b-a22b",
        }
    )

    assert "[Verification]" in appendix
    assert "Verified answer: Verified answer." in appendix
    assert "Transcript supports the answer." in appendix
    assert "Citations checked: chunk-1 | ENTITY_EVIDENCE Arthur" in appendix
    assert "Reasoning:" not in appendix
    assert "<think>" not in appendix
    assert "Verifier model: qwen3-235b-a22b" in appendix


def test_stream_chat_completion_retries_transient_proxy_error(monkeypatch: Any) -> None:
    fake = _TransientStreamRetryClient(
        ["meta-llama-3.1-8b-instruct"],
        failures_before_success=1,
        deltas=["Translated", " page"],
    )
    monkeypatch.setattr("app.services.chat_ai._create_client", lambda: fake)
    monkeypatch.setattr(settings, "chat_rag_model", "meta-llama-3.1-8b-instruct")
    monkeypatch.setattr("app.services.chat_ai.time.sleep", lambda seconds: None)

    events = list(
        stream_chat_completion(
            messages=[{"role": "user", "content": "Translate page to English"}],
            model=None,
        )
    )

    assert "".join(event["delta"] for event in events if event["type"] == "delta") == "Translated page"
    assert len(fake.calls) == 2
    assert events[-1]["text"] == "Translated page"


def test_chat_label_analysis_route_uses_dedicated_agent(monkeypatch: Any) -> None:
    class _FakeAgent:
        def run(self, payload: Any) -> Any:
            assert payload.label_name == "Embellished"
            return chat_router.LabelAnalysisResponse(
                status="ok",
                text="Decorated initial.",
                label_name=payload.label_name,
                model_used="internvl3.5-30b-a3b",
                warnings=[],
                region_count=len(payload.regions),
                crop_image_b64="abc123",
                crop_bounds_xyxy=[1, 2, 3, 4],
            )

    monkeypatch.setattr(chat_router, "_get_label_analysis_agent", lambda: _FakeAgent())

    result = asyncio.run(
        chat_router.chat_label_analysis(
            chat_router.LabelAnalysisRequest(
                question="What is this embellished initial?",
                label_name="Embellished",
                image_b64="abc123",
                regions=[
                    chat_router.LabelRegionPayload(
                        region_id="ann-1",
                        bbox_xyxy=[10, 20, 40, 60],
                        polygons=[],
                    )
                ],
            )
        )
    )

    assert result.status == "ok"
    assert result.text == "Decorated initial."
    assert result.region_count == 1
