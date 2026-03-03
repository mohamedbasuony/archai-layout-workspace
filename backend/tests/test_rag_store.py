"""Tests for the RAG store service (rag_store.py).

Covers: indexing, retrieval, deletion, evidence formatting, and
the auto-index hook wiring.  All tests use a temporary ChromaDB
and an in-memory SQLite pipeline DB — no external services needed.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Ensure the backend package is importable
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_chroma(tmp_path: Path, monkeypatch: Any) -> None:
    """Point ChromaDB and SQLite at a temp directory for every test."""
    monkeypatch.setattr("app.config.settings.chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr("app.config.settings.rag_collection_name", "test_chunks")
    monkeypatch.setattr("app.config.settings.rag_auto_index", True)

    # Use in-memory SQLite for the pipeline DB
    db_path = str(tmp_path / "test.sqlite")
    monkeypatch.setenv("ARCHAI_DB_PATH", db_path)

    # Reset singletons so each test starts fresh
    import app.services.rag_store as _rs
    _rs._CLIENT = None
    import app.db.pipeline_db as _pdb
    _pdb._DB_READY = False


@pytest.fixture()
def sample_run(tmp_path: Path) -> str:
    """Create a pipeline run with sample chunks in SQLite."""
    from app.db.pipeline_db import create_run, insert_chunks, update_run_fields

    run_id = create_run("test_asset.png", "abc123sha")
    update_run_fields(
        run_id,
        status="COMPLETED",
        current_stage="DONE",
        detected_language="la",
        ocr_text="In nomine domini amen. Hic est liber antiquus.",
    )
    insert_chunks(run_id, [
        {"idx": 0, "start_offset": 0, "end_offset": 24, "text": "In nomine domini amen."},
        {"idx": 1, "start_offset": 25, "end_offset": 47, "text": "Hic est liber antiquus."},
    ])
    return run_id


# ── Unit tests ──────────────────────────────────────────────────────────

class TestIndexRun:
    def test_index_creates_vectors(self, sample_run: str) -> None:
        from app.services.rag_store import index_run, is_run_indexed

        result = index_run(sample_run)
        assert result["status"] == "ok"
        assert result["chunks_indexed"] == 2
        assert result["chunks_total"] == 2
        assert result["chunks_skipped"] == 0
        assert result["collection_name"] == "test_chunks"
        assert result["collection_count_after"] >= 2

        status = is_run_indexed(sample_run)
        assert status["indexed"] is True
        assert status["chunks_indexed"] == 2
        assert status["missing_chunk_ids"] == []

    def test_index_unknown_run_raises(self) -> None:
        from app.services.rag_store import index_run

        with pytest.raises(ValueError, match="not found"):
            index_run("nonexistent-run-id")

    def test_index_empty_run(self, tmp_path: Path) -> None:
        from app.db.pipeline_db import create_run
        from app.services.rag_store import index_run

        run_id = create_run("empty_asset.png")
        result = index_run(run_id)
        assert result["status"] == "empty"
        assert result["chunks_indexed"] == 0
        assert result["chunks_total"] == 0

    def test_index_upsert_is_idempotent(self, sample_run: str) -> None:
        from app.services.rag_store import index_run, is_run_indexed

        index_run(sample_run)
        index_run(sample_run)  # second call should not duplicate
        status = is_run_indexed(sample_run)
        assert status["chunks_indexed"] == 2


class TestRetrieveChunks:
    def test_basic_retrieval(self, sample_run: str) -> None:
        from app.services.rag_store import index_run, retrieve_chunks

        index_run(sample_run)
        hits = retrieve_chunks("nomine domini", top_k=2)
        assert len(hits) > 0
        assert hits[0]["run_id"] == sample_run
        assert "text" in hits[0]
        assert "distance" in hits[0]

    def test_retrieval_scoped_by_run_id(self, sample_run: str) -> None:
        from app.services.rag_store import index_run, retrieve_chunks

        index_run(sample_run)
        hits = retrieve_chunks("nomine", run_ids=[sample_run])
        assert all(h["run_id"] == sample_run for h in hits)

    def test_retrieval_no_results_for_wrong_scope(self, sample_run: str) -> None:
        from app.services.rag_store import index_run, retrieve_chunks

        index_run(sample_run)
        hits = retrieve_chunks("nomine", run_ids=["totally-wrong-id"])
        assert len(hits) == 0

    def test_retrieval_returns_metadata(self, sample_run: str) -> None:
        from app.services.rag_store import index_run, retrieve_chunks

        index_run(sample_run)
        hits = retrieve_chunks("liber antiquus", top_k=1)
        assert len(hits) >= 1
        h = hits[0]
        assert "chunk_id" in h
        assert "asset_ref" in h
        assert "chunk_idx" in h
        assert "start_offset" in h
        assert "end_offset" in h


class TestDeleteRun:
    def test_delete_removes_vectors(self, sample_run: str) -> None:
        from app.services.rag_store import index_run, delete_run, is_run_indexed

        index_run(sample_run)
        deleted = delete_run(sample_run)
        assert deleted == 2

        status = is_run_indexed(sample_run)
        assert status["indexed"] is False
        assert status["chunks_indexed"] == 0
        assert len(status["missing_chunk_ids"]) == 2


class TestFormatEvidenceBlocks:
    def test_empty_list(self) -> None:
        from app.services.rag_store import format_evidence_blocks

        assert format_evidence_blocks([]) == ""

    def test_single_block(self) -> None:
        from app.services.rag_store import format_evidence_blocks

        chunks = [{
            "run_id": "r1",
            "asset_ref": "test.png",
            "chunk_id": "c1",
            "chunk_idx": 0,
            "start_offset": 0,
            "end_offset": 10,
            "text": "Hello world",
        }]
        result = format_evidence_blocks(chunks)
        assert "[EVIDENCE]" in result
        assert "[/EVIDENCE]" in result
        assert "run_id: r1" in result
        assert "chunk_id: c1" in result
        assert "offsets: 0-10" in result
        assert "text: Hello world" in result

    def test_multiple_blocks(self) -> None:
        from app.services.rag_store import format_evidence_blocks

        chunks = [
            {"run_id": "r1", "asset_ref": "a", "chunk_id": "c1",
             "chunk_idx": 0, "start_offset": 0, "end_offset": 5, "text": "one"},
            {"run_id": "r1", "asset_ref": "a", "chunk_id": "c2",
             "chunk_idx": 1, "start_offset": 6, "end_offset": 10, "text": "two"},
        ]
        result = format_evidence_blocks(chunks)
        assert result.count("[EVIDENCE]") == 2
        assert result.count("[/EVIDENCE]") == 2


class TestChatAIEvidenceInjection:
    """Verify that _prepare_messages injects evidence when RAG has data."""

    def test_prepare_messages_adds_evidence_when_indexed(
        self, sample_run: str, monkeypatch: Any
    ) -> None:
        from app.services.rag_store import index_run
        from app.services.chat_ai import _prepare_messages

        index_run(sample_run)

        context = {"run_id": sample_run}
        messages = [{"role": "user", "content": "What does the text say about domini?"}]
        prepared = _prepare_messages(messages, context)

        # The system message should now contain EVIDENCE
        system_msg = prepared[0]["content"]
        assert "[EVIDENCE]" in system_msg
        assert "run_id:" in system_msg

    def test_prepare_messages_works_without_rag(self) -> None:
        from app.services.chat_ai import _prepare_messages

        messages = [{"role": "user", "content": "Hello"}]
        prepared = _prepare_messages(messages, None)
        assert prepared[0]["role"] == "system"
        assert "[EVIDENCE]" not in prepared[0]["content"]


class TestRetrieveDebug:
    def test_returns_structured_payload(self, sample_run: str) -> None:
        from app.services.rag_store import index_run, retrieve_debug

        index_run(sample_run)
        payload = retrieve_debug("nomine domini", k=2, run_id=sample_run)
        assert payload["query"] == "nomine domini"
        assert payload["k"] == 2
        assert payload["filter"]["run_id"] == sample_run
        assert len(payload["results"]) > 0
        hit = payload["results"][0]
        assert "chunk_id" in hit
        assert "offsets" in hit
        assert "score" in hit
        assert "text_preview" in hit

    def test_filters_by_asset_ref(self, sample_run: str) -> None:
        from app.services.rag_store import index_run, retrieve_debug

        index_run(sample_run)
        payload = retrieve_debug("nomine", asset_ref="test_asset.png")
        assert len(payload["results"]) > 0
        # wrong asset → no results
        payload2 = retrieve_debug("nomine", asset_ref="no_such.png")
        assert len(payload2["results"]) == 0


class TestRagDebugFlag:
    def test_is_rag_debug_enabled(self, monkeypatch: Any) -> None:
        from app.services.rag_debug import is_rag_debug_enabled, is_rag_debug_verbose

        monkeypatch.setenv("ARCHAI_DEBUG_RAG", "1")
        assert is_rag_debug_enabled() is True
        assert is_rag_debug_verbose() is False

        monkeypatch.setenv("ARCHAI_DEBUG_RAG", "verbose")
        assert is_rag_debug_verbose() is True

        monkeypatch.setenv("ARCHAI_DEBUG_RAG", "")
        assert is_rag_debug_enabled() is False

    def test_debug_log_functions_do_not_error(self, sample_run: str, monkeypatch: Any) -> None:
        monkeypatch.setenv("ARCHAI_DEBUG_RAG", "verbose")
        from app.services.rag_debug import log_index_done, log_index_status, log_chat_evidence

        log_index_done({"run_id": "x", "chunks_indexed": 1})
        log_index_status({"run_id": "x", "chunks_indexed": 1})
        log_chat_evidence(
            run_id="x", asset_ref="a.png", k=1,
            evidence_ids=[{"chunk_id": "c1", "chunk_idx": 0, "offsets": "0-5"}],
            query="test query",
            full_evidence_text="some text",
        )


class TestChatCitationInstruction:
    def test_system_prompt_contains_citation_rule(self, sample_run: str) -> None:
        from app.services.rag_store import index_run
        from app.services.chat_ai import _prepare_messages

        index_run(sample_run)
        context = {"run_id": sample_run}
        messages = [{"role": "user", "content": "Quote the text about domini"}]
        prepared = _prepare_messages(messages, context)
        system_msg = prepared[0]["content"]
        assert "(asset_ref=" in system_msg
        assert "Do not invent quotes" in system_msg
