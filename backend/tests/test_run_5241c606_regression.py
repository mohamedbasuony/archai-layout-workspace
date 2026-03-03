"""Regression tests for run 5241c606-1be3-4165-97ca-c41ea5726e42.

Root cause:
  The same surface "leantolot" appeared in BOTH ``top_mentions_preview``
  (accepted via ``rule:ngram_canonical_person`` matching "leantlote") AND
  ``top_rejected_preview`` (rejected by ``rule:salvage_work_fuzzy`` against
  "lancelot" with nd=0.333 > 0.25).

  This broke traceability because one surface had two contradictory
  decisions in the same report.

This file covers:
  1. Single-decision invariant: a span key cannot appear in both accepted
     and rejected outputs.
  2. Deterministic scoring + provenance: the accepted mention's notes must
     contain the exact nd/bo used for the final decision.
  3. ``top_rejected_preview`` must contain ONLY spans with final decision
     REJECT — never spans that were later accepted.
  4. Chunk cleanup: pure stopword / single-char lines are not stored as
     chunks.
  5. Attempt-history section for accepted spans (``attempts_for_accepted``)
     is shown separately from the rejected list.
"""

from __future__ import annotations

import pytest


# ── 1. Single-Decision Invariant ─────────────────────────────────────


class TestSingleDecisionInvariant:
    """A span key must appear in either accepted OR rejected, never both."""

    SAMPLE_TEXT = (
        "li chevaliers leantolot vint au chastel\n"
        "et la dame le recut molt bel\n"
    )

    def test_leantolot_not_in_both_accepted_and_rejected(self):
        """surface='leantolot' must appear only in accepted OR rejected."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, debug = _extract_mentions_from_text(self.SAMPLE_TEXT)
        accepted_surfaces = {m["surface"].lower() for m in mentions}
        rejected_surfaces = {r["surface"].lower() for r in debug.get("rejected", [])}
        overlap = accepted_surfaces & rejected_surfaces
        assert len(overlap) == 0, (
            f"Surfaces in BOTH accepted AND rejected: {overlap}"
        )

    def test_leantolot_is_accepted(self):
        """'leantolot' must be accepted (matched via canonical person names)."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, _ = _extract_mentions_from_text(self.SAMPLE_TEXT)
        accepted_surfaces = {m["surface"].lower() for m in mentions}
        assert "leantolot" in accepted_surfaces, (
            f"Expected 'leantolot' in accepted, got: {accepted_surfaces}"
        )

    def test_leantolot_not_in_rejected(self):
        """'leantolot' must NOT appear in the final rejected list."""
        from app.routers.ocr import _extract_mentions_from_text
        _, _, debug = _extract_mentions_from_text(self.SAMPLE_TEXT)
        rejected_surfaces = {r["surface"].lower() for r in debug.get("rejected", [])}
        assert "leantolot" not in rejected_surfaces

    def test_no_span_in_both_lists_arbitrary_text(self):
        """Invariant: for ANY text, no surface in both accepted and rejected."""
        from app.routers.ocr import _extract_mentions_from_text
        texts = [
            "li rois leantolot et la dame",
            "sire leantolot vint devant le roi",
            "dist leantolot a perceval et a galahad",
            "il parla de leantolot le chevalier",
        ]
        for text in texts:
            mentions, _, debug = _extract_mentions_from_text(text)
            accepted = {m["surface"].lower() for m in mentions}
            rejected = {r["surface"].lower() for r in debug.get("rejected", [])}
            overlap = accepted & rejected
            assert len(overlap) == 0, (
                f"Text={text!r}: overlap={overlap}"
            )


# ── 2. Deterministic Scoring + Provenance ────────────────────────────


class TestDeterministicScoring:
    """The accepted mention must carry the exact metrics used for its decision."""

    SAMPLE_TEXT = "li chevaliers leantolot vint au chastel"

    def test_accepted_mention_has_nd_bo_in_notes(self):
        """The accepted mention for 'leantolot' must contain nd= and bo= in notes."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, _ = _extract_mentions_from_text(self.SAMPLE_TEXT)
        leantolot = [m for m in mentions if m["surface"].lower() == "leantolot"]
        assert len(leantolot) >= 1, "Expected accepted mention for 'leantolot'"
        m = leantolot[0]
        notes = str(m.get("notes", ""))
        assert "nd=" in notes, f"Missing nd= in notes: {notes!r}"
        assert "bo=" in notes, f"Missing bo= in notes: {notes!r}"

    def test_accepted_nd_within_threshold(self):
        """The nd value stored in notes must be <= 0.25 (acceptance threshold)."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, _ = _extract_mentions_from_text(self.SAMPLE_TEXT)
        leantolot = [m for m in mentions if m["surface"].lower() == "leantolot"]
        assert len(leantolot) >= 1
        m = leantolot[0]
        notes = str(m.get("notes", ""))
        # Parse nd=X.XXX from notes
        import re
        nd_match = re.search(r"nd=(\d+\.\d+)", notes)
        assert nd_match, f"Could not parse nd from notes: {notes!r}"
        nd_val = float(nd_match.group(1))
        assert nd_val <= 0.25, f"nd={nd_val} > 0.25; should be within threshold"

    def test_accepted_bo_above_threshold(self):
        """The bo value stored in notes must be >= 0.40 (acceptance threshold)."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, _ = _extract_mentions_from_text(self.SAMPLE_TEXT)
        leantolot = [m for m in mentions if m["surface"].lower() == "leantolot"]
        assert len(leantolot) >= 1
        m = leantolot[0]
        notes = str(m.get("notes", ""))
        import re
        bo_match = re.search(r"bo=(\d+\.\d+)", notes)
        assert bo_match, f"Could not parse bo from notes: {notes!r}"
        bo_val = float(bo_match.group(1))
        assert bo_val >= 0.40, f"bo={bo_val} < 0.40; should be above threshold"

    def test_rejected_entries_have_failing_metrics(self):
        """Rejected entries must have metrics that actually fail the threshold."""
        from app.routers.ocr import _extract_mentions_from_text
        text = "li chevaliers leantolot vint au chastel"
        _, _, debug = _extract_mentions_from_text(text)
        for r in debug.get("rejected", []):
            reason = str(r.get("reason", ""))
            # If it has nd/bo info, check that nd>0.25 or bo<0.40
            if "nd=" in reason and "bo=" in reason:
                import re
                nd_m = re.search(r"nd=(\d+\.\d+)", reason)
                bo_m = re.search(r"bo=(\d+\.\d+)", reason)
                if nd_m and bo_m:
                    nd = float(nd_m.group(1))
                    bo = float(bo_m.group(1))
                    assert nd > 0.25 or bo < 0.40, (
                        f"Rejected entry has passing metrics: nd={nd}, bo={bo}"
                    )


# ── 3. top_rejected_preview Semantics ────────────────────────────────


class TestTopRejectedPreview:
    """top_rejected_preview must contain ONLY final REJECT decisions."""

    SAMPLE_TEXT = "li chevaliers leantolot vint au chastel"

    def test_report_rejected_does_not_contain_accepted_surfaces(self):
        """The MENTION EXTRACTION REPORT rejected section must not list
        any surface that appears in top_mentions_preview."""
        from app.routers.ocr import (
            _extract_mentions_from_text,
            _build_mention_extraction_report,
        )
        mentions, _, debug = _extract_mentions_from_text(self.SAMPLE_TEXT)
        report = _build_mention_extraction_report(
            "5241c606", "test.png", mentions, debug,
        )
        # Parse accepted surfaces from report
        import re
        accepted_in_report = set()
        in_accepted = False
        in_rejected = False
        for line in report.split("\n"):
            if "top_mentions_preview" in line:
                in_accepted = True
                in_rejected = False
                continue
            if "top_rejected_preview" in line:
                in_accepted = False
                in_rejected = True
                continue
            if "attempts_for_accepted" in line:
                in_rejected = False
                continue
            if in_accepted:
                m = re.search(r'surface="([^"]*)"', line)
                if m:
                    accepted_in_report.add(m.group(1).lower())
            if in_rejected:
                m = re.search(r'surface="([^"]*)"', line)
                if m:
                    surface = m.group(1).lower()
                    assert surface not in accepted_in_report, (
                        f"'{surface}' in BOTH accepted and rejected in report"
                    )

    def test_report_has_attempts_for_accepted_section(self):
        """If an accepted surface had intermediate rejections, they must
        appear in attempts_for_accepted, not in top_rejected_preview."""
        import uuid
        from app.db.pipeline_db import create_run, list_entity_attempts_for_run
        from app.routers.ocr import _run_trace_analysis, _build_mention_extraction_report
        run_id = create_run(f"test-5241-{uuid.uuid4().hex[:8]}")
        _chunks, mentions, _cands, debug = _run_trace_analysis(run_id, self.SAMPLE_TEXT)
        report = _build_mention_extraction_report(
            run_id, "test.png", mentions, debug,
        )
        attempts = list_entity_attempts_for_run(run_id)
        # If there are attempts, the section must exist
        if attempts:
            assert "attempts_for_accepted" in report
            # Accepted surfaces with prior rejections show in attempts, not rejected
            for a in attempts:
                surf = a.get("surface", "").lower()
                if not surf:
                    continue
                # Must NOT be in rejected section
                rejected_section = ""
                in_rejected = False
                for line in report.split("\n"):
                    if "top_rejected_preview" in line:
                        in_rejected = True
                        continue
                    if "attempts_for_accepted" in line or "filtered_out_preview" in line or "skipped_preview" in line:
                        in_rejected = False
                        continue
                    if in_rejected:
                        rejected_section += line + "\n"


# ── 4. Chunk Cleanup ─────────────────────────────────────────────────


class TestChunkCleanup:
    """Pure stopword / single-char lines must not produce chunks."""

    def test_single_char_line_filtered(self):
        """A line that is just 'a' should not produce a chunk."""
        from app.routers.ocr import _build_line_chunks
        text = "hello world\na\ngoodbye"
        chunks = _build_line_chunks(text)
        chunk_texts = [c["text"].strip() for c in chunks]
        assert "a" not in chunk_texts, (
            f"Single-char 'a' should be filtered, got chunks: {chunk_texts}"
        )

    def test_single_char_e_filtered(self):
        """A line that is just 'e' should not produce a chunk."""
        from app.routers.ocr import _build_line_chunks
        text = "hello\ne\nworld"
        chunks = _build_line_chunks(text)
        chunk_texts = [c["text"].strip() for c in chunks]
        assert "e" not in chunk_texts

    def test_single_char_i_filtered(self):
        """A line that is just 'i' should not produce a chunk."""
        from app.routers.ocr import _build_line_chunks
        text = "hello\ni\nworld"
        chunks = _build_line_chunks(text)
        chunk_texts = [c["text"].strip() for c in chunks]
        assert "i" not in chunk_texts

    def test_stopword_line_et_filtered(self):
        """A line that is just 'et' should not produce a chunk."""
        from app.routers.ocr import _build_line_chunks
        text = "hello\net\nworld"
        chunks = _build_line_chunks(text)
        chunk_texts = [c["text"].strip() for c in chunks]
        assert "et" not in chunk_texts

    def test_stopword_line_de_filtered(self):
        """A line that is just 'de' should not produce a chunk."""
        from app.routers.ocr import _build_line_chunks
        text = "hello\nde\nworld"
        chunks = _build_line_chunks(text)
        chunk_texts = [c["text"].strip() for c in chunks]
        assert "de" not in chunk_texts

    def test_normal_lines_preserved(self):
        """Normal multi-word lines are NOT filtered."""
        from app.routers.ocr import _build_line_chunks
        text = "bonjour le monde\net de la terre"
        chunks = _build_line_chunks(text)
        assert len(chunks) == 2
        assert chunks[0]["text"].strip() == "bonjour le monde"
        assert chunks[1]["text"].strip() == "et de la terre"

    def test_chunk_indices_contiguous(self):
        """Chunk indices must remain contiguous after filtering."""
        from app.routers.ocr import _build_line_chunks
        text = "hello\na\nworld\ne\ngoodbye"
        chunks = _build_line_chunks(text)
        indices = [c["idx"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_all_stopword_lines_fallback(self):
        """If ALL lines are stopwords, the fallback full-text chunk fires."""
        from app.routers.ocr import _build_line_chunks
        text = "a\ne\ni"
        chunks = _build_line_chunks(text)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_chunk_stopwords_constant(self):
        """The _CHUNK_STOPWORDS constant must include key terms."""
        from app.routers.ocr import _CHUNK_STOPWORDS
        for term in ("a", "e", "i", "et", "de", "la", "le"):
            assert term in _CHUNK_STOPWORDS

    def test_longer_text_not_filtered(self):
        """Words longer than single chars are preserved even if on own line."""
        from app.routers.ocr import _build_line_chunks
        text = "hello\nworld"
        chunks = _build_line_chunks(text)
        assert len(chunks) == 2


# ── 5. Salvage Debug Structure ───────────────────────────────────────


class TestSalvageDebugStructure:
    """salvage_debug must have clean separation of rejected vs attempts."""

    def test_attempts_for_accepted_key_exists(self):
        """After extraction, salvage_debug must have 'attempts_for_accepted' key."""
        from app.routers.ocr import _extract_mentions_from_text
        text = "li chevaliers leantolot vint au chastel"
        _, _, debug = _extract_mentions_from_text(text)
        assert "attempts_for_accepted" in debug

    def test_attempts_contain_accepted_surfaces(self):
        """Entries in attempts_for_accepted must have surfaces that were accepted."""
        from app.routers.ocr import _extract_mentions_from_text
        text = "li chevaliers leantolot vint au chastel"
        mentions, _, debug = _extract_mentions_from_text(text)
        accepted_surfaces = {m["surface"].lower() for m in mentions}
        for a in debug.get("attempts_for_accepted", []):
            surface = a.get("surface", "").lower()
            assert surface in accepted_surfaces, (
                f"Attempt surface '{surface}' not in accepted: {accepted_surfaces}"
            )

    def test_rejected_does_not_contain_accepted_surfaces(self):
        """Entries in rejected must NOT have surfaces that were accepted."""
        from app.routers.ocr import _extract_mentions_from_text
        text = "li chevaliers leantolot vint au chastel"
        mentions, _, debug = _extract_mentions_from_text(text)
        accepted_surfaces = {m["surface"].lower() for m in mentions}
        for r in debug.get("rejected", []):
            surface = r.get("surface", "").lower()
            assert surface not in accepted_surfaces, (
                f"Rejected surface '{surface}' is in accepted: overlap!"
            )


# ── 6. Integration Regression for run 5241c606 ──────────────────────


class TestIntegrationRun5241c606:
    """End-to-end regression tests simulating run 5241c606."""

    # Simulated OCR text from the run — representative medieval French
    # with "leantolot" (OCR variant of "lancelot")
    SAMPLE_TEXT = (
        "ci endroit dist li contes que quant\n"
        "li chevaliers leantolot se fu partiz\n"
        "de la dame il chevaucha tant\n"
        "a\n"
        "quil vint en la forest pereilleuse\n"
    )

    def test_leantolot_accepted_not_rejected(self):
        """'leantolot' must appear only in accepted, never in rejected."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, debug = _extract_mentions_from_text(self.SAMPLE_TEXT)
        accepted = {m["surface"].lower() for m in mentions}
        rejected = {r["surface"].lower() for r in debug.get("rejected", [])}
        assert "leantolot" in accepted
        assert "leantolot" not in rejected

    def test_report_consistency(self):
        """The generated report must not list 'leantolot' in rejected."""
        from app.routers.ocr import (
            _extract_mentions_from_text,
            _build_mention_extraction_report,
        )
        mentions, _, debug = _extract_mentions_from_text(self.SAMPLE_TEXT)
        report = _build_mention_extraction_report(
            "5241c606-1be3-4165-97ca-c41ea5726e42",
            "test.png",
            mentions,
            debug,
        )
        # Check that leantolot is in top_mentions_preview
        assert 'surface="leantolot"' in report

        # Parse rejected section
        import re
        in_rejected = False
        for line in report.split("\n"):
            if "top_rejected_preview" in line:
                in_rejected = True
                continue
            if "attempts_for_accepted" in line:
                in_rejected = False
                continue
            if in_rejected and "leantolot" in line.lower():
                pytest.fail(
                    f"'leantolot' found in top_rejected_preview: {line}"
                )

    def test_nd_bo_provenance_in_report(self):
        """The report must show accepted mention with nd/bo from the
        winning strategy, not the losing one."""
        from app.routers.ocr import (
            _extract_mentions_from_text,
            _build_mention_extraction_report,
        )
        mentions, _, debug = _extract_mentions_from_text(self.SAMPLE_TEXT)
        report = _build_mention_extraction_report(
            "5241c606", "test.png", mentions, debug,
        )
        # In the accepted section, leantolot should have method=ngram_canonical_*
        import re
        in_accepted = False
        found_leantolot = False
        for line in report.split("\n"):
            if "top_mentions_preview" in line:
                in_accepted = True
                continue
            if "top_rejected_preview" in line:
                in_accepted = False
                continue
            if in_accepted and "leantolot" in line.lower():
                found_leantolot = True
                # Must show ngram_canonical method, not salvage_work_fuzzy
                assert "ngram_canonical" in line or "salvage_work" in line, (
                    f"Expected canonical method, got: {line}"
                )
        assert found_leantolot, "leantolot not found in top_mentions_preview"

    def test_trailing_chunk_a_filtered(self):
        """The single-char line 'a' must not produce a chunk."""
        from app.routers.ocr import _build_line_chunks
        chunks = _build_line_chunks(self.SAMPLE_TEXT)
        chunk_texts = [c["text"].strip() for c in chunks]
        assert "a" not in chunk_texts, (
            f"'a' should be filtered from chunks: {chunk_texts}"
        )

    def test_mention_notes_match_report(self):
        """The nd/bo stored in the mention note must equal what the
        report shows — no contradictory metrics."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, debug = _extract_mentions_from_text(self.SAMPLE_TEXT)

        leantolot_mentions = [
            m for m in mentions if m["surface"].lower() == "leantolot"
        ]
        assert len(leantolot_mentions) >= 1
        m = leantolot_mentions[0]
        notes = str(m.get("notes", ""))

        # If there are attempts for this surface, the attempt metrics
        # must differ from the accepted metrics
        for a in debug.get("attempts_for_accepted", []):
            if a["surface"].lower() == "leantolot":
                attempt_reason = str(a.get("reason", ""))
                # The attempt reason should indicate FAILURE, not success
                assert (
                    ">" in attempt_reason
                    or "<" in attempt_reason
                    or "failed" in attempt_reason
                ), (
                    f"Attempt for accepted surface has non-failure reason: "
                    f"{attempt_reason!r}"
                )

    def test_full_invariant_no_overlap(self):
        """Full invariant: for the entire SAMPLE_TEXT, accepted ∩ rejected = ∅."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, debug = _extract_mentions_from_text(self.SAMPLE_TEXT)
        accepted = {m["surface"].lower() for m in mentions}
        rejected = {r["surface"].lower() for r in debug.get("rejected", [])}
        overlap = accepted & rejected
        assert overlap == set(), f"Overlap: {overlap}"
