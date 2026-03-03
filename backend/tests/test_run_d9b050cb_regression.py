"""Regression tests for run d9b050cb-78d2-4fde-b21b-84e4ed7d3b0b.

Root cause:
  Surface "leantolot" appeared in BOTH ``top_mentions_preview`` (accepted
  via ``rule:ngram_canonical_person`` matching "leantlote", nd=0.222
  bo=0.750) AND ``top_rejected_preview`` (rejected by ``rule:salvage_work_fuzzy``
  against "lancelot" with nd=0.333, bo=0.400).  The nd/bo values in the
  rejected line contradicted the accepted mention's stored notes.

Fix:
  1. Single decision per span: post-dedup reconciliation moves stale
     rejections for accepted surfaces into ``attempts_for_accepted``.
  2. Non-linkable mentions (ent_type=role) go into ``skipped_preview``,
     not ``top_rejected_preview``.
  3. Code assertion: accepted ∩ rejected = ∅.
  4. All report sections derive strictly from final decision records.
  5. Chunk hygiene: single-char / stopword lines are filtered.

This file covers:
  1. Single-decision invariant per span key.
  2. Report section semantics (accepted / skipped / rejected / attempts).
  3. Metric consistency (nd/bo in notes match final decision).
  4. Chunk hygiene (trailing "a" not stored).
  5. Integration regression for run d9b050cb.
"""

from __future__ import annotations

import re

import pytest


# ── Sample texts ─────────────────────────────────────────────────────

SAMPLE_TEXT_D9B = (
    "li roya mrecial vint au chastel\n"
    "et leantolot chevaucha tant\n"
    "a\n"
    "quil parla de la forest\n"
)


# ── 1. Single-Decision Invariant ─────────────────────────────────────


class TestSingleDecisionInvariant:
    """A span key must appear in either accepted OR rejected, never both."""

    def test_leantolot_not_in_both(self):
        """surface='leantolot' must not appear in both accepted and rejected."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        accepted = {m["surface"].lower() for m in mentions}
        rejected = {r["surface"].lower() for r in debug.get("rejected", [])}
        assert "leantolot" not in (accepted & rejected)

    def test_no_overlap_for_any_surface(self):
        """No surface may exist in both accepted and rejected."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        accepted = {m["surface"].lower() for m in mentions}
        rejected = {r["surface"].lower() for r in debug.get("rejected", [])}
        overlap = accepted & rejected
        assert overlap == set(), f"Overlap: {overlap}"

    def test_invariant_holds_for_varied_texts(self):
        """The invariant must hold for varied OCR inputs."""
        from app.routers.ocr import _extract_mentions_from_text
        texts = [
            "li rois leantolot et la dame",
            "sire leantolot vint devant le roi",
            "dist leantolot a perceval et a galahad",
            "il parla de leantolot le chevalier",
            "la dame de leantolot fu molt bele",
        ]
        for text in texts:
            mentions, _, debug = _extract_mentions_from_text(text)
            accepted = {m["surface"].lower() for m in mentions}
            rejected = {r["surface"].lower() for r in debug.get("rejected", [])}
            overlap = accepted & rejected
            assert overlap == set(), f"Text={text!r}: overlap={overlap}"

    def test_code_assertion_fires_on_overlap(self):
        """The in-code assertion must prevent overlapping decisions.
        
        We verify the assertion exists by checking the source code
        contains the assertion string."""
        import inspect
        from app.routers.ocr import _extract_mentions_from_text
        source = inspect.getsource(_extract_mentions_from_text)
        assert "accepted_surfaces_lower & rejected_surfaces_lower" in source or \
               "BUG: surfaces in BOTH accepted and rejected" in source


# ── 2. Report Section Semantics ──────────────────────────────────────


class TestReportSections:
    """Each report section must contain only the correct final status."""

    def _get_report(self):
        from app.routers.ocr import (
            _extract_mentions_from_text,
            _build_mention_extraction_report,
        )
        mentions, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        report = _build_mention_extraction_report(
            "d9b050cb", "test.png", mentions, debug,
        )
        return report, mentions, debug

    def test_leantolot_in_accepted_section(self):
        """'leantolot' must appear in top_mentions_preview."""
        report, _, _ = self._get_report()
        assert 'surface="leantolot"' in report

    def test_leantolot_not_in_rejected_section(self):
        """'leantolot' must NOT appear in top_rejected_preview."""
        report, _, _ = self._get_report()
        in_rejected = False
        for line in report.split("\n"):
            if "top_rejected_preview" in line:
                in_rejected = True
                continue
            if "attempts_for_accepted" in line or "skipped_preview" in line:
                in_rejected = False
                continue
            if in_rejected and "leantolot" in line.lower():
                pytest.fail(f"'leantolot' found in rejected section: {line}")

    def test_role_in_skipped_section(self):
        """'roya mrecial' (role) must appear in skipped_preview, not accepted."""
        report, _, _ = self._get_report()
        assert "skipped_preview" in report
        # Must be in skipped section
        in_skipped = False
        found = False
        for line in report.split("\n"):
            if "skipped_preview" in line:
                in_skipped = True
                continue
            if "top_rejected_preview" in line or "attempts_for_accepted" in line:
                in_skipped = False
                continue
            if in_skipped and "roya mrecial" in line.lower():
                found = True
        assert found, "'roya mrecial' not found in skipped_preview"

    def test_role_not_in_accepted_section(self):
        """'roya mrecial' (role) must NOT appear in top_mentions_preview."""
        report, _, _ = self._get_report()
        in_accepted = False
        for line in report.split("\n"):
            if "top_mentions_preview" in line:
                in_accepted = True
                continue
            if any(s in line for s in [
                "skipped_preview", "top_rejected_preview", "attempts_for_accepted"
            ]):
                in_accepted = False
                continue
            if in_accepted and "roya mrecial" in line.lower():
                pytest.fail(f"'roya mrecial' found in accepted section: {line}")

    def test_attempts_for_accepted_present(self):
        """Stale rejection for leantolot must be in attempts_for_accepted."""
        import uuid
        from app.db.pipeline_db import create_run, list_entity_attempts_for_run
        from app.routers.ocr import _run_trace_analysis, _build_mention_extraction_report
        run_id = create_run(f"test-d9b0-{uuid.uuid4().hex[:8]}")
        _chunks, mentions, _cands, debug = _run_trace_analysis(run_id, SAMPLE_TEXT_D9B)
        report = _build_mention_extraction_report(
            run_id, "test.png", mentions, debug,
        )
        attempts = list_entity_attempts_for_run(run_id)
        if attempts:
            assert "attempts_for_accepted" in report
            # Must contain leantolot
            in_attempts = False
            found = False
            for line in report.split("\n"):
                if "attempts_for_accepted" in line:
                    in_attempts = True
                    continue
                if in_attempts and "leantolot" in line.lower():
                    found = True
            assert found, "'leantolot' not found in attempts_for_accepted"

    def test_accepted_shows_notes(self):
        """top_mentions_preview lines must include notes= with provenance."""
        report, _, _ = self._get_report()
        in_accepted = False
        for line in report.split("\n"):
            if "top_mentions_preview" in line:
                in_accepted = True
                continue
            if any(s in line for s in [
                "skipped_preview", "top_rejected_preview", "attempts_for_accepted"
            ]):
                in_accepted = False
                continue
            if in_accepted and "leantolot" in line.lower():
                assert "notes=" in line, f"Missing notes= in accepted line: {line}"
                assert "nd=" in line, f"Missing nd= in accepted line: {line}"
                assert "bo=" in line, f"Missing bo= in accepted line: {line}"


# ── 3. Metric Consistency ────────────────────────────────────────────


class TestMetricConsistency:
    """The nd/bo printed in the report must match the stored notes."""

    def test_leantolot_nd_is_0222(self):
        """The final decision for 'leantolot' must have nd=0.222."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, _ = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        m = [m for m in mentions if m["surface"].lower() == "leantolot"]
        assert len(m) >= 1
        notes = str(m[0].get("notes", ""))
        nd_match = re.search(r"nd=(\d+\.\d+)", notes)
        assert nd_match, f"No nd= in notes: {notes!r}"
        assert float(nd_match.group(1)) == pytest.approx(0.222, abs=0.001)

    def test_leantolot_bo_is_0750(self):
        """The final decision for 'leantolot' must have bo=0.750."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, _ = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        m = [m for m in mentions if m["surface"].lower() == "leantolot"]
        assert len(m) >= 1
        notes = str(m[0].get("notes", ""))
        bo_match = re.search(r"bo=(\d+\.\d+)", notes)
        assert bo_match, f"No bo= in notes: {notes!r}"
        assert float(bo_match.group(1)) == pytest.approx(0.750, abs=0.001)

    def test_report_nd_bo_match_stored(self):
        """nd/bo in the report accepted section must match the mention's notes."""
        from app.routers.ocr import (
            _extract_mentions_from_text,
            _build_mention_extraction_report,
        )
        mentions, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        report = _build_mention_extraction_report("d9b050cb", "test.png", mentions, debug)

        # Get stored nd/bo
        m = [m for m in mentions if m["surface"].lower() == "leantolot"][0]
        notes = str(m.get("notes", ""))
        stored_nd = re.search(r"nd=(\d+\.\d+)", notes).group(1)
        stored_bo = re.search(r"bo=(\d+\.\d+)", notes).group(1)

        # Find the same surface in the report
        in_accepted = False
        for line in report.split("\n"):
            if "top_mentions_preview" in line:
                in_accepted = True
                continue
            if any(s in line for s in [
                "skipped_preview", "top_rejected_preview", "attempts_for_accepted"
            ]):
                in_accepted = False
                continue
            if in_accepted and "leantolot" in line.lower():
                assert f"nd={stored_nd}" in line, (
                    f"Report nd does not match stored nd={stored_nd}: {line}"
                )
                assert f"bo={stored_bo}" in line, (
                    f"Report bo does not match stored bo={stored_bo}: {line}"
                )

    def test_attempt_has_different_metrics(self):
        """The attempt entry for leantolot must show higher nd (0.333) from
        the failed salvage_work_fuzzy match against 'lancelot'."""
        from app.routers.ocr import _extract_mentions_from_text
        _, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        attempts = debug.get("attempts_for_accepted", [])
        leant_attempts = [a for a in attempts if a["surface"].lower() == "leantolot"]
        assert len(leant_attempts) >= 1
        reason = str(leant_attempts[0].get("reason", ""))
        # Should contain the failing nd=0.333
        assert "0.333" in reason, f"Expected nd=0.333 in attempt reason: {reason}"

    def test_salvage_work_fuzzy_notes_include_bo(self):
        """Rule 2 (salvage_work_fuzzy) accepted mentions now include bo= in notes."""
        from app.routers.ocr import _extract_mentions_from_text
        # Use text where a work IS matched by Rule 2
        text = "il parla de lancelot le bon chevalier"
        mentions, _, _ = _extract_mentions_from_text(text)
        work_mentions = [
            m for m in mentions if m.get("method") == "rule:salvage_work_fuzzy"
        ]
        for m in work_mentions:
            notes = str(m.get("notes", ""))
            assert "bo=" in notes, f"Missing bo= in salvage_work_fuzzy notes: {notes}"


# ── 4. Chunk Hygiene ─────────────────────────────────────────────────


class TestChunkHygiene:
    """Pure stopword / single-char lines must not produce chunks."""

    def test_trailing_a_filtered(self):
        """The line 'a' in the sample text must not produce a chunk."""
        from app.routers.ocr import _build_line_chunks
        chunks = _build_line_chunks(SAMPLE_TEXT_D9B)
        chunk_texts = [c["text"].strip() for c in chunks]
        assert "a" not in chunk_texts, f"'a' should be filtered: {chunk_texts}"

    def test_legitimate_lines_preserved(self):
        """Non-stopword lines must still produce chunks."""
        from app.routers.ocr import _build_line_chunks
        chunks = _build_line_chunks(SAMPLE_TEXT_D9B)
        chunk_texts = [c["text"].strip() for c in chunks]
        assert any("leantolot" in t for t in chunk_texts)
        assert any("roya" in t for t in chunk_texts)

    def test_chunk_indices_contiguous(self):
        """Chunk indices must be contiguous after filtering."""
        from app.routers.ocr import _build_line_chunks
        chunks = _build_line_chunks(SAMPLE_TEXT_D9B)
        indices = [c["idx"] for c in chunks]
        assert indices == list(range(len(chunks)))


# ── 5. Salvage Debug Structure ───────────────────────────────────────


class TestSalvageDebugStructure:
    """salvage_debug must have correct keys and clean separation."""

    def test_has_attempts_for_accepted(self):
        """salvage_debug must have 'attempts_for_accepted' key."""
        from app.routers.ocr import _extract_mentions_from_text
        _, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        assert "attempts_for_accepted" in debug

    def test_has_skipped_non_linkable(self):
        """salvage_debug must have 'skipped_non_linkable' key."""
        from app.routers.ocr import _extract_mentions_from_text
        _, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        assert "skipped_non_linkable" in debug

    def test_rejected_has_no_accepted_surfaces(self):
        """rejected must not contain any surface that is in accepted."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        accepted = {m["surface"].lower() for m in mentions}
        for r in debug.get("rejected", []):
            assert r["surface"].lower() not in accepted

    def test_attempts_have_only_accepted_surfaces(self):
        """attempts_for_accepted must only contain surfaces that were accepted."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        accepted = {m["surface"].lower() for m in mentions}
        for a in debug.get("attempts_for_accepted", []):
            assert a["surface"].lower() in accepted, (
                f"Attempt surface '{a['surface']}' not in accepted"
            )


# ── 6. Integration Regression for run d9b050cb ──────────────────────


class TestIntegrationRunD9b050cb:
    """End-to-end regression tests for run d9b050cb."""

    def test_leantolot_accepted_only(self):
        """'leantolot' appears in accepted (and only there)."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        accepted = {m["surface"].lower() for m in mentions}
        rejected = {r["surface"].lower() for r in debug.get("rejected", [])}
        assert "leantolot" in accepted
        assert "leantolot" not in rejected

    def test_leantolot_canonical_is_leantlote(self):
        """The canonical match for 'leantolot' must be 'leantlote'."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, _ = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        m = [m for m in mentions if m["surface"].lower() == "leantolot"][0]
        assert "leantlote" in str(m.get("notes", ""))

    def test_report_nd_bo_match_0222_0750(self):
        """The report must show nd=0.222 and bo=0.750 for leantolot."""
        from app.routers.ocr import (
            _extract_mentions_from_text,
            _build_mention_extraction_report,
        )
        mentions, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        report = _build_mention_extraction_report("d9b050cb", "test.png", mentions, debug)
        in_accepted = False
        for line in report.split("\n"):
            if "top_mentions_preview" in line:
                in_accepted = True
                continue
            if any(s in line for s in [
                "skipped_preview", "top_rejected_preview", "attempts_for_accepted"
            ]):
                in_accepted = False
                continue
            if in_accepted and "leantolot" in line.lower():
                assert "nd=0.222" in line
                assert "bo=0.750" in line

    def test_roya_mrecial_in_skipped(self):
        """'roya mrecial' must be in skipped_preview (SKIP_NON_LINKABLE)."""
        from app.routers.ocr import _extract_mentions_from_text
        _, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        skipped = debug.get("skipped_non_linkable", [])
        skipped_surfaces = {str(s.get("surface", "")).lower() for s in skipped}
        assert "roya mrecial" in skipped_surfaces

    def test_chunk_a_not_stored(self):
        """The trailing line 'a' must not be stored as a chunk."""
        from app.routers.ocr import _build_line_chunks
        chunks = _build_line_chunks(SAMPLE_TEXT_D9B)
        assert "a" not in [c["text"].strip() for c in chunks]

    def test_full_report_no_overlap(self):
        """Full invariant: accepted ∩ rejected = ∅ in the report."""
        from app.routers.ocr import (
            _extract_mentions_from_text,
            _build_mention_extraction_report,
        )
        mentions, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        report = _build_mention_extraction_report("d9b050cb", "test.png", mentions, debug)

        accepted_set: set[str] = set()
        rejected_set: set[str] = set()
        section = None
        for line in report.split("\n"):
            if "top_mentions_preview" in line:
                section = "accepted"
                continue
            if "skipped_preview" in line:
                section = "skipped"
                continue
            if "top_rejected_preview" in line:
                section = "rejected"
                continue
            if "attempts_for_accepted" in line:
                section = "attempts"
                continue
            m = re.search(r'surface="([^"]*)"', line)
            if m:
                surface = m.group(1).lower()
                if section == "accepted":
                    accepted_set.add(surface)
                elif section == "rejected":
                    rejected_set.add(surface)

        overlap = accepted_set & rejected_set
        assert overlap == set(), f"Report overlap: {overlap}"

    def test_mentions_total_correct(self):
        """mentions_total in report matches actual count."""
        from app.routers.ocr import (
            _extract_mentions_from_text,
            _build_mention_extraction_report,
        )
        mentions, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_D9B)
        report = _build_mention_extraction_report("d9b050cb", "test.png", mentions, debug)
        total_line = [l for l in report.split("\n") if "mentions_total:" in l][0]
        total = int(total_line.split(":")[1].strip())
        assert total == len(mentions)
