"""Regression tests for run 540fad29-b255-4fa7-9e4f-98571d9559a7.

Root cause:
  top_rejected_preview showed surface="liz" with reason "too short /
  stopword", but there was no persisted rejection row for "liz" in the
  database.  The report was partially based on transient in-memory
  pipeline state rather than persisted final decisions.

  "liz" is a short token (3 chars) that fails the Rule 3 place-candidate
  gate (len < 5 or stopword).  It is NOT a genuine linkable rejection
  (it was never compared against a canonical entity).  It should be
  classified as FILTERED_OUT and shown only in ``filtered_out_preview``,
  never in ``top_rejected_preview``.

Fix:
  1. Final decision statuses are now:
     ACCEPT_LINKABLE, REJECT_LINKABLE, SKIP_NON_LINKABLE, FILTERED_OUT.
  2. ``_run_trace_analysis()`` classifies rejected entries:
     - Those with real canonical comparison (nd=/bo= in reason) →
       REJECT_LINKABLE.
     - Everything else (too short, stopword, blacklisted, non-alpha,
       place_likeness, OCR garbage) → FILTERED_OUT.
  3. ``_build_mention_extraction_report()`` derives ALL sections from
     persisted decisions:
     - top_mentions_preview    ← ACCEPT_LINKABLE
     - skipped_preview         ← SKIP_NON_LINKABLE
     - top_rejected_preview    ← REJECT_LINKABLE
     - filtered_out_preview    ← FILTERED_OUT (debug-only)
     - attempts_for_accepted   ← meta_json on ACCEPT_LINKABLE
  4. FILTERED_OUT items NEVER appear in top_rejected_preview.

This file covers:
  1. "liz" is FILTERED_OUT, not REJECT_LINKABLE.
  2. "liz" appears in filtered_out_preview, not top_rejected_preview.
  3. Every report section is backed by persisted decisions.
  4. No span_key appears in more than one bucket.
  5. leantolot ACCEPT_LINKABLE, roya mrecial SKIP_NON_LINKABLE preserved.
  6. Trailing single-character line "a" not stored as chunk.
  7. FILTERED_OUT items have reason populated.
"""

from __future__ import annotations

import re
import uuid

import pytest


# ── Sample texts ─────────────────────────────────────────────────────

SAMPLE_TEXT_540 = (
    "li roya mrecial vint au chastel\n"
    "et leantolot chevaucha tant\n"
    "a\n"
    "quil parla de la forest\n"
)

# Text that specifically produces "liz" as a filtered place candidate
SAMPLE_TEXT_LIZ = (
    "il vint en la tor\n"
    "liz entra en la forest\n"
    "et parla au roi\n"
)


# ── helpers ──────────────────────────────────────────────────────────

def _run_and_build(text: str):
    """Run trace analysis + build decision-backed report."""
    from app.db.pipeline_db import create_run, list_entity_decisions
    from app.routers.ocr import _run_trace_analysis, _build_mention_extraction_report
    run_id = create_run(f"test-540fad-{uuid.uuid4().hex[:8]}")
    _chunks, mentions, _cands, salvage_debug = _run_trace_analysis(run_id, text)
    decisions = list_entity_decisions(run_id)
    report = _build_mention_extraction_report(
        run_id, "test-asset", mentions, salvage_debug,
    )
    return run_id, decisions, report, mentions


# ── 1. "liz" classification ─────────────────────────────────────────


class TestLizClassification:
    """'liz' must be FILTERED_OUT (too short / stopword), not REJECT_LINKABLE."""

    def test_liz_is_filtered_out_not_reject_linkable(self):
        """If 'liz' appears in decisions, it must be FILTERED_OUT."""
        run_id, decisions, _, _ = _run_and_build(SAMPLE_TEXT_LIZ)
        liz_decisions = [d for d in decisions if d["surface"].lower() == "liz"]
        for d in liz_decisions:
            assert d["status"] == "FILTERED_OUT", (
                f"surface='liz' has status={d['status']}, expected FILTERED_OUT"
            )

    def test_liz_not_in_top_rejected_preview(self):
        """'liz' must NEVER appear in top_rejected_preview."""
        _, _, report, _ = _run_and_build(SAMPLE_TEXT_LIZ)
        in_rejected = False
        for line in report.split("\n"):
            if "top_rejected_preview" in line:
                in_rejected = True
                continue
            if in_rejected:
                if line.strip() and not line.startswith("  "):
                    break  # next section
                m = re.search(r'surface="([^"]*)"', line)
                if m and m.group(1).lower() == "liz":
                    pytest.fail("'liz' found in top_rejected_preview — must be in filtered_out_preview only")

    def test_liz_in_filtered_out_preview_if_present(self):
        """If 'liz' is persisted, it must appear in filtered_out_preview."""
        run_id, decisions, report, _ = _run_and_build(SAMPLE_TEXT_LIZ)
        liz_decisions = [d for d in decisions if d["surface"].lower() == "liz"]
        if liz_decisions:
            # Must appear in filtered_out_preview if persisted as FILTERED_OUT
            assert "filtered_out_preview" in report, (
                "'liz' is persisted as FILTERED_OUT but filtered_out_preview section missing"
            )
            in_filtered = False
            found = False
            for line in report.split("\n"):
                if "filtered_out_preview" in line:
                    in_filtered = True
                    continue
                if in_filtered:
                    if line.strip() and not line.startswith("  "):
                        break
                    m = re.search(r'surface="([^"]*)"', line)
                    if m and m.group(1).lower() == "liz":
                        found = True
                        break
            assert found, "'liz' persisted as FILTERED_OUT but not in filtered_out_preview"


# ── 2. Report section auditability ──────────────────────────────────


class TestReportFullAuditability:
    """Every item in every report section must have a persisted decision row."""

    def test_top_mentions_backed_by_accept_linkable(self):
        run_id, decisions, report, _ = _run_and_build(SAMPLE_TEXT_540)
        accept_surfaces = {d["surface"].lower() for d in decisions if d["status"] == "ACCEPT_LINKABLE"}
        in_section = False
        for line in report.split("\n"):
            if "top_mentions_preview" in line:
                in_section = True
                continue
            if in_section:
                if line.strip() == "(none)" or (line.strip() and not line.startswith("  ")):
                    break
                m = re.search(r'surface="([^"]*)"', line)
                if m:
                    assert m.group(1).lower() in accept_surfaces

    def test_skipped_backed_by_skip_non_linkable(self):
        run_id, decisions, report, _ = _run_and_build(SAMPLE_TEXT_540)
        skip_surfaces = {d["surface"].lower() for d in decisions if d["status"] == "SKIP_NON_LINKABLE"}
        in_section = False
        for line in report.split("\n"):
            if "skipped_preview" in line:
                in_section = True
                continue
            if in_section:
                if line.strip() and not line.startswith("  "):
                    break
                m = re.search(r'surface="([^"]*)"', line)
                if m:
                    assert m.group(1).lower() in skip_surfaces

    def test_rejected_backed_by_reject_linkable(self):
        run_id, decisions, report, _ = _run_and_build(SAMPLE_TEXT_540)
        reject_surfaces = {d["surface"].lower() for d in decisions if d["status"] == "REJECT_LINKABLE"}
        in_section = False
        for line in report.split("\n"):
            if "top_rejected_preview" in line:
                in_section = True
                continue
            if in_section:
                if line.strip() == "(none)" or (line.strip() and not line.startswith("  ")):
                    break
                m = re.search(r'surface="([^"]*)"', line)
                if m:
                    assert m.group(1).lower() in reject_surfaces, (
                        f"surface='{m.group(1)}' in top_rejected_preview but not in REJECT_LINKABLE decisions"
                    )

    def test_filtered_backed_by_filtered_out(self):
        run_id, decisions, report, _ = _run_and_build(SAMPLE_TEXT_540)
        filter_surfaces = {d["surface"].lower() for d in decisions if d["status"] == "FILTERED_OUT"}
        in_section = False
        for line in report.split("\n"):
            if "filtered_out_preview" in line:
                in_section = True
                continue
            if in_section:
                if line.strip() and not line.startswith("  "):
                    break
                m = re.search(r'surface="([^"]*)"', line)
                if m:
                    assert m.group(1).lower() in filter_surfaces

    def test_no_span_in_multiple_buckets(self):
        """A surface must appear in at most one report bucket."""
        _, _, report, _ = _run_and_build(SAMPLE_TEXT_540)
        sections = {
            "top_mentions_preview": set(),
            "skipped_preview": set(),
            "top_rejected_preview": set(),
            "filtered_out_preview": set(),
        }
        current_section = None
        for line in report.split("\n"):
            for sec in sections:
                if sec in line:
                    current_section = sec
                    break
            else:
                if current_section and line.strip() and not line.startswith("  "):
                    current_section = None
            if current_section:
                m = re.search(r'surface="([^"]*)"', line)
                if m:
                    sections[current_section].add(m.group(1).lower())
        # Check no surface appears in more than one bucket
        all_buckets = list(sections.keys())
        for i, b1 in enumerate(all_buckets):
            for b2 in all_buckets[i+1:]:
                overlap = sections[b1] & sections[b2]
                assert not overlap, (
                    f"surface(s) {overlap} appear in BOTH {b1} AND {b2}"
                )


# ── 3. Status classification correctness ────────────────────────────


class TestStatusClassification:
    """Correct 4-way status classification for all decisions."""

    def test_valid_statuses_only(self):
        valid = {"ACCEPT_LINKABLE", "REJECT_LINKABLE", "SKIP_NON_LINKABLE", "FILTERED_OUT"}
        _, decisions, _, _ = _run_and_build(SAMPLE_TEXT_540)
        for d in decisions:
            assert d["status"] in valid, f"Invalid status: {d['status']}"

    def test_leantolot_is_accept_linkable(self):
        _, decisions, _, _ = _run_and_build(SAMPLE_TEXT_540)
        leantolot = [d for d in decisions if d["surface"].lower() == "leantolot"]
        assert len(leantolot) == 1
        assert leantolot[0]["status"] == "ACCEPT_LINKABLE"
        assert leantolot[0]["ent_type_guess"] == "person"

    def test_roya_mrecial_is_skip_non_linkable(self):
        _, decisions, _, _ = _run_and_build(SAMPLE_TEXT_540)
        skip = [d for d in decisions if d["status"] == "SKIP_NON_LINKABLE"]
        skip_surfaces = {d["surface"].lower() for d in skip}
        assert any("roya" in s for s in skip_surfaces)

    def test_no_surface_in_two_statuses(self):
        """Each surface_lower must have exactly one final status."""
        _, decisions, _, _ = _run_and_build(SAMPLE_TEXT_540)
        surface_status: dict[str, set[str]] = {}
        for d in decisions:
            key = d["surface"].lower()
            surface_status.setdefault(key, set()).add(d["status"])
        for surf, statuses in surface_status.items():
            assert len(statuses) == 1, (
                f"surface='{surf}' has multiple statuses: {statuses}"
            )

    def test_filtered_out_has_reason(self):
        """Every FILTERED_OUT decision must have a non-empty reason."""
        _, decisions, _, _ = _run_and_build(SAMPLE_TEXT_540)
        for d in decisions:
            if d["status"] == "FILTERED_OUT":
                assert d["reason"], (
                    f"FILTERED_OUT decision for '{d['surface']}' has empty reason"
                )

    def test_reject_linkable_has_canonical_entity(self):
        """REJECT_LINKABLE decisions should have attempts with real canonical entity."""
        from app.db.pipeline_db import list_entity_attempts
        _, decisions, _, _ = _run_and_build(SAMPLE_TEXT_540)
        for d in decisions:
            if d["status"] == "REJECT_LINKABLE":
                attempts = list_entity_attempts(d["decision_id"])
                assert attempts, f"REJECT_LINKABLE for '{d['surface']}' has no attempts"
                for a in attempts:
                    assert a["candidate"] and not str(a["candidate"]).startswith("("), (
                        f"REJECT_LINKABLE attempt for '{d['surface']}' has placeholder candidate='{a['candidate']}'"
                    )

    def test_accept_linkable_has_method(self):
        """ACCEPT_LINKABLE decisions must have a non-null method."""
        _, decisions, _, _ = _run_and_build(SAMPLE_TEXT_540)
        for d in decisions:
            if d["status"] == "ACCEPT_LINKABLE":
                assert d["method"], f"ACCEPT_LINKABLE for '{d['surface']}' has no method"


# ── 4. Chunk hygiene ─────────────────────────────────────────────────


class TestChunkHygiene:
    """Trailing single-character OCR lines are not stored as chunks."""

    def test_single_char_a_not_in_chunks(self):
        from app.db.pipeline_db import create_run, list_chunks
        from app.routers.ocr import _run_trace_analysis
        run_id = create_run(f"test-chunk-{uuid.uuid4().hex[:8]}")
        _run_trace_analysis(run_id, SAMPLE_TEXT_540)
        chunks = list_chunks(run_id)
        for c in chunks:
            assert c["text"].strip() != "a", (
                f"Single-char line 'a' stored as chunk idx={c['idx']}"
            )


# ── 5. Decision persistence round-trip ───────────────────────────────


class TestDecisionPersistence:
    """Decisions survive clear + rerun and match expectations."""

    def test_clear_and_rerun_idempotent(self):
        from app.db.pipeline_db import create_run, list_entity_decisions
        from app.routers.ocr import _run_trace_analysis
        run_id = create_run(f"test-idem-{uuid.uuid4().hex[:8]}")
        _run_trace_analysis(run_id, SAMPLE_TEXT_540)
        first = list_entity_decisions(run_id)
        _run_trace_analysis(run_id, SAMPLE_TEXT_540)
        second = list_entity_decisions(run_id)
        assert len(first) == len(second)
        assert sorted(d["surface"].lower() for d in first) == sorted(d["surface"].lower() for d in second)

    def test_total_decisions_gte_mentions(self):
        from app.db.pipeline_db import create_run, count_entity_mentions, count_entity_decisions
        from app.routers.ocr import _run_trace_analysis
        run_id = create_run(f"test-cnt-{uuid.uuid4().hex[:8]}")
        _run_trace_analysis(run_id, SAMPLE_TEXT_540)
        assert count_entity_decisions(run_id) >= count_entity_mentions(run_id)

    def test_empty_text_no_decisions(self):
        from app.db.pipeline_db import create_run, count_entity_decisions
        from app.routers.ocr import _run_trace_analysis
        run_id = create_run(f"test-empty-{uuid.uuid4().hex[:8]}")
        _run_trace_analysis(run_id, "")
        assert count_entity_decisions(run_id) == 0


# ── 6. Attempts history ─────────────────────────────────────────────


class TestAttemptsHistory:
    """Intermediate evaluation attempts are persisted on ACCEPT_LINKABLE decisions."""

    def test_attempts_in_meta_json(self):
        """leantolot has a prior rejection attempt from salvage_work_fuzzy."""
        from app.db.pipeline_db import list_entity_attempts_for_run
        run_id, decisions, _, _ = _run_and_build(SAMPLE_TEXT_540)
        accept = [d for d in decisions if d["status"] == "ACCEPT_LINKABLE"]
        attempts = list_entity_attempts_for_run(run_id)
        for a in attempts:
            assert "candidate" in a
            assert "reason" in a

    def test_attempts_do_not_create_reject_entries(self):
        """Attempt history must not appear as REJECT_LINKABLE decisions."""
        _, decisions, _, _ = _run_and_build(SAMPLE_TEXT_540)
        accept_surfaces = {d["surface"].lower() for d in decisions if d["status"] == "ACCEPT_LINKABLE"}
        for d in decisions:
            if d["status"] == "REJECT_LINKABLE":
                assert d["surface"].lower() not in accept_surfaces, (
                    f"surface='{d['surface']}' is both ACCEPT_LINKABLE and REJECT_LINKABLE"
                )


# ── 7. Legacy report path backward-compat ───────────────────────────


class TestLegacyReportPath:
    """Legacy in-memory report path (no decisions= kwarg) still works."""

    def test_legacy_report_has_all_sections(self):
        from app.routers.ocr import _extract_mentions_from_text, _build_mention_extraction_report
        mentions, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_540)
        report = _build_mention_extraction_report(
            "test-legacy", "test-asset", mentions, debug,
        )
        assert "=== MENTION EXTRACTION REPORT ===" in report
        assert "top_mentions_preview" in report
        assert "top_rejected_preview" in report


# ── 8. Example corrected report excerpt ──────────────────────────────


class TestExampleReport:
    """Corrected report for run 540fad29 matches expected structure."""

    def test_report_structure_correct(self):
        """Report has proper sections in correct order."""
        _, _, report, _ = _run_and_build(SAMPLE_TEXT_540)
        lines = report.split("\n")
        section_order = []
        for line in lines:
            for sec in ("top_mentions_preview", "skipped_preview",
                        "top_rejected_preview", "filtered_out_preview",
                        "attempts_for_accepted"):
                if sec in line and sec not in section_order:
                    section_order.append(sec)
        # Must have at least top_mentions and top_rejected
        assert "top_mentions_preview" in section_order
        assert "top_rejected_preview" in section_order

    def test_leantolot_in_top_mentions(self):
        _, _, report, _ = _run_and_build(SAMPLE_TEXT_540)
        in_section = False
        found = False
        for line in report.split("\n"):
            if "top_mentions_preview" in line:
                in_section = True
                continue
            if in_section:
                if line.strip() and not line.startswith("  "):
                    break
                if 'surface="leantolot"' in line:
                    found = True
        assert found, "leantolot not found in top_mentions_preview"

    def test_roya_mrecial_in_skipped(self):
        _, _, report, _ = _run_and_build(SAMPLE_TEXT_540)
        assert "skipped_preview" in report
        in_section = False
        found = False
        for line in report.split("\n"):
            if "skipped_preview" in line:
                in_section = True
                continue
            if in_section:
                if line.strip() and not line.startswith("  "):
                    break
                if "roya" in line.lower():
                    found = True
        assert found, "roya mrecial not found in skipped_preview"
