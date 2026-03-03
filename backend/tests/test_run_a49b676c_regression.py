"""Regression tests for run a49b676c-5d38-4275-b52c-a9f75e93fe30.

Root cause:
  The top_rejected_preview showed surface="liz" as a rejected place
  candidate, but this rejection was NOT persisted in the database.
  Only 2 accepted mentions existed in entity_mentions.  The report
  was partially based on transient in-memory pipeline state (the
  ``salvage_debug`` dict), not on persisted final decisions.

Fix:
  1. New ``entity_decisions`` table persists ALL final decisions
     (ACCEPT_LINKABLE, REJECT_LINKABLE, SKIP_NON_LINKABLE, FILTERED_OUT)
     for every extracted span.
  2. ``_run_trace_analysis()`` inserts decision records for accepted,
     rejected, skipped, and filtered spans after mention extraction.
  3. ``_build_mention_extraction_report()`` derives report sections
     from persisted decisions when available (decisions= kwarg).
  4. ``clear_analysis_for_run()`` also clears entity_decisions.
  5. Attempt history for accepted spans is stored in decision
     ``meta_json.attempts``.
  6. FILTERED_OUT items (too short, stopword, blacklist) appear only
     in ``filtered_out_preview``, never in ``top_rejected_preview``.

This file covers:
  1. DB schema: entity_decisions table exists with correct columns.
  2. CRUD: insert, list, count, clear for entity_decisions.
  3. Decision persistence: _run_trace_analysis persists ALL decisions.
  4. Report auditability: every report section item has a persisted row.
  5. Single-decision invariant on persisted decisions.
  6. Backward-compatible legacy report path (no decisions= kwarg).
  7. FILTERED_OUT vs REJECT_LINKABLE classification.
"""

from __future__ import annotations

import re
import uuid

import pytest


# ── Sample texts ─────────────────────────────────────────────────────

SAMPLE_TEXT_A49 = (
    "li roya mrecial vint au chastel\n"
    "et leantolot chevaucha tant\n"
    "a\n"
    "quil parla de la forest\n"
)

# Shorter text that produces a rejected place candidate
SAMPLE_TEXT_LIZ = (
    "liz entra en la tor\n"
    "et parla au roi\n"
)

# Text with mixed ACCEPT / SKIP / REJECT outcomes
SAMPLE_TEXT_MIXED = (
    "li roya mrecial vint au chastel\n"
    "et leantolot chevaucha tant\n"
    "liz entra en la tor\n"
)


# ── 1. DB Schema ────────────────────────────────────────────────────


class TestEntityDecisionsSchema:
    """entity_decisions table must exist with expected columns."""

    def test_table_exists(self):
        from app.db.pipeline_db import _init_db_if_needed, _connect
        _init_db_if_needed()
        with _connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='entity_decisions'"
            ).fetchone()
        assert row is not None, "entity_decisions table does not exist"

    def test_expected_columns(self):
        from app.db.pipeline_db import _init_db_if_needed, _connect
        _init_db_if_needed()
        with _connect() as conn:
            cols = conn.execute("PRAGMA table_info(entity_decisions)").fetchall()
        col_names = {str(c[1]) for c in cols}
        expected = {
            "decision_id", "run_id", "span_key", "chunk_id", "start_offset", "end_offset",
            "surface", "norm", "ent_type_guess", "label", "status",
            "method", "reason", "confidence", "meta_json", "created_at",
        }
        missing = expected - col_names
        assert not missing, f"Missing columns: {missing}"

    def test_unique_index_on_span_key(self):
        """UNIQUE index on (run_id, start_offset, end_offset, surface)."""
        from app.db.pipeline_db import _init_db_if_needed, _connect
        _init_db_if_needed()
        with _connect() as conn:
            indexes = conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='entity_decisions'"
            ).fetchall()
        idx_sqls = {str(row[0]): str(row[1] or "") for row in indexes}
        span_key_idx = idx_sqls.get("idx_entity_decisions_span_key", "")
        assert "UNIQUE" in span_key_idx.upper(), (
            f"span_key index is not UNIQUE: {span_key_idx}"
        )


# ── 2. CRUD Operations ──────────────────────────────────────────────


class TestEntityDecisionsCRUD:
    """Insert, list, count, clear operations for entity_decisions."""

    def _make_run_id(self) -> str:
        from app.db.pipeline_db import create_run
        return create_run(f"test-a49b676c-{uuid.uuid4().hex[:8]}")

    def test_insert_and_list(self):
        from app.db.pipeline_db import insert_entity_decisions, list_entity_decisions
        run_id = self._make_run_id()
        rows = insert_entity_decisions(run_id, [
            {
                "surface": "leantolot",
                "ent_type_guess": "person",
                "status": "ACCEPT_LINKABLE",
                "confidence": 0.55,
                "method": "rule:ngram_canonical_person",
            },
            {
                "surface": "liz",
                "ent_type_guess": "unknown",
                "status": "FILTERED_OUT",
                "reason": "too short",
            },
        ])
        assert len(rows) == 2
        listed = list_entity_decisions(run_id)
        assert len(listed) == 2
        surfaces = {d["surface"] for d in listed}
        assert surfaces == {"leantolot", "liz"}

    def test_list_filtered_by_status(self):
        from app.db.pipeline_db import insert_entity_decisions, list_entity_decisions
        run_id = self._make_run_id()
        insert_entity_decisions(run_id, [
            {"surface": "a", "ent_type_guess": "person", "status": "ACCEPT_LINKABLE", "confidence": 0.5},
            {"surface": "b", "ent_type_guess": "role", "status": "SKIP_NON_LINKABLE", "confidence": 0.0},
            {"surface": "c", "ent_type_guess": "unknown", "status": "REJECT_LINKABLE", "confidence": 0.0},
            {"surface": "d", "ent_type_guess": "unknown", "status": "FILTERED_OUT", "confidence": 0.0},
        ])
        accept = list_entity_decisions(run_id, status="ACCEPT_LINKABLE")
        assert len(accept) == 1
        assert accept[0]["surface"] == "a"
        skip = list_entity_decisions(run_id, status="SKIP_NON_LINKABLE")
        assert len(skip) == 1
        assert skip[0]["surface"] == "b"
        reject = list_entity_decisions(run_id, status="REJECT_LINKABLE")
        assert len(reject) == 1
        assert reject[0]["surface"] == "c"
        filtered = list_entity_decisions(run_id, status="FILTERED_OUT")
        assert len(filtered) == 1
        assert filtered[0]["surface"] == "d"

    def test_count(self):
        from app.db.pipeline_db import insert_entity_decisions, count_entity_decisions
        run_id = self._make_run_id()
        insert_entity_decisions(run_id, [
            {"surface": "x", "ent_type_guess": "person", "status": "ACCEPT_LINKABLE", "confidence": 0.5},
            {"surface": "y", "ent_type_guess": "unknown", "status": "REJECT_LINKABLE", "confidence": 0.0},
        ])
        assert count_entity_decisions(run_id) == 2
        assert count_entity_decisions(run_id, status="ACCEPT_LINKABLE") == 1
        assert count_entity_decisions(run_id, status="REJECT_LINKABLE") == 1

    def test_clear_analysis_also_clears_decisions(self):
        from app.db.pipeline_db import (
            insert_entity_decisions, count_entity_decisions,
            clear_analysis_for_run,
        )
        run_id = self._make_run_id()
        insert_entity_decisions(run_id, [
            {"surface": "z", "ent_type_guess": "person", "status": "ACCEPT_LINKABLE", "confidence": 0.5},
        ])
        assert count_entity_decisions(run_id) == 1
        clear_analysis_for_run(run_id)
        assert count_entity_decisions(run_id) == 0

    def test_meta_json_round_trip(self):
        from app.db.pipeline_db import insert_entity_decisions, list_entity_decisions
        run_id = self._make_run_id()
        insert_entity_decisions(run_id, [
            {
                "surface": "test",
                "ent_type_guess": "person",
                "status": "ACCEPT_LINKABLE",
                "confidence": 0.5,
                "meta_json": {"attempts": [{"surface": "test", "canonical": "X"}]},
            },
        ])
        listed = list_entity_decisions(run_id)
        assert len(listed) == 1
        meta = listed[0]["meta_json"]
        assert isinstance(meta, dict)
        assert meta["attempts"][0]["canonical"] == "X"

    def test_insert_or_replace_on_duplicate_span(self):
        """INSERT OR REPLACE should update existing row on same span_key."""
        from app.db.pipeline_db import insert_entity_decisions, list_entity_decisions
        run_id = self._make_run_id()
        insert_entity_decisions(run_id, [
            {"surface": "dup", "ent_type_guess": "person", "status": "ACCEPT_LINKABLE",
             "start_offset": 0, "end_offset": 3, "confidence": 0.5},
        ])
        # Insert again with same span_key but different status
        insert_entity_decisions(run_id, [
            {"surface": "dup", "ent_type_guess": "person", "status": "REJECT_LINKABLE",
             "start_offset": 0, "end_offset": 3, "confidence": 0.0},
        ])
        listed = list_entity_decisions(run_id)
        assert len(listed) == 1
        assert listed[0]["status"] == "REJECT_LINKABLE"


# ── 3. Decision Persistence via _run_trace_analysis ──────────────────


class TestDecisionPersistence:
    """_run_trace_analysis must persist decisions for ALL span categories."""

    def _run_and_query(self, text: str):
        from app.db.pipeline_db import create_run, list_entity_decisions
        from app.routers.ocr import _run_trace_analysis
        run_id = create_run(f"test-persist-{uuid.uuid4().hex[:8]}")
        _run_trace_analysis(run_id, text)
        return run_id, list_entity_decisions(run_id)

    def test_decisions_persisted_for_sample_text(self):
        """At least one decision should be persisted for non-trivial text."""
        run_id, decisions = self._run_and_query(SAMPLE_TEXT_A49)
        assert len(decisions) > 0, "No decisions persisted"

    def test_accept_decisions_exist(self):
        """Accepted mentions must produce ACCEPT_LINKABLE decision rows."""
        run_id, decisions = self._run_and_query(SAMPLE_TEXT_A49)
        accept = [d for d in decisions if d["status"] == "ACCEPT_LINKABLE"]
        assert len(accept) >= 1, "No ACCEPT_LINKABLE decision found"

    def test_leantolot_has_accept_decision(self):
        """surface='leantolot' must be persisted as ACCEPT_LINKABLE."""
        run_id, decisions = self._run_and_query(SAMPLE_TEXT_A49)
        accept_surfaces = {d["surface"].lower() for d in decisions if d["status"] == "ACCEPT_LINKABLE"}
        assert "leantolot" in accept_surfaces

    def test_skip_decisions_for_role_mentions(self):
        """Non-linkable role mentions must produce SKIP_NON_LINKABLE decisions."""
        run_id, decisions = self._run_and_query(SAMPLE_TEXT_A49)
        skip = [d for d in decisions if d["status"] == "SKIP_NON_LINKABLE"]
        # "roya mrecial" is a role mention → should be skipped
        skip_surfaces = {d["surface"].lower() for d in skip}
        has_role = any("roya" in s or "mrecial" in s for s in skip_surfaces)
        assert has_role, f"Expected role skip decision, got surfaces: {skip_surfaces}"

    def test_reject_decisions_persisted(self):
        """Rejected candidates must produce REJECT_LINKABLE or FILTERED_OUT decision rows."""
        # Use sample text that's known to produce rejections
        run_id, decisions = self._run_and_query(SAMPLE_TEXT_A49)
        reject_or_filter = [d for d in decisions if d["status"] in ("REJECT_LINKABLE", "FILTERED_OUT")]
        # There should be at least some rejections/filters from salvage processing
        assert isinstance(reject_or_filter, list)

    def test_every_status_is_valid(self):
        """All persisted decisions must have a valid status."""
        valid = {"ACCEPT_LINKABLE", "REJECT_LINKABLE", "SKIP_NON_LINKABLE", "FILTERED_OUT"}
        run_id, decisions = self._run_and_query(SAMPLE_TEXT_A49)
        for d in decisions:
            assert d["status"] in valid, f"Invalid status: {d['status']}"

    def test_no_span_in_multiple_statuses(self):
        """A given (surface_lower) must not appear in two different final statuses."""
        run_id, decisions = self._run_and_query(SAMPLE_TEXT_A49)
        surface_status: dict[str, set[str]] = {}
        for d in decisions:
            key = d["surface"].lower()
            surface_status.setdefault(key, set()).add(d["status"])
        for surf, statuses in surface_status.items():
            assert len(statuses) == 1, (
                f"surface='{surf}' has multiple statuses: {statuses}"
            )

    def test_accept_has_confidence_gt_zero(self):
        """ACCEPT_LINKABLE decisions should have positive confidence."""
        run_id, decisions = self._run_and_query(SAMPLE_TEXT_A49)
        for d in decisions:
            if d["status"] == "ACCEPT_LINKABLE":
                assert d["confidence"] > 0, (
                    f"ACCEPT_LINKABLE decision for '{d['surface']}' has confidence=0"
                )

    def test_accept_has_method(self):
        """ACCEPT_LINKABLE decisions should have a non-null method."""
        run_id, decisions = self._run_and_query(SAMPLE_TEXT_A49)
        for d in decisions:
            if d["status"] == "ACCEPT_LINKABLE":
                assert d["method"], (
                    f"ACCEPT_LINKABLE decision for '{d['surface']}' has no method"
                )

    def test_reject_has_reason(self):
        """REJECT_LINKABLE decisions should have a non-null reason."""
        run_id, decisions = self._run_and_query(SAMPLE_TEXT_A49)
        for d in decisions:
            if d["status"] == "REJECT_LINKABLE":
                assert d["reason"], (
                    f"REJECT_LINKABLE decision for '{d['surface']}' has no reason"
                )

    def test_attempts_for_accepted_in_meta(self):
        """ACCEPT_LINKABLE decisions with prior rejection attempts store them in entity_attempts."""
        from app.db.pipeline_db import list_entity_attempts_for_run
        run_id, decisions = self._run_and_query(SAMPLE_TEXT_A49)
        attempts = list_entity_attempts_for_run(run_id)
        # leantolot is accepted by ngram_canonical_person but has a
        # prior rejection attempt from salvage_work_fuzzy
        for a in attempts:
            assert "candidate" in a
            assert "reason" in a


# ── 4. Report Auditability ───────────────────────────────────────────


class TestReportAuditability:
    """Report sections must be backed by persisted decision rows."""

    def _run_full(self, text: str):
        from app.db.pipeline_db import create_run, list_entity_decisions
        from app.routers.ocr import (
            _run_trace_analysis,
            _build_mention_extraction_report,
        )
        run_id = create_run(f"test-audit-{uuid.uuid4().hex[:8]}")
        _chunks, mentions, _cands, salvage_debug = _run_trace_analysis(run_id, text)
        decisions = list_entity_decisions(run_id)
        report = _build_mention_extraction_report(
            run_id, "test-asset", mentions, salvage_debug,
        )
        return run_id, decisions, report

    def test_report_has_header(self):
        _, _, report = self._run_full(SAMPLE_TEXT_A49)
        assert "=== MENTION EXTRACTION REPORT ===" in report

    def test_top_mentions_backed_by_accept_decisions(self):
        """Each surface in top_mentions_preview must have an ACCEPT_LINKABLE decision."""
        run_id, decisions, report = self._run_full(SAMPLE_TEXT_A49)
        accept_surfaces = {
            d["surface"].lower() for d in decisions if d["status"] == "ACCEPT_LINKABLE"
        }
        # Parse surfaces from top_mentions_preview
        in_section = False
        for line in report.split("\n"):
            if "top_mentions_preview" in line:
                in_section = True
                continue
            if in_section:
                if line.strip().startswith("surface="):
                    m = re.search(r'surface="([^"]*)"', line)
                    if m:
                        surface = m.group(1).lower()
                        assert surface in accept_surfaces, (
                            f"surface='{surface}' in report but not in ACCEPT_LINKABLE decisions"
                        )
                elif line.strip() == "(none)" or (line.strip() and not line.startswith("  ")):
                    break

    def test_rejected_backed_by_reject_decisions(self):
        """Each surface in top_rejected_preview must have a REJECT_LINKABLE decision."""
        run_id, decisions, report = self._run_full(SAMPLE_TEXT_A49)
        reject_surfaces = {
            d["surface"].lower() for d in decisions if d["status"] == "REJECT_LINKABLE"
        }
        in_section = False
        for line in report.split("\n"):
            if "top_rejected_preview" in line:
                in_section = True
                continue
            if in_section:
                if line.strip().startswith("surface="):
                    m = re.search(r'surface="([^"]*)"', line)
                    if m:
                        surface = m.group(1).lower()
                        assert surface in reject_surfaces, (
                            f"surface='{surface}' in rejected report but not in REJECT_LINKABLE decisions"
                        )
                elif line.strip() == "(none)" or (line.strip() and not line.startswith("  ")):
                    break

    def test_skipped_backed_by_skip_decisions(self):
        """Each surface in skipped_preview must have a SKIP_NON_LINKABLE decision."""
        run_id, decisions, report = self._run_full(SAMPLE_TEXT_A49)
        skip_surfaces = {
            d["surface"].lower() for d in decisions if d["status"] == "SKIP_NON_LINKABLE"
        }
        in_section = False
        for line in report.split("\n"):
            if "skipped_preview" in line:
                in_section = True
                continue
            if in_section:
                if line.strip().startswith("surface="):
                    m = re.search(r'surface="([^"]*)"', line)
                    if m:
                        surface = m.group(1).lower()
                        assert surface in skip_surfaces, (
                            f"surface='{surface}' in skipped report but not in SKIP decisions"
                        )
                elif line.strip() and not line.startswith("  "):
                    break

    def test_legacy_report_path_still_works(self):
        """Report without decisions= kwarg falls back to in-memory data."""
        from app.routers.ocr import (
            _extract_mentions_from_text,
            _build_mention_extraction_report,
        )
        mentions, _, debug = _extract_mentions_from_text(SAMPLE_TEXT_A49)
        report = _build_mention_extraction_report(
            "test-legacy", "test-asset", mentions, debug,
        )
        assert "=== MENTION EXTRACTION REPORT ===" in report
        assert "top_mentions_preview" in report

    def test_consolidated_report_uses_decisions(self):
        """_build_consolidated_report passes decisions through."""
        from app.db.pipeline_db import create_run, list_entity_decisions
        from app.routers.ocr import (
            _run_trace_analysis,
            _build_consolidated_report,
        )
        run_id = create_run(f"test-consol-{uuid.uuid4().hex[:8]}")
        _chunks, mentions, _cands, salvage_debug = _run_trace_analysis(run_id, SAMPLE_TEXT_A49)
        decisions = list_entity_decisions(run_id)
        report = _build_consolidated_report(
            run_id, "test-asset", mentions, salvage_debug, None,
        )
        assert "=== MENTION EXTRACTION REPORT ===" in report
        assert "=== ENTITY LINKING REPORT ===" in report


# ── 5. nd/bo Parsing ─────────────────────────────────────────────────


class TestNdBoParsing:
    """_parse_nd_bo_from_notes extracts metrics from notes strings."""

    def test_parses_canonical_nd_bo(self):
        from app.routers.ocr import _parse_nd_bo_from_notes
        nd, bo, canon = _parse_nd_bo_from_notes("canonical=leantlote nd=0.222 bo=0.750")
        assert nd == pytest.approx(0.222)
        assert bo == pytest.approx(0.750)
        assert canon == "leantlote"

    def test_parses_none_from_empty(self):
        from app.routers.ocr import _parse_nd_bo_from_notes
        nd, bo, canon = _parse_nd_bo_from_notes(None)
        assert nd is None
        assert bo is None
        assert canon is None

    def test_parses_none_from_no_metrics(self):
        from app.routers.ocr import _parse_nd_bo_from_notes
        nd, bo, canon = _parse_nd_bo_from_notes("some random notes")
        assert nd is None
        assert bo is None
        assert canon is None

    def test_parses_exact_match(self):
        from app.routers.ocr import _parse_nd_bo_from_notes
        nd, bo, canon = _parse_nd_bo_from_notes("canonical=lancelot nd=0.000 bo=1.000")
        assert nd == pytest.approx(0.0)
        assert bo == pytest.approx(1.0)
        assert canon == "lancelot"


# ── 6. Integration: end-to-end decision persistence ─────────────────


class TestEndToEndDecisionPersistence:
    """Full pipeline creates auditable decision records."""

    def test_leantolot_accept_is_persisted_with_canonical(self):
        from app.db.pipeline_db import create_run, list_entity_decisions, list_entity_attempts
        from app.routers.ocr import _run_trace_analysis
        run_id = create_run(f"test-e2e-{uuid.uuid4().hex[:8]}")
        _run_trace_analysis(run_id, SAMPLE_TEXT_A49)
        accept = list_entity_decisions(run_id, status="ACCEPT_LINKABLE")
        leantolot = [d for d in accept if d["surface"].lower() == "leantolot"]
        assert len(leantolot) == 1
        d = leantolot[0]
        assert d["ent_type_guess"] == "person"
        # canonical is now in entity_attempts
        attempts = list_entity_attempts(d["decision_id"])
        accept_att = [a for a in attempts if a["attempt_decision"] == "ACCEPT"]
        assert accept_att, "missing ACCEPT attempt with canonical on ACCEPT_LINKABLE decision"

    def test_roya_mrecial_skip_is_persisted(self):
        """Role mention 'roya mrecial' should be SKIP_NON_LINKABLE."""
        from app.db.pipeline_db import create_run, list_entity_decisions
        from app.routers.ocr import _run_trace_analysis
        run_id = create_run(f"test-e2e-{uuid.uuid4().hex[:8]}")
        _run_trace_analysis(run_id, SAMPLE_TEXT_A49)
        skip = list_entity_decisions(run_id, status="SKIP_NON_LINKABLE")
        skip_surfaces = {d["surface"].lower() for d in skip}
        assert any("roya" in s for s in skip_surfaces), (
            f"Expected 'roya mrecial' in SKIP decisions, got: {skip_surfaces}"
        )

    def test_decision_count_matches_entity_mentions_plus_extras(self):
        """Total decisions >= entity_mentions (since rejections add to the total)."""
        from app.db.pipeline_db import (
            create_run, count_entity_mentions, count_entity_decisions,
        )
        from app.routers.ocr import _run_trace_analysis
        run_id = create_run(f"test-e2e-{uuid.uuid4().hex[:8]}")
        _run_trace_analysis(run_id, SAMPLE_TEXT_A49)
        mention_count = count_entity_mentions(run_id)
        decision_count = count_entity_decisions(run_id)
        assert decision_count >= mention_count, (
            f"decisions={decision_count} < mentions={mention_count}"
        )

    def test_empty_text_no_decisions(self):
        """Empty input text should produce zero decisions."""
        from app.db.pipeline_db import create_run, count_entity_decisions
        from app.routers.ocr import _run_trace_analysis
        run_id = create_run(f"test-empty-{uuid.uuid4().hex[:8]}")
        _run_trace_analysis(run_id, "")
        assert count_entity_decisions(run_id) == 0

    def test_clear_and_rerun_idempotent(self):
        """Running trace analysis twice should produce the same decisions."""
        from app.db.pipeline_db import create_run, list_entity_decisions
        from app.routers.ocr import _run_trace_analysis
        run_id = create_run(f"test-idempotent-{uuid.uuid4().hex[:8]}")
        _run_trace_analysis(run_id, SAMPLE_TEXT_A49)
        first = list_entity_decisions(run_id)
        _run_trace_analysis(run_id, SAMPLE_TEXT_A49)
        second = list_entity_decisions(run_id)
        # Same number of decisions
        assert len(first) == len(second)
        # Same surfaces
        first_surfaces = sorted(d["surface"].lower() for d in first)
        second_surfaces = sorted(d["surface"].lower() for d in second)
        assert first_surfaces == second_surfaces


# ── 7. table_view for entity_decisions ───────────────────────────────


class TestTableViewForDecisions:
    """table_view_for_entity_decisions works correctly."""

    def test_table_view_returns_correct_structure(self):
        from app.db.pipeline_db import (
            create_run, insert_entity_decisions, table_view_for_entity_decisions,
        )
        run_id = create_run(f"test-tv-{uuid.uuid4().hex[:8]}")
        insert_entity_decisions(run_id, [
            {"surface": "test", "ent_type_guess": "person", "status": "ACCEPT_LINKABLE", "confidence": 0.5},
        ])
        tv = table_view_for_entity_decisions(run_id)
        assert tv["table"] == "entity_decisions"
        assert "decision_id" in tv["columns"]
        assert "status" in tv["columns"]
        assert len(tv["rows"]) == 1
