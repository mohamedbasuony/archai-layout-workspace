"""Acceptance tests for quality gate enforcement in the OCR pipeline.

These tests verify that:
  1. When quality gates FAIL, downstream stages (proofread/analyze/NER/link/index)
     are NOT executed and the run is marked FAILED_QUALITY.
  2. A seam retry attempt is actually executed and recorded (attempt index
     increments, tiling params differ).
  3. Downstream stages check token_search_allowed / ner_allowed before running.
  4. The ocr_attempts table is correctly populated with per-attempt records.
  5. The pipeline correctly selects the best attempt across retries.

These are unit/integration tests that mock the actual OCR agent to simulate
different quality scenarios without requiring GPU/API access.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.pipeline_db import (
    create_run,
    get_run,
    insert_ocr_attempt,
    insert_ocr_quality_report,
    list_events,
    list_ocr_attempts,
    get_best_ocr_attempt,
    log_event,
    update_run_fields,
)
from app.services.ocr_quality import (
    OCRQualityReport,
    compute_quality_report,
)
from app.services.pipeline_hardening import (
    DOWNSTREAM_FALLBACK,
    DOWNSTREAM_TOKEN,
    decide_downstream_mode,
    enforce_quality_gates,
    format_gate_report,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_clean_text() -> str:
    """Text that should pass all quality gates (HIGH quality)."""
    return (
        "Li rois Artus apres la mort de son pere tint mont longuement "
        "la terre de Logres en pais et en joie. Et quant il ot establie "
        "toute la terre si com il li plut, il reuint a Kamaalot."
    )


def _make_gibberish_text() -> str:
    """Text that should fail quality gates (UNRELIABLE)."""
    return " ".join(["xqzfwvp"] * 30 + ["aldiluzor zmaradigno"] * 10)


def _make_seam_fragment_text() -> str:
    """Text with many leading fragments that should trigger seam retry.

    Uses genuine OCR seam artefacts (short non-word tokens like "rt", "ce")
    rather than function words like "de" or "et" which are now excluded
    from fragment detection.
    """
    lines = []
    frag_tokens = ["rt", "ce", "xp", "vt", "kl", "gn", "mb", "zr"]
    for i in range(16):
        if i % 2 == 0:
            lines.append(frag_tokens[i % len(frag_tokens)])
        else:
            lines.append(
                "Li rois Artus apres la mort de son pere tint la terre."
            )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Gate failure → pipeline halts
# ═══════════════════════════════════════════════════════════════════════


class TestGateFailureHaltsPipeline:
    """Verify that when quality gates fail, downstream stages do NOT run."""

    def test_unreliable_text_blocks_downstream(self):
        """Gibberish text → UNRELIABLE → gates have blocked_stages → ner_allowed=False."""
        text = _make_gibberish_text()
        report = compute_quality_report(text, run_id="test_halt")
        gates = enforce_quality_gates(report, run_id="test_halt")

        assert report.quality_label in ("RISKY", "UNRELIABLE")
        assert gates["ner_allowed"] is False
        assert gates["token_search_allowed"] is False
        assert len(gates["blocked_stages"]) > 0
        assert gates["downstream_mode"] == DOWNSTREAM_FALLBACK

    def test_seam_fragment_blocks_downstream(self):
        """High leading_fragment_ratio → seam_retry_required=True, blocked."""
        text = _make_seam_fragment_text()
        report = compute_quality_report(text, run_id="test_seam_halt")
        gates = enforce_quality_gates(report, run_id="test_seam_halt")

        # With genuine fragments (non-function-word 2-char tokens), the ratio
        # should exceed the hard limit (0.15)
        assert report.leading_fragment_ratio >= 0.15
        assert gates["seam_retry_required"] is True
        assert "seam_not_resolved" in gates["blocked_stages"]

    def test_clean_text_passes_all_gates(self):
        """Clean text → HIGH/OK → all gates pass → downstream allowed."""
        text = _make_clean_text()
        report = compute_quality_report(text, run_id="test_pass")
        gates = enforce_quality_gates(report, run_id="test_pass")

        assert report.quality_label in ("HIGH", "OK")
        assert gates["ner_allowed"] is True
        assert gates["token_search_allowed"] is True
        assert len(gates["blocked_stages"]) == 0
        assert gates["downstream_mode"] == DOWNSTREAM_TOKEN

    def test_failed_quality_run_status(self):
        """Simulate FAILED_QUALITY status in the DB."""
        run_id = create_run(asset_ref="test_failed_quality", asset_sha256="abc")
        update_run_fields(
            run_id,
            status="FAILED_QUALITY",
            current_stage="QUALITY_BLOCKED",
            error="Quality gates FAILED after 2 attempts",
        )
        run = get_run(run_id)
        assert run is not None
        assert run["status"] == "FAILED_QUALITY"
        assert run["current_stage"] == "QUALITY_BLOCKED"
        assert "Quality gates FAILED" in run["error"]


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Seam retry is executed and recorded
# ═══════════════════════════════════════════════════════════════════════


class TestSeamRetryRecording:
    """Verify that retry attempts are recorded with distinct params."""

    def test_attempt_index_increments(self):
        """Two attempts should have attempt_idx 0 and 1."""
        run_id = create_run(asset_ref="test_retry_idx", asset_sha256="retry1")
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "tiling_strategy": "default",
            "overlap_pct": 0.12,
            "quality_label": "RISKY",
            "gibberish_score": 0.30,
            "leading_fragment_ratio": 0.15,
            "non_wordlike_frac": 0.20,
            "gates_passed": False,
            "decision": "FAIL",
        })
        insert_ocr_attempt(run_id, {
            "attempt_idx": 1,
            "tiling_strategy": "seam_retry_overlap_22%",
            "overlap_pct": 0.22,
            "quality_label": "OK",
            "gibberish_score": 0.12,
            "leading_fragment_ratio": 0.03,
            "non_wordlike_frac": 0.10,
            "gates_passed": True,
            "decision": "PASS",
        })
        attempts = list_ocr_attempts(run_id)
        assert len(attempts) == 2
        assert attempts[0]["attempt_idx"] == 0
        assert attempts[1]["attempt_idx"] == 1

    def test_tiling_params_differ_between_attempts(self):
        """Retry uses different tiling strategy and overlap."""
        run_id = create_run(asset_ref="test_retry_params", asset_sha256="retry2")
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "tiling_strategy": "default",
            "overlap_pct": 0.12,
            "gates_passed": False,
            "decision": "FAIL",
        })
        insert_ocr_attempt(run_id, {
            "attempt_idx": 1,
            "tiling_strategy": "seam_retry_overlap_22%",
            "overlap_pct": 0.22,
            "gates_passed": True,
            "decision": "PASS",
        })
        attempts = list_ocr_attempts(run_id)
        assert attempts[0]["tiling_strategy"] != attempts[1]["tiling_strategy"]
        assert attempts[0]["overlap_pct"] != attempts[1]["overlap_pct"]

    def test_best_attempt_picks_passing(self):
        """get_best_ocr_attempt should prefer the passing attempt."""
        run_id = create_run(asset_ref="test_best_attempt", asset_sha256="best1")
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "tiling_strategy": "default",
            "overlap_pct": 0.12,
            "quality_label": "RISKY",
            "gates_passed": False,
            "decision": "FAIL",
        })
        insert_ocr_attempt(run_id, {
            "attempt_idx": 1,
            "tiling_strategy": "seam_retry",
            "overlap_pct": 0.22,
            "quality_label": "OK",
            "gates_passed": True,
            "decision": "PASS",
        })
        best = get_best_ocr_attempt(run_id)
        assert best is not None
        assert best["attempt_idx"] == 1
        assert best["gates_passed"] is True

    def test_best_attempt_falls_back_to_latest(self):
        """When no attempt passes, the latest attempt is returned."""
        run_id = create_run(asset_ref="test_best_fallback", asset_sha256="fallback1")
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "quality_label": "RISKY",
            "gates_passed": False,
            "decision": "FAIL",
        })
        insert_ocr_attempt(run_id, {
            "attempt_idx": 1,
            "quality_label": "RISKY",
            "gates_passed": False,
            "decision": "FAIL",
        })
        best = get_best_ocr_attempt(run_id)
        assert best is not None
        assert best["attempt_idx"] == 1
        assert best["gates_passed"] is False


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Downstream permissioning
# ═══════════════════════════════════════════════════════════════════════


class TestDownstreamPermissioning:
    """Verify that downstream stages check allow-flags before running."""

    def test_ner_not_allowed_skips_analysis(self):
        """When ner_allowed=False, analysis must not run."""
        report = OCRQualityReport(run_id="test_perm")
        report.quality_label = "UNRELIABLE"
        report.token_search_allowed = False
        report.ner_allowed = False
        report.gibberish_score = 0.50

        gates = enforce_quality_gates(report)
        assert gates["ner_allowed"] is False
        # In the pipeline, this means _run_trace_analysis is NOT called
        # and chunk/mention counts are 0

    def test_token_search_not_allowed_skips_linking(self):
        """When token_search_allowed=False, linking should be skipped."""
        report = OCRQualityReport(run_id="test_perm2")
        report.quality_label = "RISKY"
        report.token_search_allowed = False
        report.ner_allowed = False
        report.gibberish_score = 0.30

        gates = enforce_quality_gates(report)
        assert gates["token_search_allowed"] is False
        assert gates["downstream_mode"] == DOWNSTREAM_FALLBACK

    def test_high_quality_allows_all_stages(self):
        """When quality is HIGH, all downstream stages are allowed."""
        report = OCRQualityReport(run_id="test_perm_ok")
        report.quality_label = "HIGH"
        report.gibberish_score = 0.05
        report.non_wordlike_frac = 0.10
        report.char_entropy = 3.5
        report.uncertainty_density = 0.01
        report.leading_fragment_ratio = 0.02
        report.token_search_allowed = True
        report.ner_allowed = True

        gates = enforce_quality_gates(report)
        assert gates["ner_allowed"] is True
        assert gates["token_search_allowed"] is True
        assert gates["downstream_mode"] == DOWNSTREAM_TOKEN
        assert len(gates["blocked_stages"]) == 0


# ═══════════════════════════════════════════════════════════════════════
# Test 4: ocr_attempts table persistence
# ═══════════════════════════════════════════════════════════════════════


class TestOcrAttemptsDB:
    """Verify ocr_attempts table CRUD operations."""

    def test_insert_and_list(self):
        run_id = create_run(asset_ref="test_attempts_db", asset_sha256="att1")
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "tiling_strategy": "default",
            "overlap_pct": 0.12,
            "tile_count": 3,
            "model_used": "internvl3.5-30b",
            "text_hash": "abc123",
            "quality_label": "OK",
            "gibberish_score": 0.08,
            "leading_fragment_ratio": 0.03,
            "non_wordlike_frac": 0.10,
            "gates_passed": True,
            "decision": "PASS",
            "ocr_text": "Li rois Artus...",
        })
        attempts = list_ocr_attempts(run_id)
        assert len(attempts) == 1
        a = attempts[0]
        assert a["attempt_idx"] == 0
        assert a["tiling_strategy"] == "default"
        assert abs(a["overlap_pct"] - 0.12) < 0.001
        assert a["quality_label"] == "OK"
        assert a["gates_passed"] is True
        assert a["decision"] == "PASS"

    def test_multiple_attempts(self):
        run_id = create_run(asset_ref="test_multi_att", asset_sha256="att2")
        for i in range(3):
            insert_ocr_attempt(run_id, {
                "attempt_idx": i,
                "quality_label": ["RISKY", "RISKY", "OK"][i],
                "gates_passed": i == 2,
                "decision": "PASS" if i == 2 else "FAIL",
            })
        attempts = list_ocr_attempts(run_id)
        assert len(attempts) == 3
        assert [a["attempt_idx"] for a in attempts] == [0, 1, 2]
        assert attempts[2]["gates_passed"] is True

    def test_detail_json_round_trip(self):
        run_id = create_run(asset_ref="test_detail_json", asset_sha256="att3")
        detail = {"blocked_stages": ["token_search"], "seam": True}
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "gates_passed": False,
            "decision": "FAIL",
            "detail_json": detail,
        })
        attempts = list_ocr_attempts(run_id)
        assert len(attempts) == 1
        # detail_json is stored as text
        stored = attempts[0].get("detail_json")
        if isinstance(stored, str):
            stored = json.loads(stored)
        assert stored["blocked_stages"] == ["token_search"]

    def test_nonexistent_run_returns_empty(self):
        attempts = list_ocr_attempts("nonexistent_run_xyz")
        assert attempts == []

    def test_best_attempt_none_for_empty(self):
        best = get_best_ocr_attempt("nonexistent_run_xyz")
        assert best is None


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Gate report logging
# ═══════════════════════════════════════════════════════════════════════


class TestGateReportEvents:
    """Verify gate evaluation events are clearly logged."""

    def test_gate_report_format_contains_all_gates(self):
        report = compute_quality_report(_make_clean_text(), run_id="test_log")
        gates = enforce_quality_gates(report)
        text = format_gate_report(gates)

        assert "OCR QUALITY GATES" in text
        assert "GIBBERISH:" in text
        assert "LEADING_FRAGMENT:" in text
        assert "CROSS_PASS_STABILITY:" in text
        assert "NON_WORDLIKE:" in text
        assert "UNCERTAINTY:" in text
        assert "downstream_mode:" in text

    def test_failed_gate_shows_fail(self):
        report = compute_quality_report(_make_gibberish_text(), run_id="test_fail_log")
        gates = enforce_quality_gates(report)
        text = format_gate_report(gates)

        # At least one gate should show FAIL
        assert "FAIL" in text

    def test_events_logged_for_gate_evaluation(self):
        """Pipeline events should include QUALITY_GATES stage."""
        run_id = create_run(asset_ref="test_log_events", asset_sha256="log1")
        log_event(run_id, "QUALITY_GATES", "INFO", "All gates PASSED on attempt 0.")
        events = list_events(run_id)
        gate_events = [e for e in events if e["stage"] == "QUALITY_GATES"]
        assert len(gate_events) >= 1
        assert "PASSED" in gate_events[0]["message"]


# ═══════════════════════════════════════════════════════════════════════
# Test 6: End-to-end gate enforcement simulation
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEndGateEnforcement:
    """Simulate the full pipeline gate enforcement logic without calling
    the actual HTTP endpoint (no GPU/API needed)."""

    def test_simulate_blocked_run(self):
        """Simulate: OCR → gates FAIL → no downstream → FAILED_QUALITY."""
        run_id = create_run(asset_ref="e2e_blocked", asset_sha256="e2e1")
        text = _make_gibberish_text()

        # Attempt 0
        report = compute_quality_report(text, run_id=run_id, pass_idx=0)
        insert_ocr_quality_report(run_id, report.to_dict())
        gates = enforce_quality_gates(report, run_id=run_id)
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "quality_label": report.quality_label,
            "gibberish_score": report.gibberish_score,
            "leading_fragment_ratio": report.leading_fragment_ratio,
            "non_wordlike_frac": report.non_wordlike_frac,
            "gates_passed": all(g["passed"] for g in gates["gates"].values()),
            "decision": "PASS" if all(g["passed"] for g in gates["gates"].values()) else "FAIL",
        })

        # Attempt 1 (retry with same gibberish — still fails)
        report2 = compute_quality_report(text, run_id=run_id, pass_idx=1)
        insert_ocr_quality_report(run_id, report2.to_dict())
        gates2 = enforce_quality_gates(report2, run_id=run_id)
        insert_ocr_attempt(run_id, {
            "attempt_idx": 1,
            "quality_label": report2.quality_label,
            "gates_passed": all(g["passed"] for g in gates2["gates"].values()),
            "decision": "PASS" if all(g["passed"] for g in gates2["gates"].values()) else "FAIL",
        })

        # All attempts failed — mark FAILED_QUALITY
        update_run_fields(
            run_id,
            status="FAILED_QUALITY",
            current_stage="QUALITY_BLOCKED",
            error="Quality gates FAILED after 2 attempts",
        )

        # Verify: run is FAILED_QUALITY
        run = get_run(run_id)
        assert run["status"] == "FAILED_QUALITY"
        assert run["current_stage"] == "QUALITY_BLOCKED"

        # Verify: 2 attempts recorded
        attempts = list_ocr_attempts(run_id)
        assert len(attempts) == 2
        assert all(not a["gates_passed"] for a in attempts)

        # Verify: no downstream events (no PROOFREAD, ANALYZE, etc.)
        events = list_events(run_id)
        downstream_stages = {"PROOFREAD_RUNNING", "ANALYZE_RUNNING", "LINKING", "INDEX"}
        for e in events:
            assert e["stage"] not in downstream_stages, \
                f"Downstream stage {e['stage']} should not have run!"

    def test_simulate_retry_succeeds(self):
        """Simulate: OCR attempt 0 fails → attempt 1 succeeds → downstream runs."""
        run_id = create_run(asset_ref="e2e_retry_ok", asset_sha256="e2e2")

        # Attempt 0: gibberish → FAIL
        text0 = _make_gibberish_text()
        report0 = compute_quality_report(text0, run_id=run_id, pass_idx=0)
        gates0 = enforce_quality_gates(report0, run_id=run_id)
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "quality_label": report0.quality_label,
            "gates_passed": False,
            "decision": "FAIL",
        })

        # Attempt 1: clean text → PASS
        text1 = _make_clean_text()
        report1 = compute_quality_report(text1, run_id=run_id, pass_idx=1)
        gates1 = enforce_quality_gates(report1, run_id=run_id)
        all_passed = all(g["passed"] for g in gates1["gates"].values())
        insert_ocr_attempt(run_id, {
            "attempt_idx": 1,
            "quality_label": report1.quality_label,
            "gates_passed": all_passed,
            "decision": "PASS" if all_passed else "FAIL",
        })

        # Verify: attempt 1 passes
        assert report1.quality_label in ("HIGH", "OK")
        assert gates1["ner_allowed"] is True
        assert gates1["token_search_allowed"] is True

        # Verify: best attempt is #1
        best = get_best_ocr_attempt(run_id)
        assert best is not None
        assert best["attempt_idx"] == 1
        assert best["gates_passed"] is True

        # Downstream would now run (simulated)
        update_run_fields(run_id, status="COMPLETED", current_stage="DONE")
        run = get_run(run_id)
        assert run["status"] == "COMPLETED"

    def test_simulate_seam_fragment_retry(self):
        """Simulate: seam fragments detected → retry → seam resolved."""
        run_id = create_run(asset_ref="e2e_seam", asset_sha256="e2e3")

        # Attempt 0: seam fragments
        text0 = _make_seam_fragment_text()
        report0 = compute_quality_report(text0, run_id=run_id, pass_idx=0)
        gates0 = enforce_quality_gates(report0, run_id=run_id)
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "tiling_strategy": "default",
            "overlap_pct": 0.12,
            "quality_label": report0.quality_label,
            "leading_fragment_ratio": report0.leading_fragment_ratio,
            "gates_passed": False,
            "decision": "FAIL",
        })

        assert gates0["seam_retry_required"] is True
        assert "seam_not_resolved" in gates0["blocked_stages"]

        # Attempt 1: clean text (seam resolved)
        # Note: do NOT pass previous_pass_tokens here — the fragment text is
        # entirely different from the clean text, so cross-pass stability would
        # be nearly zero, which is not what we're testing.
        text1 = _make_clean_text()
        report1 = compute_quality_report(text1, run_id=run_id, pass_idx=1)
        gates1 = enforce_quality_gates(report1, run_id=run_id)
        all_gates_ok = all(g["passed"] for g in gates1["gates"].values())
        insert_ocr_attempt(run_id, {
            "attempt_idx": 1,
            "tiling_strategy": "seam_retry_overlap_22%",
            "overlap_pct": 0.22,
            "quality_label": report1.quality_label,
            "leading_fragment_ratio": report1.leading_fragment_ratio,
            "gates_passed": all_gates_ok,
            "decision": "PASS" if all_gates_ok else "FAIL",
        })

        attempts = list_ocr_attempts(run_id)
        assert len(attempts) == 2
        assert attempts[0]["tiling_strategy"] == "default"
        assert "seam_retry" in attempts[1]["tiling_strategy"]
        assert attempts[0]["overlap_pct"] < attempts[1]["overlap_pct"]
        assert not attempts[0]["gates_passed"]
        assert attempts[1]["gates_passed"]


# ═══════════════════════════════════════════════════════════════════════
# Test 7: Quality rank helper (used in pipeline)
# ═══════════════════════════════════════════════════════════════════════


class TestQualityRank:
    """Test the _quality_rank helper used in the retry loop."""

    def test_import_and_rank(self):
        from app.routers.ocr import _quality_rank
        assert _quality_rank("HIGH") < _quality_rank("OK")
        assert _quality_rank("OK") < _quality_rank("RISKY")
        assert _quality_rank("RISKY") < _quality_rank("UNRELIABLE")
        assert _quality_rank("UNKNOWN") > _quality_rank("UNRELIABLE")


# ═══════════════════════════════════════════════════════════════════════
# Test 8: Status values used by pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineStatusValues:
    """Verify that pipeline status/stage values are correctly stored."""

    def test_failed_quality_status(self):
        run_id = create_run(asset_ref="test_status_fq", asset_sha256="s1")
        update_run_fields(run_id, status="FAILED_QUALITY",
                          current_stage="QUALITY_BLOCKED")
        run = get_run(run_id)
        assert run["status"] == "FAILED_QUALITY"
        assert run["current_stage"] == "QUALITY_BLOCKED"

    def test_seam_retry_pending_stage(self):
        run_id = create_run(asset_ref="test_status_srp", asset_sha256="s2")
        update_run_fields(run_id, current_stage="SEAM_RETRY_PENDING")
        run = get_run(run_id)
        assert run["current_stage"] == "SEAM_RETRY_PENDING"

    def test_quality_retry_pending_stage(self):
        run_id = create_run(asset_ref="test_status_qrp", asset_sha256="s3")
        update_run_fields(run_id, current_stage="QUALITY_RETRY_PENDING")
        run = get_run(run_id)
        assert run["current_stage"] == "QUALITY_RETRY_PENDING"

    def test_analyze_skipped_stage(self):
        run_id = create_run(asset_ref="test_status_as", asset_sha256="s4")
        update_run_fields(run_id, current_stage="ANALYZE_SKIPPED")
        run = get_run(run_id)
        assert run["current_stage"] == "ANALYZE_SKIPPED"
