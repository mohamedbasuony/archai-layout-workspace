"""Tests for seam-aware retry strategies and NO-OP detection.

Covers the user's hard requirements:
  1. select_retry_strategy produces DIFFERENT tile_boxes for every attempt
  2. NO-OP guard detects identical text_sha256 or identical tile_boxes
  3. Verse text with short function words does NOT trigger LEADING_FRAGMENT
  4. EffectiveQuality unifies quality into one object
  5. Gate fail after max_attempts → quality_label RISKY/UNRELIABLE
  6. seam_fragment_ratio is computed and geometry-aware
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from app.services.seam_strategies import (
    TilingPlan,
    default_plan_from_suggestions,
    expand_overlap,
    grid_shift,
    is_noop_retry,
    plan_to_suggestions,
    seam_band_crop,
    select_retry_strategy,
)
from app.services.ocr_quality import (
    EffectiveQuality,
    LEADING_FRAG_HARD_LIMIT,
    SEAM_FRAG_HARD_LIMIT,
    OCRQualityReport,
    build_effective_quality,
    compute_quality_report,
    leading_fragment_ratio,
    seam_fragment_ratio,
)
from app.services.pipeline_hardening import (
    enforce_quality_gates,
)
from app.db.pipeline_db import (
    create_run,
    insert_ocr_attempt,
    list_ocr_attempts,
    get_best_ocr_attempt,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _default_3tile_plan(img_w: int = 2000, img_h: int = 3000) -> TilingPlan:
    """A simple 1×3 default plan (3 horizontal bands)."""
    tile_h = img_h // 3
    boxes = [
        (0, 0, img_w, tile_h + 50),
        (0, tile_h - 50, img_w, 2 * tile_h + 50),
        (0, 2 * tile_h - 50, img_w, img_h),
    ]
    return TilingPlan(
        strategy="default",
        grid="3x1",
        overlap_pct=0.12,
        tile_boxes=boxes,
    )


def _make_medieval_verse() -> str:
    """Synthetic medieval French verse with many function-word line-starts.

    Lines begin with "En", "Et", "Li", "Si", "De", "A", "Il", "Ne" —
    all legitimate function words that should NOT count as fragments.
    """
    return "\n".join([
        "En la forest de Brocéliande",
        "Li chevalier se sont armé",
        "Et ont monté sor lor chevaus",
        "Si s'en alerent par lo bois",
        "De tant fu il de grant valour",
        "A Kamaalot vint li rois Artus",
        "Il ot grant joie et grant pais",
        "Ne savoit nul qui il fust",
        "Li rois Artus apres la mort",
        "En la terre de Logres fu",
        "Et quant il ot establie",
        "Si com il li plut reuint",
        "De grant richece et de pooir",
        "A merveille estoit preudom",
        "Il tint la terre longuement",
        "Ne fu si bon chevalier el monde",
    ])


def _make_fragment_verse() -> str:
    """Text with genuine fragments (non-function-word debris) at line starts."""
    return "\n".join([
        "rt de la terre de Logres",
        "Li rois Artus apres la mort",
        "ce et de grant pooir",
        "Il estoit de grant richece",
        "xp ala forest de Brocéliande",
        "Et li chevalier de la Table",
        "vt ont monté sor lor chevaus",
        "Si s'en alerent par lo bois",
    ])


# ═══════════════════════════════════════════════════════════════════════
# Test 1: select_retry_strategy changes tile_boxes
# ═══════════════════════════════════════════════════════════════════════


class TestRetryChangesBoxes:
    """select_retry_strategy must produce a DIFFERENT boxes_signature."""

    def test_grid_shift_changes_boxes(self):
        plan0 = _default_3tile_plan()
        plan1 = grid_shift(plan0.tile_boxes, 2000, 3000, attempt_idx=1)
        assert plan1.boxes_signature() != plan0.boxes_signature()
        assert plan1.strategy == "grid_shift"
        assert len(plan1.tile_boxes) > 0

    def test_seam_band_crop_changes_boxes(self):
        plan0 = _default_3tile_plan()
        plan1 = seam_band_crop(plan0.tile_boxes, 2000, 3000, attempt_idx=1)
        assert plan1.boxes_signature() != plan0.boxes_signature()
        assert len(plan1.tile_boxes) > 0

    def test_expand_overlap_changes_boxes(self):
        plan0 = _default_3tile_plan()
        plan1 = expand_overlap(plan0.tile_boxes, 2000, 3000, attempt_idx=1)
        assert plan1.boxes_signature() != plan0.boxes_signature()
        assert plan1.strategy == "expand_overlap"

    def test_select_retry_always_different(self):
        """Three consecutive retries must all produce different tile_boxes."""
        plan0 = _default_3tile_plan()
        seen_sigs = {plan0.boxes_signature()}

        plan1 = select_retry_strategy(plan0, 2000, 3000, attempt_idx=1)
        assert plan1.boxes_signature() not in seen_sigs, \
            f"Attempt 1 produced same boxes as attempt 0"
        seen_sigs.add(plan1.boxes_signature())

        plan2 = select_retry_strategy(plan1, 2000, 3000, attempt_idx=2)
        assert plan2.boxes_signature() not in seen_sigs, \
            f"Attempt 2 produced same boxes as previous attempt(s)"

    def test_select_with_small_image(self):
        """Even a tiny image (100×100) gets different boxes."""
        plan0 = TilingPlan(
            strategy="default", grid="1x1", overlap_pct=0.0,
            tile_boxes=[(0, 0, 100, 100)],
        )
        plan1 = select_retry_strategy(plan0, 100, 100, attempt_idx=1)
        assert plan1.boxes_signature() != plan0.boxes_signature()

    def test_plan_to_suggestions_round_trip(self):
        """plan_to_suggestions → default_plan_from_suggestions gives same boxes."""
        plan0 = _default_3tile_plan()
        suggestions = plan_to_suggestions(plan0, 2000, 3000)
        assert len(suggestions) == len(plan0.tile_boxes)
        for s in suggestions:
            assert "bbox_xywh" in s
            assert "region_id" in s
            assert len(s["bbox_xywh"]) == 4

        # Round-trip
        plan_rt = default_plan_from_suggestions(suggestions, 2000, 3000)
        assert len(plan_rt.tile_boxes) == len(plan0.tile_boxes)
        # Boxes should be very close to originals (within clipping precision)
        for b_orig, b_rt in zip(plan0.tile_boxes, plan_rt.tile_boxes):
            for a, b in zip(b_orig, b_rt):
                assert abs(a - b) <= 1, f"Round-trip changed boxes: {b_orig} → {b_rt}"


# ═══════════════════════════════════════════════════════════════════════
# Test 2: NO-OP guard
# ═══════════════════════════════════════════════════════════════════════


class TestNoOpGuard:
    """is_noop_retry must detect identical text or identical tile boxes."""

    def test_identical_text_hash_is_noop(self):
        plan_a = _default_3tile_plan()
        plan_b = TilingPlan(
            strategy="grid_shift", grid="2x2", overlap_pct=0.15,
            tile_boxes=[(0, 0, 1000, 1500), (1000, 0, 2000, 1500),
                        (0, 1500, 1000, 3000), (1000, 1500, 2000, 3000)],
        )
        sha = hashlib.sha256(b"same text").hexdigest()
        assert is_noop_retry(sha, sha, plan_a, plan_b) is True

    def test_identical_boxes_is_noop(self):
        plan_a = _default_3tile_plan()
        plan_b = TilingPlan(
            strategy="grid_shift", grid="3x1", overlap_pct=0.12,
            tile_boxes=list(plan_a.tile_boxes),  # same boxes
        )
        sha_a = hashlib.sha256(b"text_a").hexdigest()
        sha_b = hashlib.sha256(b"text_b").hexdigest()
        assert is_noop_retry(sha_a, sha_b, plan_a, plan_b) is True

    def test_different_text_and_boxes_is_not_noop(self):
        plan_a = _default_3tile_plan()
        plan_b = grid_shift(plan_a.tile_boxes, 2000, 3000, attempt_idx=1)
        sha_a = hashlib.sha256(b"text_a").hexdigest()
        sha_b = hashlib.sha256(b"text_b").hexdigest()
        assert is_noop_retry(sha_a, sha_b, plan_a, plan_b) is False

    def test_empty_hashes_check_boxes_only(self):
        plan_a = _default_3tile_plan()
        plan_b = _default_3tile_plan()
        # Empty hashes → only box identity matters
        assert is_noop_retry("", "", plan_a, plan_b) is True


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Verse text doesn't fail LEADING_FRAGMENT
# ═══════════════════════════════════════════════════════════════════════


class TestVerseFunctionWords:
    """Medieval verse with function-word line starts should NOT fail the
    LEADING_FRAGMENT gate. This was the false-positive that prompted the fix."""

    def test_verse_leading_fragment_is_low(self):
        """Verse with 'En', 'Et', 'Li', 'Si', 'De', 'A', 'Il', 'Ne' starts
        should have leading_fragment_ratio ≈ 0.0 (not 0.5+)."""
        verse = _make_medieval_verse()
        lines = verse.splitlines()
        lfr = leading_fragment_ratio(lines, "latin")
        assert lfr < 0.05, (
            f"Verse text with function-word starts should have very low "
            f"leading_fragment_ratio, got {lfr:.4f}"
        )

    def test_verse_quality_report_not_risky(self):
        """Full quality report on verse text should be HIGH or OK, not RISKY."""
        verse = _make_medieval_verse()
        report = compute_quality_report(verse, run_id="test_verse")
        assert report.quality_label in ("HIGH", "OK"), (
            f"Expected HIGH/OK for verse, got {report.quality_label}"
        )
        assert report.leading_fragment_ratio < LEADING_FRAG_HARD_LIMIT
        assert report.seam_retry_required is False

    def test_verse_passes_all_gates(self):
        """Verse text should pass all quality gates without triggering retry."""
        verse = _make_medieval_verse()
        report = compute_quality_report(verse, run_id="test_verse_gates")
        gates = enforce_quality_gates(report, run_id="test_verse_gates")
        assert all(g["passed"] for g in gates["gates"].values()), (
            f"Verse text should pass all gates, but some failed: "
            + str({k: v for k, v in gates["gates"].items() if not v["passed"]})
        )
        assert gates["seam_retry_required"] is False

    def test_genuine_fragments_still_detected(self):
        """Non-function-word debris (like 'rt', 'ce', 'xp') should still
        be detected as fragments."""
        text = _make_fragment_verse()
        lines = text.splitlines()
        lfr = leading_fragment_ratio(lines, "latin")
        assert lfr >= 0.2, (
            f"Genuine OCR fragments should produce high leading_fragment_ratio, "
            f"got {lfr:.4f}"
        )

    def test_mixed_verse_and_fragments(self):
        """A text with mostly function-word starts but a few real fragments."""
        lines = [
            "En la forest de Brocéliande",
            "rt de la terre de Logres",          # genuine fragment
            "Et ont monté sor lor chevaus",
            "Li chevalier se sont armé",
            "vt ala forest de Brocéliande",       # genuine fragment
            "Si s'en alerent par lo bois",
            "De tant fu il de grant valour",
            "A Kamaalot vint li rois Artus",
        ]
        lfr = leading_fragment_ratio(lines, "latin")
        # 2 fragments out of 8 lines = 0.25
        assert 0.15 < lfr < 0.50, f"Expected ~0.25, got {lfr:.4f}"


# ═══════════════════════════════════════════════════════════════════════
# Test 4: EffectiveQuality unified object
# ═══════════════════════════════════════════════════════════════════════


class TestEffectiveQuality:
    def test_build_from_report_and_gates(self):
        report = compute_quality_report(
            "Li rois Artus apres la mort de son pere tint la terre.",
            run_id="test_eff_q",
        )
        gates = enforce_quality_gates(report, run_id="test_eff_q")
        eq = build_effective_quality(report, gates, confidence=0.85)

        assert eq.label in ("HIGH", "OK")
        assert eq.downstream in ("token_based", "vision_fallback")
        assert eq.confidence == 0.85
        assert eq.ner_allowed is True
        assert eq.token_search_allowed is True
        assert eq.seam_retry_required is False
        assert "gibberish_score" in eq.metrics
        assert "seam_fragment_ratio" in eq.metrics

    def test_to_dict_round_trip(self):
        eq = EffectiveQuality(
            label="RISKY",
            downstream="vision_fallback",
            confidence=0.42,
            ner_allowed=False,
            token_search_allowed=False,
            seam_retry_required=True,
            metrics={"gibberish_score": 0.35, "seam_fragment_ratio": 0.12},
        )
        d = eq.to_dict()
        assert d["label"] == "RISKY"
        assert d["confidence"] == 0.42
        assert d["ner_allowed"] is False
        assert d["metrics"]["gibberish_score"] == 0.35

    def test_risky_report_produces_risky_effective(self):
        """A RISKY quality report → EffectiveQuality with correct flags."""
        report = OCRQualityReport(run_id="test_eff_risky")
        report.gibberish_score = 0.30
        report.quality_label = "RISKY"
        report.ner_allowed = False
        report.token_search_allowed = False
        report.seam_retry_required = True

        gates = enforce_quality_gates(report, run_id="test_eff_risky")
        eq = build_effective_quality(report, gates, confidence=0.30)

        assert eq.label == "RISKY"
        assert eq.ner_allowed is False
        assert eq.seam_retry_required is True


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Gate fail after max_attempts
# ═══════════════════════════════════════════════════════════════════════


class TestGateFailAfterMaxAttempts:
    """If seam_retry_required after max_attempts, quality label should be RISKY/UNRELIABLE."""

    def test_seam_still_required_after_all_attempts(self):
        """Simulate: fragment text → 3 attempts → seam_fragment never resolved."""
        text = _make_fragment_verse()
        report = compute_quality_report(text, run_id="test_max_fail")
        gates = enforce_quality_gates(report, run_id="test_max_fail")

        assert report.leading_fragment_ratio >= LEADING_FRAG_HARD_LIMIT or \
               report.seam_fragment_ratio >= SEAM_FRAG_HARD_LIMIT
        assert gates["seam_retry_required"] is True
        assert "seam_not_resolved" in gates["blocked_stages"]

    def test_noop_detected_stored_in_db(self):
        """noop_detected should be persisted and retrievable."""
        run_id = create_run(asset_ref="test_noop_db", asset_sha256="noop1")
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "tiling_strategy": "default",
            "quality_label": "RISKY",
            "gates_passed": False,
            "noop_detected": False,
            "decision": "FAIL",
        })
        insert_ocr_attempt(run_id, {
            "attempt_idx": 1,
            "tiling_strategy": "grid_shift",
            "tile_grid": "2x2",
            "quality_label": "RISKY",
            "gates_passed": False,
            "noop_detected": True,
            "decision": "NOOP",
        })
        attempts = list_ocr_attempts(run_id)
        assert len(attempts) == 2
        assert attempts[0]["noop_detected"] is False
        assert attempts[1]["noop_detected"] is True
        assert attempts[1]["decision"] == "NOOP"

    def test_text_sha256_stored_in_db(self):
        """text_sha256 should be persisted and retrievable."""
        run_id = create_run(asset_ref="test_sha_db", asset_sha256="sha1")
        sha = hashlib.sha256(b"sample text").hexdigest()
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "text_sha256": sha,
            "quality_label": "OK",
            "gates_passed": True,
            "decision": "PASS",
        })
        attempts = list_ocr_attempts(run_id)
        assert attempts[0]["text_sha256"] == sha

    def test_tile_boxes_json_round_trip(self):
        """tile_boxes_json → JSON string → retrieved correctly."""
        run_id = create_run(asset_ref="test_boxes_rt", asset_sha256="brt1")
        boxes = [(0, 0, 1000, 1500), (0, 1500, 1000, 3000)]
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "tile_boxes_json": boxes,
            "quality_label": "OK",
            "gates_passed": True,
            "decision": "PASS",
        })
        attempts = list_ocr_attempts(run_id)
        import json
        stored_boxes = json.loads(attempts[0]["tile_boxes_json"])
        assert stored_boxes == [[0, 0, 1000, 1500], [0, 1500, 1000, 3000]]

    def test_effective_quality_json_round_trip(self):
        """effective_quality_json is stored and retrievable."""
        run_id = create_run(asset_ref="test_effq_rt", asset_sha256="eq1")
        eq = EffectiveQuality(
            label="OK", downstream="token_based", confidence=0.8,
            metrics={"gibberish_score": 0.05},
        )
        insert_ocr_attempt(run_id, {
            "attempt_idx": 0,
            "effective_quality_json": eq.to_dict(),
            "quality_label": "OK",
            "gates_passed": True,
            "decision": "PASS",
        })
        attempts = list_ocr_attempts(run_id)
        import json
        stored_eq = json.loads(attempts[0]["effective_quality_json"])
        assert stored_eq["label"] == "OK"
        assert stored_eq["confidence"] == 0.8


# ═══════════════════════════════════════════════════════════════════════
# Test 6: seam_fragment_ratio
# ═══════════════════════════════════════════════════════════════════════


class TestSeamFragmentRatio:
    """seam_fragment_ratio should detect fragments at seam boundaries."""

    def test_no_fragments_at_seams(self):
        lines = [
            "Li rois Artus apres la mort",
            "de son pere tint la terre",
            "Et quant il ot establie",
            "toute la terre en pais",
        ]
        sfr = seam_fragment_ratio(lines, seam_line_indices=[1, 2], script="latin")
        assert sfr < 0.05

    def test_fragments_at_seams(self):
        lines = [
            "Li rois Artus apres la mort",
            "rt de son pere tint la terre",     # seam → leading fragment "rt"
            "vt quant il ot establie",           # seam → leading fragment "vt"
            "toute la terre en pais",
        ]
        sfr = seam_fragment_ratio(lines, seam_line_indices=[1, 2], script="latin")
        assert sfr > 0.3, f"Expected >0.3 with fragments at seams, got {sfr:.4f}"

    def test_fragments_not_at_seams_ignored(self):
        """If fragments are at non-seam lines, seam_fragment_ratio ignores them."""
        lines = [
            "rt de la terre de Logres",           # fragment but NOT at a seam line
            "Li rois Artus apres la mort",
            "Et quant il ot establie",            # seam line → no fragment
            "toute la terre en pais",
        ]
        sfr = seam_fragment_ratio(lines, seam_line_indices=[2], script="latin")
        # Only seam line 2 is checked → "Et" is a function word → no fragment
        assert sfr < 0.1

    def test_default_fallback_checks_all(self):
        """Without seam_line_indices, all lines are checked."""
        lines = [
            "rt de la terre de Logres",
            "Li rois Artus apres la mort",
            "ce et de grant pooir",
            "toute la terre en pais",
        ]
        sfr = seam_fragment_ratio(lines, script="latin")
        assert sfr > 0.0  # at least some fragments detected

    def test_quality_report_includes_seam_fragment(self):
        """compute_quality_report should populate seam_fragment_ratio."""
        text = _make_fragment_verse()
        report = compute_quality_report(text, run_id="test_sfr")
        assert hasattr(report, "seam_fragment_ratio")
        assert isinstance(report.seam_fragment_ratio, float)


# ═══════════════════════════════════════════════════════════════════════
# Test 7: TilingPlan data integrity
# ═══════════════════════════════════════════════════════════════════════


class TestTilingPlanIntegrity:
    def test_boxes_signature_deterministic(self):
        plan = _default_3tile_plan()
        sig1 = plan.boxes_signature()
        sig2 = plan.boxes_signature()
        assert sig1 == sig2

    def test_different_boxes_different_signature(self):
        plan_a = _default_3tile_plan()
        plan_b = TilingPlan(
            strategy="test", grid="2x2", overlap_pct=0.0,
            tile_boxes=[(0, 0, 1000, 1500), (1000, 0, 2000, 1500),
                        (0, 1500, 1000, 3000), (1000, 1500, 2000, 3000)],
        )
        assert plan_a.boxes_signature() != plan_b.boxes_signature()

    def test_to_dict(self):
        plan = _default_3tile_plan()
        d = plan.to_dict()
        assert d["strategy"] == "default"
        assert d["grid"] == "3x1"
        assert len(d["tile_boxes"]) == 3

    def test_default_plan_from_empty_suggestions(self):
        """Empty suggestions → full image as single tile."""
        plan = default_plan_from_suggestions([], 2000, 3000)
        assert len(plan.tile_boxes) == 1
        assert plan.tile_boxes[0] == (0, 0, 2000, 3000)


# ═══════════════════════════════════════════════════════════════════════
# Test 8: Threshold constants
# ═══════════════════════════════════════════════════════════════════════


class TestThresholdConstants:
    def test_leading_frag_hard_limit_raised(self):
        """LEADING_FRAG_HARD_LIMIT should be 0.15 (raised from 0.08)."""
        assert LEADING_FRAG_HARD_LIMIT == 0.15

    def test_seam_frag_hard_limit(self):
        """SEAM_FRAG_HARD_LIMIT should be 0.10."""
        assert SEAM_FRAG_HARD_LIMIT == 0.10
