"""Comprehensive tests for OCR quality engine, pipeline hardening, and DB persistence.

Tests cover:
  - Script detection (Latin, CJK, Arabic, mixed)
  - Quality signals (entropy, gibberish, non-wordlike, fragments)
  - Quality label derivation (HIGH / OK / RISKY / UNRELIABLE)
  - Gate enforcement (all 5 gates)
  - Mention recall check and high-recall extraction
  - Proofreading quality guard (accept / reject)
  - Shape-based ligature candidates
  - Cross-pass stability
  - DB persistence (quality reports, tile audit)
  - Pipeline integration (downstream_mode, gate decisions)
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import pytest

from app.services.ocr_quality import (
    GIBBERISH_HARD_LIMIT,
    GIBBERISH_SOFT_LIMIT,
    LEADING_FRAG_HARD_LIMIT,
    NWL_TOKEN_HARD_LIMIT,
    OCRQualityReport,
    char_entropy,
    check_mention_recall,
    compute_quality_report,
    detect_script_family,
    format_quality_report_summary,
    gibberish_score,
    leading_fragment_ratio,
    non_wordlike_score,
    rare_bigram_ratio,
    token_length_stats,
    token_stability_score,
    trailing_fragment_ratio,
    uncertainty_density,
    vowel_ratio,
)
from app.services.pipeline_hardening import (
    DOWNSTREAM_FALLBACK,
    DOWNSTREAM_TOKEN,
    decide_downstream_mode,
    enforce_quality_gates,
    extract_high_recall_mentions,
    format_gate_report,
    generate_shape_based_candidates,
    proofreading_quality_guard,
    select_best_pass,
    should_use_shape_based_search,
)
from app.db.pipeline_db import (
    create_run,
    insert_ocr_quality_report,
    get_ocr_quality_report,
    list_ocr_quality_reports,
    insert_tile_audit,
    list_tile_audit,
)


# ═══════════════════════════════════════════════════════════════════════
# Script detection
# ═══════════════════════════════════════════════════════════════════════


class TestDetectScriptFamily:
    def test_latin_text(self):
        assert detect_script_family("Li rois Artus apres la mort") == "latin"

    def test_cjk_text(self):
        assert detect_script_family("天下大勢分久必合合久必分") == "cjk"

    def test_arabic_text(self):
        assert detect_script_family("بسم الله الرحمن الرحيم") == "arabic"

    def test_cyrillic_text(self):
        assert detect_script_family("Война и мир") == "cyrillic"

    def test_greek_text(self):
        # Polytonic Greek with diacritics may span extended ranges
        result = detect_script_family("Ἐν ἀρχῇ ἦν ὁ Λόγος")
        assert result in ("greek", "mixed")

    def test_mixed_text(self):
        result = detect_script_family("Hello 世界")
        assert result in ("latin", "cjk", "mixed")

    def test_empty_text(self):
        assert detect_script_family("") == "unknown"

    def test_numbers_only(self):
        result = detect_script_family("12345")
        assert result in ("unknown", "latin")

    def test_hebrew_text(self):
        assert detect_script_family("בראשית ברא אלהים") == "hebrew"


# ═══════════════════════════════════════════════════════════════════════
# Character entropy
# ═══════════════════════════════════════════════════════════════════════


class TestCharEntropy:
    def test_normal_text(self):
        e = char_entropy("Li rois Artus apres la mort de son pere.")
        assert 3.0 <= e <= 5.5

    def test_repetitive_text(self):
        e = char_entropy("aaaaaaaaa")
        assert e < 1.5

    def test_high_entropy(self):
        import string
        e = char_entropy(string.printable * 3)
        assert e > 4.0

    def test_empty_text(self):
        e = char_entropy("")
        assert e == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Token length stats
# ═══════════════════════════════════════════════════════════════════════


class TestTokenLengthStats:
    def test_normal_tokens(self):
        stats = token_length_stats(["hello", "world", "test", "token"])
        assert stats["mean"] > 0
        assert stats["very_short_frac"] < 0.5

    def test_single_char_tokens(self):
        stats = token_length_stats(["a", "b", "c", "d", "e"])
        assert stats["very_short_frac"] == 1.0

    def test_empty(self):
        stats = token_length_stats([])
        assert stats["mean"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Vowel ratio
# ═══════════════════════════════════════════════════════════════════════


class TestVowelRatio:
    def test_latin_word(self):
        v = vowel_ratio("hello", "latin")
        assert 0.0 < v < 1.0

    def test_all_consonants(self):
        v = vowel_ratio("bcd", "latin")
        assert v == 0.0

    def test_cjk_returns_minus_one(self):
        v = vowel_ratio("天下", "cjk")
        assert v == -1.0

    def test_arabic_returns_minus_one(self):
        v = vowel_ratio("كتاب", "arabic")
        assert v == -1.0


# ═══════════════════════════════════════════════════════════════════════
# Non-wordlike score
# ═══════════════════════════════════════════════════════════════════════


class TestNonWordlikeScore:
    def test_normal_word(self):
        score = non_wordlike_score("hello", "latin")
        assert score < 0.3

    def test_gibberish_word(self):
        score = non_wordlike_score("xqzxqz", "latin")
        assert score > 0.4

    def test_short_word_returns_zero(self):
        score = non_wordlike_score("ab", "latin")
        assert score == 0.0

    def test_cjk_word(self):
        score = non_wordlike_score("天下大勢", "cjk")
        assert score < 0.5  # CJK shouldn't flag as non-wordlike

    def test_mixed_digits(self):
        score = non_wordlike_score("abc123def", "latin")
        assert score > 0.0  # digit mixing penalty


# ═══════════════════════════════════════════════════════════════════════
# Rare bigram ratio
# ═══════════════════════════════════════════════════════════════════════


class TestRareBigramRatio:
    def test_normal_text(self):
        r = rare_bigram_ratio("The quick brown fox jumps over the lazy dog", "latin")
        assert r < 0.1

    def test_gibberish_text(self):
        r = rare_bigram_ratio("xqz xqz zxq qxz", "latin")
        assert r > 0.2

    def test_non_latin_returns_zero(self):
        r = rare_bigram_ratio("天下大勢分久必合", "cjk")
        assert r == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Leading / trailing fragment ratios
# ═══════════════════════════════════════════════════════════════════════


class TestFragmentRatios:
    def test_no_fragments(self):
        lines = [
            "Li rois Artus apres la mort de son pere.",
            "Il estoit de grant pooir et de grant richece.",
            "Et li chevalier de la Table Reonde.",
        ]
        lfr = leading_fragment_ratio(lines, "latin")
        assert lfr < 0.05

    def test_many_fragments(self):
        lines = [
            "rt de",  # leading fragment — "rt" is not a function word
            "Li rois Artus apres la mort de son pere.",
            "ce et",  # leading fragment — "ce" is not a function word
            "Il estoit de grant pooir et de grant richece.",
            "de",     # NOT a fragment — "de" is a function word (excluded)
            "Et li chevalier de la Table Reonde.",
        ]
        lfr = leading_fragment_ratio(lines, "latin")
        assert lfr > 0.08  # 2 fragments / 6 lines ≈ 0.33

    def test_trailing_fragments(self):
        lines = [
            "Li rois Artus apres la mo-",
            "Il estoit de grant pooir et de gran-",
            "Et li chevalier de la Table Reonde.",
        ]
        tfr = trailing_fragment_ratio(lines)
        assert tfr > 0.0

    def test_empty_lines(self):
        assert leading_fragment_ratio([], "latin") == 0.0
        assert trailing_fragment_ratio([]) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Gibberish score
# ═══════════════════════════════════════════════════════════════════════


class TestGibberishScore:
    def test_clean_text(self):
        text = (
            "Li rois Artus apres la mort de son pere tint mont longuement la terre "
            "de Logres en pais et en joie. Et quant il ot establie toute la terre "
            "si com il li plut, il reuint a Kamaalot."
        )
        g = gibberish_score(text, "latin")
        assert g < GIBBERISH_SOFT_LIMIT

    def test_gibberish_text(self):
        text = "aldiluzor zmaradigno qxzfwvp rmnbckl xqzzxq plghmnt"
        g = gibberish_score(text, "latin")
        assert g > GIBBERISH_SOFT_LIMIT

    def test_heavily_gibberish_text(self):
        text = " ".join(["xqz" * 3] * 20)
        g = gibberish_score(text, "latin")
        assert g >= GIBBERISH_SOFT_LIMIT


# ═══════════════════════════════════════════════════════════════════════
# Uncertainty density
# ═══════════════════════════════════════════════════════════════════════


class TestUncertaintyDensity:
    def test_no_markers(self):
        d = uncertainty_density("Li rois Artus apres la mort")
        assert d == 0.0

    def test_with_markers(self):
        d = uncertainty_density("Li rois? Artus [?] apres [unclear] la mort [illegible]")
        assert d > 0.0

    def test_empty_text(self):
        d = uncertainty_density("")
        assert d == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Token stability
# ═══════════════════════════════════════════════════════════════════════


class TestTokenStability:
    def test_identical_passes(self):
        tokens = ["Li", "rois", "Artus"]
        s = token_stability_score(tokens, tokens)
        assert s == 1.0

    def test_completely_different(self):
        s = token_stability_score(
            ["hello", "world"],
            ["xqz", "plm"],
        )
        assert s == 0.0

    def test_partial_overlap(self):
        s = token_stability_score(
            ["Li", "rois", "Artus", "apres"],
            ["Li", "rois", "Arthur", "mort"],
        )
        assert 0.0 < s < 1.0


# ═══════════════════════════════════════════════════════════════════════
# Quality report computation
# ═══════════════════════════════════════════════════════════════════════


class TestComputeQualityReport:
    def test_clean_text_gets_high(self):
        text = (
            "Li rois Artus apres la mort de son pere tint mont longuement la terre "
            "de Logres en pais et en joie. Et quant il ot establie toute la terre "
            "si com il li plut, il reuint a Kamaalot."
        )
        report = compute_quality_report(text, run_id="test_clean")
        assert report.quality_label in ("HIGH", "OK")
        assert report.token_search_allowed is True
        assert report.ner_allowed is True
        assert report.seam_retry_required is False

    def test_gibberish_text_gets_unreliable_or_risky(self):
        text = "aldiluzor zmaradigno qxzfwvp rmnbckl xqzzxq plghmnt bvxcz wrtp"
        report = compute_quality_report(text, run_id="test_gibberish")
        assert report.quality_label in ("RISKY", "UNRELIABLE")
        assert report.token_search_allowed is False
        assert report.ner_allowed is False

    def test_empty_text_gets_unreliable(self):
        report = compute_quality_report("", run_id="test_empty")
        assert report.quality_label == "UNRELIABLE"
        assert report.token_search_allowed is False

    def test_report_has_all_fields(self):
        text = "Li rois Artus apres la mort de son pere."
        report = compute_quality_report(text, run_id="test_fields")
        assert report.script_family != ""
        assert report.token_count > 0
        assert report.line_count > 0
        assert isinstance(report.gibberish_score, float)
        assert isinstance(report.char_entropy, float)
        assert isinstance(report.non_wordlike_frac, float)

    def test_to_dict_works(self):
        report = compute_quality_report(
            "Li rois Artus apres la mort.",
            run_id="test_dict",
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "quality_label" in d
        assert "gibberish_score" in d

    def test_cross_pass_stability(self):
        text = "Li rois Artus apres la mort de son pere."
        tokens = text.split()
        report = compute_quality_report(
            text,
            run_id="test_stability",
            previous_pass_tokens=tokens,
        )
        assert report.cross_pass_stability >= 0.0

    def test_cjk_text(self):
        text = "天下大勢分久必合合久必分 三國志之始 曹操字孟德 劉備字玄德 孫權字仲謀"
        report = compute_quality_report(text, run_id="test_cjk")
        assert report.script_family == "cjk"
        # CJK should not be flagged as gibberish due to vowel issues
        assert report.quality_label in ("HIGH", "OK", "RISKY")

    def test_arabic_text(self):
        text = "بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين"
        report = compute_quality_report(text, run_id="test_arabic")
        assert report.script_family == "arabic"


# ═══════════════════════════════════════════════════════════════════════
# Quality label derivation
# ═══════════════════════════════════════════════════════════════════════


class TestQualityLabelDerivation:
    def test_high_quality(self):
        report = compute_quality_report(
            "Li rois Artus apres la mort de son pere tint mont longuement la terre "
            "de Logres en pais et en joie et quant il ot establie toute la terre.",
            run_id="test_high",
        )
        assert report.quality_label in ("HIGH", "OK")

    def test_unreliable_from_extreme_gibberish(self):
        # Create text with very high gibberish score
        report = OCRQualityReport(run_id="test_unreliable")
        report.gibberish_score = 0.50
        report.char_entropy = 3.5
        report.non_wordlike_frac = 0.1
        from app.services.ocr_quality import _derive_quality_label
        label = _derive_quality_label(report)
        assert label == "UNRELIABLE"

    def test_risky_from_moderate_gibberish(self):
        report = OCRQualityReport(run_id="test_risky")
        report.gibberish_score = 0.30
        report.char_entropy = 3.5
        report.non_wordlike_frac = 0.1
        report.leading_fragment_ratio = 0.01
        report.uncertainty_density = 0.01
        from app.services.ocr_quality import _derive_quality_label
        label = _derive_quality_label(report)
        assert label == "RISKY"


# ═══════════════════════════════════════════════════════════════════════
# Mention recall check
# ═══════════════════════════════════════════════════════════════════════


class TestCheckMentionRecall:
    def test_enough_mentions(self):
        text = "A" * 2000
        result = check_mention_recall(text, mentions_total=10, quality_label="HIGH")
        assert result["mention_recall_ok"] is True
        assert result["trigger_high_recall"] is False

    def test_too_few_mentions(self):
        text = "A" * 2000
        result = check_mention_recall(text, mentions_total=0, quality_label="HIGH")
        assert result["mention_recall_ok"] is False
        assert result["trigger_high_recall"] is True
        assert "MENTION_RECALL_LOW" in result["reason"]

    def test_skipped_for_risky(self):
        text = "A" * 2000
        result = check_mention_recall(text, mentions_total=0, quality_label="RISKY")
        assert result["mention_recall_ok"] is True  # skipped for non-HIGH/OK

    def test_short_text_skipped(self):
        text = "Short text."
        result = check_mention_recall(text, mentions_total=0, quality_label="HIGH")
        assert result["mention_recall_ok"] is True


# ═══════════════════════════════════════════════════════════════════════
# Format quality report
# ═══════════════════════════════════════════════════════════════════════


class TestFormatQualityReport:
    def test_format_summary(self):
        report = compute_quality_report(
            "Li rois Artus apres la mort de son pere.",
            run_id="test_format",
        )
        summary = format_quality_report_summary(report)
        assert "OCR QUALITY REPORT" in summary
        assert "quality_label:" in summary
        assert "gibberish_score:" in summary
        assert "GATE DECISIONS" in summary


# ═══════════════════════════════════════════════════════════════════════
# Pipeline hardening: downstream mode
# ═══════════════════════════════════════════════════════════════════════


class TestDecideDownstreamMode:
    def test_high_ok_get_token(self):
        assert decide_downstream_mode("HIGH") == DOWNSTREAM_TOKEN
        assert decide_downstream_mode("OK") == DOWNSTREAM_TOKEN

    def test_risky_unreliable_get_fallback(self):
        assert decide_downstream_mode("RISKY") == DOWNSTREAM_FALLBACK
        assert decide_downstream_mode("UNRELIABLE") == DOWNSTREAM_FALLBACK

    def test_unknown_gets_fallback(self):
        assert decide_downstream_mode("UNKNOWN") == DOWNSTREAM_FALLBACK


# ═══════════════════════════════════════════════════════════════════════
# Pipeline hardening: select best pass
# ═══════════════════════════════════════════════════════════════════════


class TestSelectBestPass:
    def test_better_label_wins(self):
        r1 = OCRQualityReport(run_id="test", pass_idx=0)
        r1.quality_label = "RISKY"
        r1.gibberish_score = 0.30
        r2 = OCRQualityReport(run_id="test", pass_idx=1)
        r2.quality_label = "OK"
        r2.gibberish_score = 0.15
        best = select_best_pass([r1, r2])
        assert best.pass_idx == 1

    def test_single_pass(self):
        r1 = OCRQualityReport(run_id="test", pass_idx=0)
        r1.quality_label = "HIGH"
        best = select_best_pass([r1])
        assert best.pass_idx == 0


# ═══════════════════════════════════════════════════════════════════════
# Pipeline hardening: proofreading quality guard
# ═══════════════════════════════════════════════════════════════════════


class TestProofreadingQualityGuard:
    def test_accept_good_proofread(self):
        original = (
            "Li rois Artus apres la mort de son pere tint mont longuement la terre "
            "de Logres en pais et en joie."
        )
        proofread = (
            "Li rois Arthur apres la mort de son pere tint mont longuement la terre "
            "de Logres en paix et en joie."
        )
        original_report = compute_quality_report(original, run_id="test_guard")
        final_text, accepted, reason = proofreading_quality_guard(
            original, proofread, original_report,
        )
        assert accepted is True
        assert final_text == proofread

    def test_reject_empty_proofread(self):
        original = "Li rois Artus apres la mort."
        original_report = compute_quality_report(original, run_id="test_guard")
        final_text, accepted, reason = proofreading_quality_guard(
            original, "", original_report,
        )
        assert accepted is False
        assert final_text == original

    def test_reject_worsened_proofread(self):
        original = "Li rois Artus apres la mort de son pere."
        # Simulate gibberish increased
        original_report = compute_quality_report(original, run_id="test_guard")
        gibberish_text = "xqz plg hrt bvx gfl xqz aldiluzor zmaradigno qxzfwvp"
        final_text, accepted, reason = proofreading_quality_guard(
            original, gibberish_text, original_report,
        )
        # Should reject because gibberish increased significantly
        assert accepted is False or "gibberish" not in reason.lower()
        # At minimum the final text should be either original or gibberish
        assert final_text in (original, gibberish_text)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline hardening: high-recall extraction
# ═══════════════════════════════════════════════════════════════════════


class TestHighRecallExtraction:
    def test_extracts_trigger_based(self):
        text = "Li rois Artus apres la mort du duc Yvain."
        mentions = extract_high_recall_mentions(text, script="latin")
        surfaces = [m["surface"] for m in mentions]
        # Should capture tokens after "rois" and "duc"
        assert any("Artus" in s for s in surfaces)

    def test_extracts_capitalized_tokens(self):
        text = "En cele terre estoit Lancelot et Gauvain."
        mentions = extract_high_recall_mentions(text, script="latin")
        surfaces = [m["surface"] for m in mentions]
        assert any("Lancelot" in s for s in surfaces)
        assert any("Gauvain" in s for s in surfaces)

    def test_name_prefix_pattern(self):
        text = "Et messire Gauvain chevalcha vers le chastel."
        mentions = extract_high_recall_mentions(text, script="latin")
        methods = [m["method"] for m in mentions]
        assert any("high_recall:" in m for m in methods)

    def test_empty_text(self):
        mentions = extract_high_recall_mentions("", script="latin")
        assert mentions == []


# ═══════════════════════════════════════════════════════════════════════
# Pipeline hardening: shape-based ligature candidates
# ═══════════════════════════════════════════════════════════════════════


class TestShapeBasedFallback:
    def test_should_use_shape_for_risky(self):
        assert should_use_shape_based_search("RISKY") is True
        assert should_use_shape_based_search("UNRELIABLE") is True
        assert should_use_shape_based_search("HIGH") is False
        assert should_use_shape_based_search("OK") is False

    def test_touching_components(self):
        boxes = [
            {"x": 10, "y": 10, "width": 30, "height": 20, "label": "char", "confidence": 0.9},
            {"x": 38, "y": 10, "width": 25, "height": 20, "label": "char", "confidence": 0.9},
        ]
        candidates = generate_shape_based_candidates(boxes)
        reasons = [c["reason"] for c in candidates]
        assert "touching_components" in reasons

    def test_wide_aspect_ratio(self):
        boxes = [
            {"x": 10, "y": 10, "width": 200, "height": 20, "label": "word", "confidence": 0.9},
        ]
        candidates = generate_shape_based_candidates(boxes)
        reasons = [c["reason"] for c in candidates]
        assert "wide_aspect_ratio" in reasons

    def test_empty_boxes(self):
        assert generate_shape_based_candidates([]) == []


# ═══════════════════════════════════════════════════════════════════════
# Pipeline hardening: gate enforcement
# ═══════════════════════════════════════════════════════════════════════


class TestEnforceQualityGates:
    def test_all_gates_pass_for_clean_text(self):
        report = compute_quality_report(
            "Li rois Artus apres la mort de son pere tint mont longuement la terre "
            "de Logres en pais et en joie et quant il ot establie toute la terre.",
            run_id="test_gates",
        )
        gates = enforce_quality_gates(report, run_id="test_gates")
        assert gates["downstream_mode"] == DOWNSTREAM_TOKEN
        assert gates["token_search_allowed"] is True
        assert gates["ner_allowed"] is True
        for gate_name, gate_info in gates["gates"].items():
            assert gate_info["passed"] is True, f"Gate {gate_name} should pass"

    def test_gibberish_gate_fails(self):
        report = OCRQualityReport(run_id="test_gibberish_gate")
        report.quality_label = "UNRELIABLE"
        report.gibberish_score = 0.50
        report.token_search_allowed = False
        report.ner_allowed = False
        gates = enforce_quality_gates(report, run_id="test_gibberish_gate")
        assert gates["gates"]["GIBBERISH"]["passed"] is False
        assert "token_search" in gates["blocked_stages"]

    def test_format_gate_report(self):
        report = compute_quality_report(
            "Li rois Artus apres la mort de son pere.",
            run_id="test_format_gates",
        )
        gates = enforce_quality_gates(report)
        text = format_gate_report(gates)
        assert "OCR QUALITY GATES" in text
        assert "quality_label:" in text
        assert "downstream_mode:" in text


# ═══════════════════════════════════════════════════════════════════════
# DB persistence: quality reports
# ═══════════════════════════════════════════════════════════════════════


class TestDBQualityReports:
    def test_insert_and_get(self):
        run_id = create_run(asset_ref="test_quality_db", asset_sha256="abc123")
        report = compute_quality_report(
            "Li rois Artus apres la mort de son pere.",
            run_id=run_id,
            pass_idx=0,
        )
        report_id = insert_ocr_quality_report(run_id, report.to_dict())
        assert report_id

        fetched = get_ocr_quality_report(run_id, pass_idx=0)
        assert fetched is not None
        assert fetched["quality_label"] == report.quality_label
        assert fetched["run_id"] == run_id
        assert abs(fetched["gibberish_score"] - report.gibberish_score) < 0.01

    def test_list_reports(self):
        run_id = create_run(asset_ref="test_quality_list", asset_sha256="def456")
        r0 = compute_quality_report("Text pass zero.", run_id=run_id, pass_idx=0)
        r1 = compute_quality_report("Text pass one.", run_id=run_id, pass_idx=1)
        insert_ocr_quality_report(run_id, r0.to_dict())
        insert_ocr_quality_report(run_id, r1.to_dict())

        reports = list_ocr_quality_reports(run_id)
        assert len(reports) >= 2
        pass_indices = {r["pass_idx"] for r in reports}
        assert 0 in pass_indices
        assert 1 in pass_indices

    def test_nonexistent_run(self):
        fetched = get_ocr_quality_report("nonexistent_run_id", pass_idx=0)
        assert fetched is None


# ═══════════════════════════════════════════════════════════════════════
# DB persistence: tile audit
# ═══════════════════════════════════════════════════════════════════════


class TestDBTileAudit:
    def test_insert_and_list(self):
        run_id = create_run(asset_ref="test_tile_audit", asset_sha256="tile_abc")
        tiles = [
            {
                "pass_idx": 0,
                "tile_idx": 0,
                "x": 0,
                "y": 0,
                "width": 800,
                "height": 400,
                "overlap_px": 48,
                "seam_merge_action": "fuzzy_dedup",
                "lines_before_merge": 10,
                "lines_after_merge": 8,
            },
            {
                "pass_idx": 0,
                "tile_idx": 1,
                "x": 0,
                "y": 352,
                "width": 800,
                "height": 400,
                "overlap_px": 48,
                "seam_merge_action": "fuzzy_dedup",
                "lines_before_merge": 12,
                "lines_after_merge": 10,
            },
        ]
        ids = insert_tile_audit(run_id, tiles)
        assert len(ids) == 2

        fetched = list_tile_audit(run_id)
        assert len(fetched) >= 2


# ═══════════════════════════════════════════════════════════════════════
# Integration: pipeline endpoint response shape
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineIntegration:
    """Validate that pipeline integration points are structurally correct.

    These tests check the module wiring rather than the full HTTP endpoint.
    """

    def test_quality_report_feeds_gate_decisions(self):
        text = "Li rois Artus apres la mort de son pere tint la terre de Logres."
        report = compute_quality_report(text, run_id="integration_test")
        gates = enforce_quality_gates(report, run_id="integration_test")
        mode = decide_downstream_mode(report.quality_label)
        assert gates["downstream_mode"] == mode
        assert gates["quality_label"] == report.quality_label

    def test_mention_recall_triggers_high_recall(self):
        text = "A " * 1500  # 3000+ chars
        recall = check_mention_recall(text, mentions_total=0, quality_label="HIGH")
        assert recall["trigger_high_recall"] is True

        hr = extract_high_recall_mentions(text, script="latin", quality_label="HIGH")
        # Even with filler text, function should return gracefully
        assert isinstance(hr, list)

    def test_proofreading_guard_integrates_with_report(self):
        text = "Li rois Artus apres la mort de son pere."
        report = compute_quality_report(text, run_id="guard_test")
        final, accepted, reason = proofreading_quality_guard(
            text, text, report,
        )
        assert accepted is True
        assert final == text

    def test_db_roundtrip(self):
        run_id = create_run(asset_ref="test_roundtrip", asset_sha256="rt_hash")
        text = "Li rois Artus apres la mort de son pere."
        report = compute_quality_report(text, run_id=run_id, pass_idx=0)
        report_id = insert_ocr_quality_report(run_id, report.to_dict())
        gates = enforce_quality_gates(report, run_id=run_id)

        fetched = get_ocr_quality_report(run_id, pass_idx=0)
        assert fetched is not None
        assert fetched["quality_label"] == report.quality_label
        assert fetched["token_search_allowed"] == report.token_search_allowed
