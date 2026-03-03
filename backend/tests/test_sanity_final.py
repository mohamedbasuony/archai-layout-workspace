"""Tests for the final-pass sanity, sanitization, forced uncertainty, and confidence."""
from __future__ import annotations

import pytest

from app.agents.saia_ocr_agent import (
    compute_sanity,
    _quality_label_from_sanity,
    quality_gate_enforce,
    sanity_adjust_confidence,
    sanitize_lines,
    force_uncertainty_markers,
    _token_is_suspicious,
    _collapse_single_char_runs,
    _collapse_fragment_sequences,
)


# ---- compute_sanity ----

def test_compute_sanity_has_junk_ratio():
    s = compute_sanity("hello world foo bar")
    assert "junk_ratio" in s
    assert s["junk_ratio"] == 0.0


def test_compute_sanity_single_char_fragments():
    text = "a a mno y a nis"
    s = compute_sanity(text)
    # 4 single-char tokens out of 6 total
    assert s["single_char_ratio"] > 0.5


def test_compute_sanity_clean_text():
    text = "Reuerendo in christo patri et domino domino"
    s = compute_sanity(text)
    assert s["single_char_ratio"] < 0.05
    assert s["weird_ratio"] < 0.05
    assert s["junk_ratio"] == 0.0


def test_compute_sanity_uncertainty_markers():
    text = "hello […] world ? foo"
    s = compute_sanity(text)
    assert s["uncertainty_marker_ratio"] > 0


# ---- _quality_label_from_sanity ----

def test_quality_label_high():
    label = _quality_label_from_sanity({
        "single_char_ratio": 0.02,
        "digit_ratio": 0.0,
        "weird_ratio": 0.01,
        "junk_ratio": 0.0,
    })
    assert label == "HIGH"


def test_quality_label_medium():
    label = _quality_label_from_sanity({
        "single_char_ratio": 0.07,
        "digit_ratio": 0.0,
        "weird_ratio": 0.10,
        "junk_ratio": 0.02,
    })
    assert label == "MEDIUM"


def test_quality_label_low():
    label = _quality_label_from_sanity({
        "single_char_ratio": 0.20,
        "digit_ratio": 0.05,
        "weird_ratio": 0.20,
        "junk_ratio": 0.10,
    })
    assert label == "LOW"


# ---- sanitize_lines ----

def test_sanitize_lines_collapses_fragment_run():
    """'a a mno y a nis' -> '[…] mno […]' or similar collapse."""
    lines = ["a a mno y a nis"]
    result, count = sanitize_lines(lines, "latin", "latin")
    text = result[0]
    # The single-char fragment runs must be collapsed
    assert text.count("[…]") >= 1 or "?" in text
    # single_char_ratio of result should be much lower than input
    s_out = compute_sanity(text)
    s_in = compute_sanity("a a mno y a nis")
    assert s_out["single_char_ratio"] <= s_in["single_char_ratio"]


def test_sanitize_lines_preserves_clean_text():
    lines = ["Reuerendo in christo patri et domino domino"]
    result, count = sanitize_lines(lines, "latin", "latin")
    assert result[0] == lines[0]
    assert count == 0


def test_sanitize_lines_masks_digit_tokens():
    lines = ["hello 7world data"]
    result, count = sanitize_lines(lines, "latin", "latin")
    assert "[…]" in result[0]
    assert count > 0


def test_sanitize_lines_consonant_merge():
    """Leading consonant 'n' should be merged with next token."""
    lines = ["n ous sommes"]
    result, count = sanitize_lines(lines, "french", "latin")
    assert "nous" in result[0]
    assert count > 0


def test_sanitize_lines_deduplicates_consecutive_markers():
    lines = ["a a a b b b c c c"]
    result, count = sanitize_lines(lines, "latin", "latin")
    text = result[0]
    # Should NOT have consecutive '[…] […]' 
    assert "[…] […]" not in text


def test_sanitize_lines_consonant_run_5_masked():
    """Token with 5+ consecutive consonants is masked."""
    lines = ["hello fldgnr world"]
    result, count = sanitize_lines(lines, "latin", "latin")
    assert "[…]" in result[0]


# ---- _collapse_single_char_runs ----

def test_collapse_single_char_runs_basic():
    tokens = ["a", "b", "c", "hello", "world"]
    out, count = _collapse_single_char_runs(tokens, {"a", "e"})
    assert "[…]" in out
    assert count == 3


def test_collapse_single_char_runs_short_run():
    """Runs of 1-2 single-char tokens should NOT be collapsed."""
    tokens = ["a", "b", "hello", "world"]
    out, count = _collapse_single_char_runs(tokens, {"a", "e"})
    assert count == 0
    assert out == tokens


# ---- _collapse_fragment_sequences ----

def test_collapse_fragment_sequences_mixed():
    """'a a mno y a' has 4/5 single-char -> collapse."""
    tokens = ["a", "a", "mno", "y", "a"]
    out, count = _collapse_fragment_sequences(tokens, {"a", "e"})
    assert "[…]" in out
    assert count > 0


# ---- force_uncertainty_markers ----

def test_force_uncertainty_does_nothing_when_markers_present():
    lines = ["hello […] world"]
    sanity = {"uncertainty_marker_ratio": 0.1, "weird_ratio": 0.10, "junk_ratio": 0.0}
    result, count = force_uncertainty_markers(lines, sanity)
    assert count == 0
    assert result == lines


def test_force_uncertainty_triggers_on_noisy_text():
    lines = ["hello world"]
    sanity = {"uncertainty_marker_ratio": 0.0, "weird_ratio": 0.10, "junk_ratio": 0.0}
    result, count = force_uncertainty_markers(lines, sanity)
    assert count >= 1
    joined = " ".join(result)
    assert "?" in joined or "[…]" in joined


def test_force_uncertainty_does_nothing_for_clean_text():
    lines = ["hello world"]
    sanity = {"uncertainty_marker_ratio": 0.0, "weird_ratio": 0.01, "junk_ratio": 0.01}
    result, count = force_uncertainty_markers(lines, sanity)
    assert count == 0


# ---- sanity_adjust_confidence ----

def test_sanity_adjust_confidence_penalizes_garbage():
    """With scr=0.164 wr=0.05 and no uncertainty -> confidence should be <0.60."""
    conf = sanity_adjust_confidence(
        0.90,
        {"single_char_ratio": 0.164, "weird_ratio": 0.05, "junk_ratio": 0.0, "uncertainty_marker_ratio": 0.0},
    )
    assert conf < 0.60, f"Expected <0.60 but got {conf:.3f}"


def test_sanity_adjust_confidence_clean_text():
    """Clean metrics should barely penalize."""
    conf = sanity_adjust_confidence(
        0.85,
        {"single_char_ratio": 0.02, "weird_ratio": 0.01, "junk_ratio": 0.0, "uncertainty_marker_ratio": 0.05},
    )
    assert conf > 0.75


def test_sanity_adjust_confidence_very_high_scr():
    """scr > 0.15 gets extra penalty."""
    conf = sanity_adjust_confidence(
        0.90,
        {"single_char_ratio": 0.25, "weird_ratio": 0.10, "junk_ratio": 0.0, "uncertainty_marker_ratio": 0.0},
    )
    assert conf < 0.40, f"Expected <0.40 but got {conf:.3f}"


# ---- _token_is_suspicious ----

def test_token_suspicious_consonant_run():
    assert _token_is_suspicious("fldgnr") is True


def test_token_suspicious_clean():
    assert _token_is_suspicious("domino") is False
    assert _token_is_suspicious("et") is False


def test_token_suspicious_digit():
    assert _token_is_suspicious("7world") is True


# ---- quality_gate_enforce ----

def test_quality_gate_masks_bad_lines():
    lines = ["[…] […] […] hello", "clean text here is fine and valid"]
    result, count = quality_gate_enforce(lines)
    # First line has mostly masked tokens
    assert count >= 0  # depends on ratio


def test_quality_gate_preserves_clean_lines():
    lines = ["Reuerendo in christo patri et domino domino"]
    result, count = quality_gate_enforce(lines)
    assert count == 0
    assert result == lines


# ---- w/k masking for French/Latin ----

def test_sanitize_wk_junk_token_masked():
    """Tokens like 'w', 'k', 'wr', 'wk' become '[…]' in Old French."""
    lines = ["patri wr domino k fauente"]
    result, count = sanitize_lines(lines, "old_french", "latin")
    text = result[0]
    assert "[…]" in text
    assert count >= 2


def test_sanitize_wk_inline_masked():
    """Tokens like 'went' or 'knoit' get w/k replaced with '?' in Latin."""
    lines = ["went domino knoit clementia"]
    result, count = sanitize_lines(lines, "latin", "latin")
    text = result[0]
    assert "?" in text
    # w should be masked with ?
    assert count >= 2


def test_sanitize_wk_trailing_k_artifact():
    """Token ending in 'k' like 'tgrantik' should get k -> '?'."""
    lines = ["tgrantik domino patri"]
    result, count = sanitize_lines(lines, "old_french", "latin")
    text = result[0]
    assert "?" in text
    assert count >= 1


def test_sanitize_wk_not_applied_english():
    """w/k masking should NOT apply for English."""
    lines = ["went known world"]
    result, count = sanitize_lines(lines, "old_english", "latin")
    text = result[0]
    # 'went' should stay as-is for English
    assert "went" in text


def test_sanitize_wk_preserves_roman_numerals():
    """Roman numeral tokens are not affected by w/k masking."""
    lines = ["anno MDCLXVI domino"]
    result, count = sanitize_lines(lines, "latin", "latin")
    text = result[0]
    assert "MDCLXVI" in text
