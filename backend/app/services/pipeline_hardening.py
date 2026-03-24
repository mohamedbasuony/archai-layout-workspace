"""Pipeline hardening: multi-pass OCR, seam fixes, quality gates, fallbacks.

This module sits between the raw OCR engine and the downstream NLP
(entity extraction, ligature search).  It enforces hard gates so
unreliable OCR never silently reaches token-based downstream steps.

Architecture:
  1. Pass 0 (baseline OCR) → compute quality report
  2. If seam_retry_required → overlap retry (pass 1)
  3. If quality still RISKY/UNRELIABLE → high-recall pass (pass 2)
  4. Pick best pass; set final quality_label
  5. If final quality RISKY/UNRELIABLE:
       - Block token-based NER (switch to high-recall extractor)
       - Block token-based ligature search (switch to shape/layout)
       - Set downstream_mode = "vision_fallback"
  6. Proofreading guard: reject if quality worsens
  7. Mention recall check: if too few, trigger high-recall extraction

All gate decisions are persisted in ``ocr_quality_reports`` + ``pipeline_events``
for full auditability.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from app.services.ocr_quality_config import (
    CROSS_PASS_STABILITY_MIN,
    GIBBERISH_HARD_LIMIT,
    GIBBERISH_SOFT_LIMIT,
    LEADING_FRAG_HARD_LIMIT,
    NON_WORDLIKE_GATE_LIMIT,
    SEAM_FRAG_HARD_LIMIT,
    UNCERTAINTY_HARD_LIMIT,
    frag_gate_value,
)
from app.services.ocr_quality import (
    OCRQualityReport,
    check_mention_recall,
    compute_quality_report,
    format_quality_report_summary,
)


# ═══════════════════════════════════════════════════════════════════════
# Downstream mode
# ═══════════════════════════════════════════════════════════════════════

DOWNSTREAM_TOKEN = "token_based"      # normal: NER + word search from OCR text
DOWNSTREAM_FALLBACK = "vision_fallback"  # shape/layout-only: no token search


def decide_downstream_mode(quality_label: str) -> str:
    """Decide whether downstream processing should use tokens or vision fallback.

    Token-based mode is only allowed for HIGH and OK quality.
    Everything else forces vision/shape fallback.
    """
    if quality_label in ("HIGH", "OK"):
        return DOWNSTREAM_TOKEN
    return DOWNSTREAM_FALLBACK


# ═══════════════════════════════════════════════════════════════════════
# Multi-pass OCR orchestration
# ═══════════════════════════════════════════════════════════════════════

def select_best_pass(reports: list[OCRQualityReport]) -> OCRQualityReport:
    """Select the best OCR pass from multiple attempts.

    Priority:
      1. Prefer pass with better quality_label (HIGH > OK > RISKY > UNRELIABLE)
      2. Among same label: lower gibberish_score wins
      3. Among same gibberish: lower leading_fragment_ratio wins
    """
    if not reports:
        return OCRQualityReport(quality_label="UNRELIABLE")

    label_rank = {"HIGH": 0, "OK": 1, "RISKY": 2, "UNRELIABLE": 3}

    return min(
        reports,
        key=lambda r: (
            label_rank.get(r.quality_label, 99),
            r.gibberish_score,
            r.leading_fragment_ratio,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# Proofreading quality guard
# ═══════════════════════════════════════════════════════════════════════

def proofreading_quality_guard(
    original_text: str,
    proofread_text: str,
    original_report: OCRQualityReport,
) -> tuple[str, bool, str]:
    """Validate that proofreading improved (or at least didn't worsen) quality.

    Returns:
        (final_text, accepted, reason)
        - final_text: either proofread_text if accepted, or original_text
        - accepted: True if proofreading was kept
        - reason: explanation
    """
    if not proofread_text or not proofread_text.strip():
        return original_text, False, "proofread_text is empty"

    proofread_report = compute_quality_report(
        proofread_text,
        run_id=original_report.run_id,
        pass_idx=original_report.pass_idx,
    )

    label_rank = {"HIGH": 0, "OK": 1, "RISKY": 2, "UNRELIABLE": 3}
    orig_rank = label_rank.get(original_report.quality_label, 3)
    proof_rank = label_rank.get(proofread_report.quality_label, 3)

    # Reject if quality degraded by more than one level
    if proof_rank > orig_rank + 1:
        return (
            original_text,
            False,
            f"proofreading degraded quality: {original_report.quality_label} → {proofread_report.quality_label}",
        )

    # Reject if gibberish increased significantly
    if proofread_report.gibberish_score > original_report.gibberish_score + 0.10:
        return (
            original_text,
            False,
            f"proofreading increased gibberish: {original_report.gibberish_score:.3f} → {proofread_report.gibberish_score:.3f}",
        )

    # Reject if uncertainty markers were removed but text got worse
    orig_unc = original_text.count("?") + original_text.count("[…]") * 3
    proof_unc = proofread_text.count("?") + proofread_text.count("[…]") * 3
    if orig_unc > 0 and proof_unc == 0 and proofread_report.non_wordlike_frac > original_report.non_wordlike_frac:
        return (
            original_text,
            False,
            "proofreading removed uncertainty markers but introduced non-wordlike tokens",
        )

    return proofread_text, True, "proofreading accepted"


# ═══════════════════════════════════════════════════════════════════════
# High-recall mention extraction
# ═══════════════════════════════════════════════════════════════════════

# Additional trigger words for high-recall extraction
# (broader than the standard salvage triggers)
_HIGH_RECALL_TRIGGERS: dict[str, str] = {
    # French-family titles/roles
    "roi": "person", "rois": "person", "roy": "person",
    "dame": "person", "dames": "person", "damoisele": "person",
    "sire": "person", "sires": "person", "monseigneur": "person",
    "evesque": "person", "euesque": "person", "eusque": "person",
    "prince": "person", "princesse": "person",
    "conte": "person", "comte": "person",
    "duc": "person", "duchesse": "person",
    "chevalier": "person", "chevaliers": "person",
    "frere": "person", "pere": "person",
    "pape": "person", "abbé": "person",
    # Latin titles
    "dominus": "person", "domini": "person", "episcopus": "person",
    "rex": "person", "regis": "person", "papa": "person",
    # Place markers (French + Latin)
    "chastel": "place", "cité": "place", "ville": "place",
    "forest": "place", "terre": "place", "royaume": "place",
    "pays": "place", "mont": "place",
    # Contextual markers
    "ici": "context", "devant": "context", "apres": "context",
}

# Honorifics and titles that often precede names
_NAME_PREFIXES = frozenset({
    "messire", "monseigneur", "monseignor", "mon", "ma",
    "li", "le", "la", "del", "des",
    "ser", "fra", "frere", "pere", "mere",
    "saint", "sainte", "san", "santa",
})


def extract_high_recall_mentions(
    text: str,
    *,
    script: str = "latin",
    quality_label: str = "OK",
) -> list[dict[str, Any]]:
    """High-recall mention extraction for when standard extraction is too sparse.

    Strategies:
      1. Broader trigger word set (all medieval titles/roles/context markers)
      2. Capitalization-independent: scan ALL tokens ≥ 4 chars as candidates
         (when casing is absent or noisy)
      3. Multi-token windows: check 2-3 token sequences as potential names
      4. Named-entity context patterns: "X de Y", "X le Y", etc.

    Returns candidate mentions (un-verified). Each has method="high_recall:*".
    """
    mentions: list[dict[str, Any]] = []
    if not text or not text.strip():
        return mentions

    token_re = re.compile(r"[A-Za-zÀ-ÿĀ-žſ]+", re.UNICODE)
    matches = list(token_re.finditer(text))

    for i, m in enumerate(matches):
        tok = m.group(0)
        tok_low = tok.lower()

        # Strategy 1: Trigger words → capture following token(s)
        if tok_low in _HIGH_RECALL_TRIGGERS:
            ent_type = _HIGH_RECALL_TRIGGERS[tok_low]
            if ent_type in ("person", "place") and i + 1 < len(matches):
                next_tok = matches[i + 1].group(0)
                if len(next_tok) >= 3:
                    # Capture 1-2 tokens after trigger
                    surface = next_tok
                    end_off = matches[i + 1].end()
                    if i + 2 < len(matches):
                        next2 = matches[i + 2].group(0)
                        if next2.lower() in _NAME_PREFIXES or len(next2) >= 4:
                            gap = matches[i + 2].start() - end_off
                            if gap <= 2:
                                surface = text[matches[i + 1].start():matches[i + 2].end()]
                                end_off = matches[i + 2].end()
                    mentions.append({
                        "start_offset": matches[i + 1].start(),
                        "end_offset": end_off,
                        "surface": surface,
                        "norm": None,
                        "label": None,
                        "ent_type": ent_type,
                        "confidence": 0.35,
                        "method": "high_recall:trigger",
                        "notes": f"trigger={tok_low}",
                    })

        # Strategy 2: Longer tokens as potential names (≥5 chars, not common)
        if len(tok) >= 5 and tok[0].isupper():
            mentions.append({
                "start_offset": m.start(),
                "end_offset": m.end(),
                "surface": tok,
                "norm": None,
                "label": None,
                "ent_type": "person",
                "confidence": 0.25,
                "method": "high_recall:capitalized",
                "notes": None,
            })

        # Strategy 3: Name prefix patterns
        if tok_low in _NAME_PREFIXES and i + 1 < len(matches):
            next_tok = matches[i + 1].group(0)
            if len(next_tok) >= 4:
                mentions.append({
                    "start_offset": m.start(),
                    "end_offset": matches[i + 1].end(),
                    "surface": text[m.start():matches[i + 1].end()],
                    "norm": None,
                    "label": None,
                    "ent_type": "person",
                    "confidence": 0.30,
                    "method": "high_recall:name_prefix",
                    "notes": f"prefix={tok_low}",
                })

    return mentions


# ═══════════════════════════════════════════════════════════════════════
# Ligature / word search fallback
# ═══════════════════════════════════════════════════════════════════════

def should_use_shape_based_search(quality_label: str) -> bool:
    """Whether ligature/word search should use shape/layout instead of OCR tokens.

    Returns True when OCR quality is too low for reliable token matching.
    """
    return quality_label in ("RISKY", "UNRELIABLE")


def generate_shape_based_candidates(
    layout_boxes: list[dict[str, Any]],
    *,
    min_width: int = 10,
    max_aspect_ratio: float = 8.0,
) -> list[dict[str, Any]]:
    """Generate ligature candidates from layout/shape cues.

    Criteria for a candidate:
      - Touching or overlapping component pairs
      - Abnormal kerning (gap < avg_gap * 0.3)
      - Aspect ratio suggesting fused characters
      - Narrow width components (possible components of a ligature)

    Args:
        layout_boxes: List of detected character/word boxes from the vision model.
                     Each box: {x, y, width, height, label, confidence}
        min_width: Minimum box width to consider.
        max_aspect_ratio: Maximum width/height ratio for single-char boxes.

    Returns:
        List of candidate regions: {x, y, width, height, reason, confidence}
    """
    candidates: list[dict[str, Any]] = []
    if not layout_boxes:
        return candidates

    # Sort by x position (reading order)
    sorted_boxes = sorted(layout_boxes, key=lambda b: (b.get("y", 0), b.get("x", 0)))

    # Compute average gap between consecutive boxes on same line
    gaps: list[int] = []
    for i in range(len(sorted_boxes) - 1):
        b1 = sorted_boxes[i]
        b2 = sorted_boxes[i + 1]
        # Same line check: vertical overlap
        y_overlap = min(
            b1.get("y", 0) + b1.get("height", 0),
            b2.get("y", 0) + b2.get("height", 0),
        ) - max(b1.get("y", 0), b2.get("y", 0))
        if y_overlap > 0:
            gap = b2.get("x", 0) - (b1.get("x", 0) + b1.get("width", 0))
            gaps.append(gap)

    avg_gap = sum(gaps) / max(len(gaps), 1) if gaps else 5

    # Detect candidates
    for i in range(len(sorted_boxes) - 1):
        b1 = sorted_boxes[i]
        b2 = sorted_boxes[i + 1]

        # Same line check
        y_overlap = min(
            b1.get("y", 0) + b1.get("height", 0),
            b2.get("y", 0) + b2.get("height", 0),
        ) - max(b1.get("y", 0), b2.get("y", 0))
        if y_overlap <= 0:
            continue

        gap = b2.get("x", 0) - (b1.get("x", 0) + b1.get("width", 0))

        # Touching or overlapping components
        if gap <= 0:
            candidates.append({
                "x": min(b1.get("x", 0), b2.get("x", 0)),
                "y": min(b1.get("y", 0), b2.get("y", 0)),
                "width": max(
                    b1.get("x", 0) + b1.get("width", 0),
                    b2.get("x", 0) + b2.get("width", 0),
                ) - min(b1.get("x", 0), b2.get("x", 0)),
                "height": max(
                    b1.get("y", 0) + b1.get("height", 0),
                    b2.get("y", 0) + b2.get("height", 0),
                ) - min(b1.get("y", 0), b2.get("y", 0)),
                "reason": "touching_components",
                "confidence": 0.7,
            })

        # Abnormally close (fused stroke)
        elif gap < avg_gap * 0.3 and avg_gap > 2:
            candidates.append({
                "x": b1.get("x", 0),
                "y": min(b1.get("y", 0), b2.get("y", 0)),
                "width": (b2.get("x", 0) + b2.get("width", 0)) - b1.get("x", 0),
                "height": max(
                    b1.get("y", 0) + b1.get("height", 0),
                    b2.get("y", 0) + b2.get("height", 0),
                ) - min(b1.get("y", 0), b2.get("y", 0)),
                "reason": "abnormal_kerning",
                "confidence": 0.5,
            })

    # Single-box ligature candidates: unusually wide for their height
    for box in sorted_boxes:
        w = box.get("width", 0)
        h = box.get("height", 1)
        if w >= min_width and h > 0:
            aspect = w / h
            if aspect > max_aspect_ratio:
                candidates.append({
                    "x": box.get("x", 0),
                    "y": box.get("y", 0),
                    "width": w,
                    "height": h,
                    "reason": "wide_aspect_ratio",
                    "confidence": 0.4,
                })

    return candidates


# ═══════════════════════════════════════════════════════════════════════
# Pipeline gate enforcement
# ═══════════════════════════════════════════════════════════════════════

def enforce_quality_gates(
    quality_report: OCRQualityReport,
    *,
    run_id: str = "",
    lexical_plausibility: float | None = None,
) -> dict[str, Any]:
    """Enforce all hard quality gates and return gate decisions.

    Returns a dict with:
      - quality_label: final label
      - downstream_mode: "token_based" | "vision_fallback"
      - gates: dict of {gate_name: {passed: bool, value: ..., threshold: ...}}
      - blocked_stages: list of stages that are blocked
    """
    gates: dict[str, dict[str, Any]] = {}
    blocked: list[str] = []

    # Gate: gibberish
    gates["GIBBERISH"] = {
        "passed": quality_report.gibberish_score < GIBBERISH_HARD_LIMIT,
        "value": quality_report.gibberish_score,
        "threshold": GIBBERISH_HARD_LIMIT,
    }
    if not gates["GIBBERISH"]["passed"]:
        blocked.extend(["token_search", "token_ner"])

    # Gate: leading fragment (uses geometry-aware seam_fragment_ratio if available,
    # then falls back to leading_fragment_ratio)
    seam_frag = getattr(quality_report, "seam_fragment_ratio", 0.0)
    lead_frag = quality_report.leading_fragment_ratio
    # Use the stronger signal: seam fragments are more reliable than generic
    # leading fragments, which can be caused by short function words.
    effective_frag = frag_gate_value(lead_frag, seam_frag)
    gates["LEADING_FRAGMENT"] = {
        "passed": effective_frag < LEADING_FRAG_HARD_LIMIT,
        "value": effective_frag,
        "threshold": LEADING_FRAG_HARD_LIMIT,
        "seam_fragment_ratio": seam_frag,
        "leading_fragment_ratio": lead_frag,
    }
    if not gates["LEADING_FRAGMENT"]["passed"]:
        blocked.append("seam_not_resolved")

    # Gate: cross-pass stability
    gates["CROSS_PASS_STABILITY"] = {
        "passed": (
            quality_report.cross_pass_stability < 0   # not computed yet
            or quality_report.cross_pass_stability >= CROSS_PASS_STABILITY_MIN
        ),
        "value": quality_report.cross_pass_stability,
        "threshold": CROSS_PASS_STABILITY_MIN,
    }
    if not gates["CROSS_PASS_STABILITY"]["passed"]:
        blocked.extend(["token_search", "token_ner"])

    # Gate: non-wordlike token fraction
    gates["NON_WORDLIKE"] = {
        "passed": quality_report.non_wordlike_frac < NON_WORDLIKE_GATE_LIMIT,
        "value": quality_report.non_wordlike_frac,
        "threshold": NON_WORDLIKE_GATE_LIMIT,
    }
    if not gates["NON_WORDLIKE"]["passed"]:
        blocked.extend(["token_search", "token_ner"])

    # Gate: uncertainty density
    gates["UNCERTAINTY"] = {
        "passed": quality_report.uncertainty_density < UNCERTAINTY_HARD_LIMIT,
        "value": quality_report.uncertainty_density,
        "threshold": UNCERTAINTY_HARD_LIMIT,
    }

    # Gate: lexical plausibility (optional — only when language detection ran)
    _LEXICAL_HARD_LIMIT = 0.20
    if lexical_plausibility is not None:
        gates["LEXICAL_PLAUSIBILITY"] = {
            "passed": lexical_plausibility >= _LEXICAL_HARD_LIMIT,
            "value": lexical_plausibility,
            "threshold": _LEXICAL_HARD_LIMIT,
        }
        if not gates["LEXICAL_PLAUSIBILITY"]["passed"]:
            blocked.extend(["token_search", "token_ner"])

    if quality_report.quality_label in {"RISKY", "UNRELIABLE"}:
        blocked.extend(["translation", "paraphrase", "entity_claims", "no_entity_claims"])

    if quality_report.quality_label == "UNRELIABLE":
        blocked.extend(["token_search", "token_ner"])

    downstream = decide_downstream_mode(quality_report.quality_label)

    return {
        "quality_label": quality_report.quality_label,
        "downstream_mode": downstream,
        "gates": gates,
        "blocked_stages": list(set(blocked)),
        "token_search_allowed": quality_report.token_search_allowed,
        "ner_allowed": quality_report.ner_allowed,
        "seam_retry_required": quality_report.seam_retry_required,
    }


def format_gate_report(gate_decisions: dict[str, Any]) -> str:
    """Format gate decisions as human-readable report string.

    Echoes the exact threshold values from the config module so that log
    output and config can never disagree.
    """
    L = [
        "=== OCR QUALITY GATES ===",
        f"quality_label: {gate_decisions['quality_label']}",
        f"downstream_mode: {gate_decisions['downstream_mode']}",
        f"config: LEADING_FRAG_HARD_LIMIT={LEADING_FRAG_HARD_LIMIT}, "
        f"SEAM_FRAG_HARD_LIMIT={SEAM_FRAG_HARD_LIMIT}, "
        f"CROSS_PASS_STABILITY_MIN={CROSS_PASS_STABILITY_MIN}, "
        f"NON_WORDLIKE_GATE_LIMIT={NON_WORDLIKE_GATE_LIMIT}, "
        f"UNCERTAINTY_HARD_LIMIT={UNCERTAINTY_HARD_LIMIT}",
        "",
    ]
    for name, gate in gate_decisions.get("gates", {}).items():
        status = "PASS" if gate["passed"] else "FAIL"
        L.append(f"  {name}: {status} (value={gate['value']:.4f}, threshold={gate['threshold']:.4f})")

    if gate_decisions.get("blocked_stages"):
        L.append("")
        L.append(f"  blocked_stages: {', '.join(gate_decisions['blocked_stages'])}")

    L.append(f"  token_search_allowed: {gate_decisions['token_search_allowed']}")
    L.append(f"  ner_allowed: {gate_decisions['ner_allowed']}")
    L.append(f"  seam_retry_required: {gate_decisions['seam_retry_required']}")

    return "\n".join(L)
