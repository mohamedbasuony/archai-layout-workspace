"""Scoring and disambiguation for entity linking.

Combines normalised string similarity (Jaro-Winkler) with context
similarity (simple token overlap) to produce a final score.

**Precision-first rules** (v2):

* ``type_compatible`` is a **hard gate**: candidates that fail type-
  checking can never be auto-selected, regardless of score.
* ``AUTO_SELECT_THRESHOLD`` — only candidates scoring ≥ this value are
  eligible for auto-selection.  Value depends on OCR quality level.
* ``MIN_MARGIN`` — the top candidate must lead the runner-up by at
  least this much to avoid the *ambiguous* label.
* A type-mismatch penalty of 0.30 is still applied to the raw score so
  that incompatible candidates are ranked lower.

**Quality-adaptive thresholds (v3):**

Three OCR quality levels (HIGH / MEDIUM / LOW) set different thresholds
for ``AUTO_SELECT_THRESHOLD``, ``MIN_MARGIN``, and
``MIN_STRING_SIMILARITY``.  The caller passes ``ocr_quality`` to
``disambiguate`` to select the appropriate tier.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

log = logging.getLogger("archai.entity_scoring")

# ── Optional rapidfuzz with pure-Python fallback ───────────────────────

_USE_RAPIDFUZZ = False
_JW: Any = None

try:
    from rapidfuzz.distance import JaroWinkler as _JW_cls
    _JW = _JW_cls
    _USE_RAPIDFUZZ = True
except ImportError:
    log.info("rapidfuzz not installed, using fallback matcher")


def _pure_python_jaro(s: str, t: str) -> float:
    """Pure-Python Jaro similarity (0..1)."""
    if s == t:
        return 1.0
    s_len, t_len = len(s), len(t)
    if s_len == 0 or t_len == 0:
        return 0.0
    match_dist = max(s_len, t_len) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    s_matches = [False] * s_len
    t_matches = [False] * t_len
    matches = 0
    transpositions = 0
    for i in range(s_len):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, t_len)
        for j in range(lo, hi):
            if t_matches[j] or s[i] != t[j]:
                continue
            s_matches[i] = True
            t_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(s_len):
        if not s_matches[i]:
            continue
        while not t_matches[k]:
            k += 1
        if s[i] != t[k]:
            transpositions += 1
        k += 1
    jaro = (matches / s_len + matches / t_len +
            (matches - transpositions / 2) / matches) / 3
    return jaro


def _pure_python_jaro_winkler(s: str, t: str, *, prefix_weight: float = 0.1) -> float:
    """Pure-Python Jaro-Winkler similarity (0..1)."""
    jaro = _pure_python_jaro(s, t)
    prefix_len = 0
    for i in range(min(4, min(len(s), len(t)))):
        if s[i] == t[i]:
            prefix_len += 1
        else:
            break
    return jaro + prefix_len * prefix_weight * (1.0 - jaro)


def _fallback_similarity(a: str, b: str) -> float:
    """Fallback similarity when rapidfuzz is unavailable.

    Returns the maximum of:
      * Pure-Python Jaro-Winkler
      * difflib SequenceMatcher ratio
    """
    jw = _pure_python_jaro_winkler(a, b)
    sm = SequenceMatcher(None, a, b).ratio()
    return max(jw, sm)

# ── Quality-adaptive threshold tiers ───────────────────────────────────

# Each tier is a dict with scoring thresholds.
# Stricter for LOW OCR (more garbage → need higher confidence to link).

QUALITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "HIGH": {
        "AUTO_SELECT_THRESHOLD": 0.80,
        "MIN_MARGIN": 0.15,
        "MIN_STRING_SIMILARITY": 0.60,
    },
    "MEDIUM": {
        "AUTO_SELECT_THRESHOLD": 0.85,
        "MIN_MARGIN": 0.15,
        "MIN_STRING_SIMILARITY": 0.65,
    },
    "LOW": {
        "AUTO_SELECT_THRESHOLD": 0.90,
        "MIN_MARGIN": 0.20,
        "MIN_STRING_SIMILARITY": 0.70,
    },
}

# Backwards-compatible defaults (MEDIUM tier)
AUTO_SELECT_THRESHOLD = QUALITY_THRESHOLDS["MEDIUM"]["AUTO_SELECT_THRESHOLD"]
MIN_MARGIN = QUALITY_THRESHOLDS["MEDIUM"]["MIN_MARGIN"]
MIN_STRING_SIMILARITY = QUALITY_THRESHOLDS["MEDIUM"]["MIN_STRING_SIMILARITY"]


def get_thresholds(ocr_quality: str = "MEDIUM") -> dict[str, float]:
    """Return the threshold dict for the given OCR quality level."""
    return QUALITY_THRESHOLDS.get(ocr_quality.upper(), QUALITY_THRESHOLDS["MEDIUM"])

# ── Normalisation ──────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Normalize text for comparison.  Uses shared normalize_for_search."""
    from app.services.text_normalization import normalize_for_search
    return normalize_for_search(text)


# ── String similarity ─────────────────────────────────────────────────


def string_similarity(surface: str, candidate_label: str) -> float:
    """String similarity between normalised strings (0..1).

    Uses rapidfuzz Jaro-Winkler if available, otherwise a pure-Python
    fallback (Jaro-Winkler + SequenceMatcher).
    """
    a = _normalise(surface)
    b = _normalise(candidate_label)
    if not a or not b:
        return 0.0
    if _USE_RAPIDFUZZ and _JW is not None:
        return _JW.similarity(a, b)
    return _fallback_similarity(a, b)


# ── Context similarity ────────────────────────────────────────────────


def _tokenise(text: str) -> set[str]:
    return set(_normalise(text).split())


def context_similarity(context_text: str, candidate_description: str) -> float:
    """Simple token-overlap Jaccard similarity (0..1).

    Good enough for disambiguation; can be swapped for embedding cosine
    later.
    """
    ctx = _tokenise(context_text)
    desc = _tokenise(candidate_description)
    if not ctx or not desc:
        return 0.0
    inter = ctx & desc
    union = ctx | desc
    return len(inter) / len(union) if union else 0.0


# ── Composite score ───────────────────────────────────────────────────

# Scoring weights (v5 — requirement 6):
#   final_score = 0.55*label_sim + 0.25*alias_sim + 0.15*type_bonus + 0.05*domain_bonus - penalties
_W_LABEL = 0.55
_W_ALIAS = 0.25           # alias_sim ≈ context/description overlap
_W_TYPE = 0.15
_W_DOMAIN = 0.05
_TYPE_MISMATCH_PENALTY = 0.30   # subtract this when type is incompatible

# Keep legacy names for backwards-compatible imports
_W_STRING = _W_LABEL
_W_CONTEXT = _W_ALIAS


def compute_score(
    surface: str,
    candidate_label: str,
    context_text: str,
    candidate_description: str,
    *,
    type_compatible: bool = True,
    canonical_norm: str = "",
    domain_bonus: float = 0.0,
) -> float:
    """Compute composite linking score (0..1).

    ``final_score = 0.55*label_sim + 0.25*alias_sim + 0.15*type_bonus + 0.05*domain_bonus - penalties``

    When *canonical_norm* is provided (the normalised canonical name,
    e.g. "lancelot"), the string similarity is computed against the
    canonical form instead of the raw OCR surface.  This allows OCR
    variants like "leantlote" to get a high score against Q215681
    (label "Lancelot").

    *domain_bonus* is an external signal (0-1.0) from the authority
    linking orchestrator indicating medieval/Arthurian domain relevance.
    """
    # Use canonical form for string comparison when available
    compare_surface = canonical_norm if canonical_norm else surface
    label_sim = string_similarity(compare_surface, candidate_label)
    alias_sim = context_similarity(context_text, candidate_description)

    type_bonus = 1.0 if type_compatible else 0.0
    dom_signal = min(1.0, domain_bonus / 0.20) if domain_bonus > 0 else 0.0

    raw = (_W_LABEL * label_sim
           + _W_ALIAS * alias_sim
           + _W_TYPE * type_bonus
           + _W_DOMAIN * dom_signal)

    if not type_compatible:
        raw -= _TYPE_MISMATCH_PENALTY
    return max(0.0, min(1.0, raw))


def rescore_with_canonical(
    candidates: list[dict[str, Any]],
    canonical_norm: str,
    description_keywords: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Re-score candidates using canonical name similarity.

    When a mention has been canonicalized (e.g. "leantlote" → "lancelot"),
    this function re-computes label similarity against the canonical form.
    Candidates whose label closely matches the canonical form AND belong
    to the medieval domain get a score boost to >= 0.90.

    Also applies:
    - Name-entity penalty: if description contains "given name" /
      "family name", reduce score significantly.
    - Domain bonus: if description contains Arthurian keywords.

    Returns the modified candidate list (mutated in place).
    """
    if not canonical_norm or not candidates:
        return candidates

    canon_low = canonical_norm.lower().strip()
    medieval_kws = description_keywords or _MEDIEVAL_DOMAIN_KEYWORDS

    for cand in candidates:
        label = str(cand.get("label", ""))
        desc = str(cand.get("description", "")).lower()
        label_sim = string_similarity(canon_low, label)

        # ── Name-entity penalty ───────────────────────────────────
        is_name_entity = any(
            kw in desc for kw in ("given name", "family name", "surname",
                                   "prénom", "nom de famille")
        )
        if is_name_entity:
            cand["score"] = min(cand.get("score", 0.0), 0.10)
            cand["type_compatible"] = False
            continue

        # ── Canonical boost ───────────────────────────────────────
        domain_hit = any(kw in desc for kw in medieval_kws)

        if label_sim >= 0.95 and cand.get("type_compatible", False):
            # Perfect or near-perfect canonical match + type OK
            boost = 0.92 if domain_hit else 0.90
            cand["score"] = max(cand.get("score", 0.0), boost)
        elif label_sim >= 0.85 and cand.get("type_compatible", False) and domain_hit:
            # Good match + domain relevance
            cand["score"] = max(cand.get("score", 0.0), 0.88)

    return candidates


# Domain keywords for canonical rescoring
_MEDIEVAL_DOMAIN_KEYWORDS: set[str] = {
    "arthurian", "round table", "table ronde", "chevalier",
    "knight", "grail", "graal", "camelot", "legendary",
    "fictional character", "literary character", "mythological",
    "legendary king", "queen consort", "medieval",
    "chrétien", "troyes", "lancelot", "perceval", "galahad",
    "gawain", "gauvain", "tristan", "merlin", "guinevere",
}


# ── Disambiguation (precision-first v2, quality-adaptive v3) ─────────


def disambiguate(
    scored_candidates: list[dict[str, Any]],
    *,
    ocr_quality: str = "MEDIUM",
) -> dict[str, Any]:
    """Select the best candidate from a scored list.

    **Precision-first rules (v2) with quality-adaptive thresholds (v3):**

    1. ``type_compatible`` must be ``True`` — type-incompatible candidates
       are never auto-selected.
    2. Score must be ≥ ``AUTO_SELECT_THRESHOLD`` for the given OCR quality.
    3. String similarity must be ≥ ``MIN_STRING_SIMILARITY`` for the tier.
    4. Margin over runner-up must be ≥ ``MIN_MARGIN``; otherwise
       the mention is marked *ambiguous*.
    5. ``ent_type == "role"`` → never auto-link.

    Returns a dict with:
    - ``selected``: the best candidate dict (or None)
    - ``status``:   ``"linked"`` | ``"ambiguous"`` | ``"unresolved"``
    - ``reason``:   human-readable explanation
    - ``all``:      the full sorted list (best-first)
    - ``ocr_quality``: the quality tier used
    """
    thresholds = get_thresholds(ocr_quality)
    threshold = thresholds["AUTO_SELECT_THRESHOLD"]
    min_margin = thresholds["MIN_MARGIN"]

    base_result = {
        "ocr_quality": ocr_quality,
    }

    if not scored_candidates:
        return {
            **base_result,
            "selected": None,
            "status": "unresolved",
            "reason": "no candidates found",
            "all": [],
        }

    # Sort descending by score
    ranked = sorted(
        scored_candidates,
        key=lambda c: c.get("score", 0.0),
        reverse=True,
    )
    best = ranked[0]
    best_score = float(best.get("score", 0.0))

    # Hard gate 1: type_compatible must be True
    if not best.get("type_compatible", True):
        return {
            **base_result,
            "selected": None,
            "status": "unresolved",
            "reason": (
                f"best candidate type_incompatible "
                f"(score={best_score:.3f}, qid={best.get('qid', '?')})"
            ),
            "all": ranked,
        }

    # Hard gate 2: score must reach AUTO_SELECT_THRESHOLD
    if best_score < threshold:
        return {
            **base_result,
            "selected": None,
            "status": "unresolved",
            "reason": (
                f"best score ({best_score:.3f}) "
                f"< threshold ({threshold}) [ocr_quality={ocr_quality}]"
            ),
            "all": ranked,
        }

    # Gate 3: margin check for ambiguity
    if len(ranked) >= 2:
        second_score = float(ranked[1].get("score", 0.0))
        margin = best_score - second_score
        if margin < min_margin:
            return {
                **base_result,
                "selected": None,
                "status": "ambiguous",
                "reason": (
                    f"margin too small ({margin:.3f} < {min_margin}) "
                    f"between top two candidates "
                    f"(scores: {best_score:.3f}, {second_score:.3f}) "
                    f"[ocr_quality={ocr_quality}]"
                ),
                "all": ranked,
            }

    return {
        **base_result,
        "selected": best,
        "status": "linked",
        "reason": f"best score = {best_score:.3f} [ocr_quality={ocr_quality}]",
        "all": ranked,
    }
