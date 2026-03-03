"""Language-agnostic OCR quality engine.

Computes per-token and per-line quality signals that work across scripts
(Latin, Arabic, CJK, mixed, unknown).  No dictionary lookups — only
statistical / character-level heuristics.

Quality labels:
  HIGH        — reliable for token-based downstream (NER, word search)
  OK          — usable with caveats; may miss some entities
  RISKY       — token search unreliable; use with fallback
  UNRELIABLE  — block token-based downstream; use vision/shape fallback

Hard gates:
  If quality_label in {RISKY, UNRELIABLE}:
    - Token-based NER/word-search MUST be disabled or run in fallback mode
    - Ligature candidate mining switches to shape/layout-driven path
  If leading_fragment_ratio >= LEADING_FRAG_HARD_LIMIT:
    - Force seam-aware retry before accepting result

This module is stateless.  All functions take strings / lists and return
dicts / labels.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from app.services.ocr_quality_config import (
    CROSS_PASS_STABILITY_MIN,
    ENTROPY_HIGH_LIMIT,
    ENTROPY_LOW_LIMIT,
    GIBBERISH_HARD_LIMIT,
    GIBBERISH_SOFT_LIMIT,
    LEADING_FRAG_HARD_LIMIT,
    MENTION_ABSOLUTE_MIN,
    MENTION_MIN_PER_1K_CHARS,
    NWL_TOKEN_HARD_LIMIT,
    NON_WORDLIKE_GATE_LIMIT,
    SEAM_FRAG_HARD_LIMIT,
    UNCERTAINTY_HARD_LIMIT,
    UNCERTAINTY_RISKY_LIMIT,
    frag_gate_value,
)


# ═══════════════════════════════════════════════════════════════════════
# Script detection
# ═══════════════════════════════════════════════════════════════════════

_SCRIPT_RANGES: list[tuple[str, int, int]] = [
    ("latin",     0x0000, 0x024F),
    ("latin",     0x1E00, 0x1EFF),   # Latin Extended Additional
    ("greek",     0x0370, 0x03FF),
    ("cyrillic",  0x0400, 0x04FF),
    ("arabic",    0x0600, 0x06FF),
    ("arabic",    0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
    ("hebrew",    0x0590, 0x05FF),
    ("cjk",       0x4E00, 0x9FFF),   # CJK Unified
    ("cjk",       0x3400, 0x4DBF),   # CJK Extension A
    ("cjk",       0x3040, 0x309F),   # Hiragana
    ("cjk",       0x30A0, 0x30FF),   # Katakana
    ("cjk",       0xAC00, 0xD7AF),   # Hangul
    ("devanagari", 0x0900, 0x097F),
    ("thai",      0x0E00, 0x0E7F),
    ("georgian",  0x10A0, 0x10FF),
    ("armenian",  0x0530, 0x058F),
    ("ethiopic",  0x1200, 0x137F),
]


def detect_script_family(text: str) -> str:
    """Classify the dominant script family in *text*.

    Returns one of: latin, greek, cyrillic, arabic, hebrew, cjk,
    devanagari, thai, georgian, armenian, ethiopic, mixed, unknown.
    """
    if not text or not text.strip():
        return "unknown"

    counts: Counter[str] = Counter()
    total_letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total_letters += 1
        cp = ord(ch)
        matched = False
        for script, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[script] += 1
                matched = True
                break
        if not matched:
            counts["other"] += 1

    if total_letters == 0:
        return "unknown"

    # Dominant script = highest count
    top_script, top_count = counts.most_common(1)[0]
    ratio = top_count / total_letters
    if ratio >= 0.80:
        return top_script
    if len(counts) >= 2:
        return "mixed"
    return top_script


# ═══════════════════════════════════════════════════════════════════════
# Character-level entropy
# ═══════════════════════════════════════════════════════════════════════

def char_entropy(text: str) -> float:
    """Shannon entropy (bits) of character distribution.

    Low entropy (< 2.5) → repetitive / few unique chars → suspicious.
    Very high entropy (> 5.5) → random-looking → suspicious.
    Normal text: 3.0–5.0.
    """
    if not text:
        return 0.0
    counts = Counter(text.lower())
    total = len(text)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
        if c > 0
    )


# ═══════════════════════════════════════════════════════════════════════
# Token-length distribution
# ═══════════════════════════════════════════════════════════════════════

def token_length_stats(tokens: Sequence[str]) -> dict[str, float]:
    """Compute token-length distribution statistics.

    Returns mean, stddev, fraction of very short (<3) and very long (>15).
    """
    if not tokens:
        return {"mean": 0.0, "stddev": 0.0, "very_short_frac": 1.0, "very_long_frac": 0.0}

    lengths = [len(t) for t in tokens]
    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((l - mean) ** 2 for l in lengths) / max(n, 1)
    stddev = variance ** 0.5

    very_short = sum(1 for l in lengths if l < 3)
    very_long = sum(1 for l in lengths if l > 15)

    return {
        "mean": round(mean, 3),
        "stddev": round(stddev, 3),
        "very_short_frac": round(very_short / n, 3),
        "very_long_frac": round(very_long / n, 3),
    }


# ═══════════════════════════════════════════════════════════════════════
# Vowel / consonant heuristics (script-aware)
# ═══════════════════════════════════════════════════════════════════════

# Vowel sets per script family  (configurable)
_VOWELS_LATIN = set("aeiouyàáâãäåæèéêëìíîïòóôõöùúûüýÿœ")
_VOWELS_GREEK = set("αεηιουωάέήίόύώ")
_VOWELS_CYRILLIC = set("аеёиоуыэюя")

_VOWEL_SETS: dict[str, set[str]] = {
    "latin": _VOWELS_LATIN,
    "greek": _VOWELS_GREEK,
    "cyrillic": _VOWELS_CYRILLIC,
}


def vowel_ratio(token: str, script: str = "latin") -> float:
    """Fraction of alphabetic chars that are vowels (0..1).

    Uses script-specific vowel sets.  For unknown/CJK/Arabic scripts,
    returns -1.0 (not applicable).
    """
    vowels = _VOWEL_SETS.get(script)
    if vowels is None:
        return -1.0  # not applicable for this script
    letters = [ch for ch in token.lower() if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch in vowels) / len(letters)


def token_vowel_anomaly(token: str, script: str = "latin") -> bool:
    """True if token has anomalous vowel ratio for its script.

    Latin: normal 0.20–0.70.  Outside → anomaly.
    """
    vr = vowel_ratio(token, script)
    if vr < 0:
        return False  # not applicable
    if script in ("latin", "greek", "cyrillic"):
        return vr < 0.15 or vr > 0.75
    return False


# ═══════════════════════════════════════════════════════════════════════
# Rare bigram / improbable char transitions
# ═══════════════════════════════════════════════════════════════════════

# For Latin script: bigrams that almost never occur in real words
_RARE_LATIN_BIGRAMS = frozenset({
    "zx", "xz", "qx", "xq", "jx", "xj", "zq", "qz",
    "bx", "xb", "vx", "xv", "kx", "xk", "wx", "xw",
    "jq", "qj", "jz", "zj", "bk", "kb", "fq", "qf",
    "vq", "qv", "zz", "xx", "qq",
})


def rare_bigram_ratio(text: str, script: str = "latin") -> float:
    """Fraction of character bigrams that are "rare" for the script.

    Only implemented for Latin script.  Returns 0.0 for other scripts.
    """
    if script != "latin":
        return 0.0
    low = text.lower()
    # Only consider alpha bigrams
    bigrams = [low[i:i+2] for i in range(len(low) - 1) if low[i].isalpha() and low[i+1].isalpha()]
    if not bigrams:
        return 0.0
    rare = sum(1 for bg in bigrams if bg in _RARE_LATIN_BIGRAMS)
    return rare / len(bigrams)


# ═══════════════════════════════════════════════════════════════════════
# Non-wordlike token scoring (character-level)
# ═══════════════════════════════════════════════════════════════════════

def non_wordlike_score(token: str, script: str = "latin") -> float:
    """Score how "non-wordlike" a token is (0 = normal, 1 = gibberish).

    Combines:
      - vowel anomaly
      - max consonant run (for Latin-like scripts)
      - digit/symbol mixing
      - rare bigram presence
    """
    if not token or len(token) < 2:
        return 0.5

    score = 0.0
    low = token.lower()
    alpha_chars = [ch for ch in low if ch.isalpha()]
    alpha_count = len(alpha_chars)

    # Non-alpha mixing penalty
    if alpha_count > 0:
        non_alpha_ratio = (len(low) - alpha_count) / len(low)
        score += min(0.3, non_alpha_ratio)

    # Vowel anomaly
    if script in _VOWEL_SETS:
        vr = vowel_ratio(low, script)
        if vr >= 0:
            if vr < 0.12:
                score += 0.30
            elif vr < 0.18:
                score += 0.15
            elif vr > 0.80:
                score += 0.20

    # Consonant cluster (Latin/Greek/Cyrillic only)
    if script in ("latin", "greek", "cyrillic"):
        vowels = _VOWEL_SETS.get(script, set())
        max_run = 0
        run = 0
        for ch in low:
            if ch.isalpha() and ch not in vowels:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run >= 5:
            score += 0.35
        elif max_run >= 4:
            score += 0.15

    # Rare bigrams
    if script == "latin" and alpha_count >= 4:
        rbr = rare_bigram_ratio(low, script)
        score += rbr * 0.5

    return min(1.0, score)


# ═══════════════════════════════════════════════════════════════════════
# Leading / trailing fragment ratios
# ═══════════════════════════════════════════════════════════════════════

# Common 1-2 letter function words per script that should NOT be counted
# as seam fragments.  Lowercase only.
_FUNCTION_WORDS_LATIN: frozenset[str] = frozenset({
    # French / Old French
    "a", "à", "au", "de", "du", "en", "et", "il", "je", "la", "le", "li",
    "ne", "ni", "nu", "on", "ou", "où", "si", "tu", "un",
    # Latin
    "ab", "ad", "an", "at", "ex", "in", "is", "it", "id", "ob", "se", "ut",
    # Spanish / Italian / Portuguese
    "al", "da", "di", "do", "el", "eu", "ha", "ho", "lo", "mi", "no", "os",
    "su", "te", "ti", "un", "va", "vi", "yo",
    # English
    "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is",
    "it", "me", "my", "no", "of", "ok", "on", "or", "so", "to", "up", "us",
    "we",
    # German
    "ob", "um", "zu",
    # Single vowels that are legitimate words
    "a", "e", "i", "o", "u", "y",
})

_FUNCTION_WORDS_GREEK: frozenset[str] = frozenset({
    "αν", "ας", "γη", "δε", "εν", "θα", "κι", "με", "να", "ου", "σε", "ωσ",
})

_FUNCTION_WORDS_CYRILLIC: frozenset[str] = frozenset({
    "а", "в", "и", "к", "о", "с", "у", "я", "да", "до", "за", "из", "на",
    "не", "ни", "но", "ну", "об", "от", "по", "то",
})

_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "latin": _FUNCTION_WORDS_LATIN,
    "greek": _FUNCTION_WORDS_GREEK,
    "cyrillic": _FUNCTION_WORDS_CYRILLIC,
}


def leading_fragment_ratio(lines: Sequence[str], script: str = "latin") -> float:
    """Fraction of lines starting with a partial-word fragment.

    Improved heuristic that avoids false-positives on legitimate short
    function words that commonly appear at line starts in medieval verse
    and prose (e.g. "de", "et", "li", "si", "en").

    True fragment indicators:
      - Single non-letter character at line start
      - Single consonant (not a known function word) at line start
      - 1-2 char lowercase token that is NOT in the script's function-word list
      - Line starts with a continuation marker (e.g. hyphen leftover)
    """
    if not lines:
        return 0.0

    func_words = _FUNCTION_WORDS.get(script, frozenset())
    fragments = 0
    total = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        total += 1
        tokens = stripped.split()
        first_word = tokens[0] if tokens else ""
        if not first_word:
            continue

        is_fragment = False

        # Continuation marker: line starts with hyphen or dash
        if first_word[0] in "-–—":
            is_fragment = True
        elif script in ("latin", "greek", "cyrillic"):
            lw = first_word.lower()
            # Skip known function words — they are NOT fragments
            if lw in func_words:
                continue
            # Single non-vowel letter that isn't a function word
            if len(first_word) == 1 and first_word.isalpha():
                vowels = _VOWEL_SETS.get(script, set())
                if first_word.lower() not in vowels:
                    is_fragment = True
            # 2-char lowercase token not in function words → likely fragment
            elif len(first_word) == 2 and first_word.isalpha() and not first_word[0].isupper():
                # Already checked it's not in func_words above
                is_fragment = True
        else:
            # For non-Latin scripts: only single non-alpha chars
            if len(first_word) == 1 and not first_word.isalpha():
                is_fragment = True

        if is_fragment:
            fragments += 1

    return fragments / max(total, 1)


def seam_fragment_ratio(
    lines: Sequence[str],
    seam_line_indices: Sequence[int] | None = None,
    script: str = "latin",
) -> float:
    """Fraction of seam-adjacent lines that show fragmentation.

    If *seam_line_indices* are provided (line numbers where tile seams fall),
    only those lines are checked — giving a geometry-aware detection.
    If not provided, all lines are checked (falling back to heuristic).
    A fragment at a seam boundary is a much stronger signal than a generic
    short token at an arbitrary line start.
    """
    if not lines:
        return 0.0

    func_words = _FUNCTION_WORDS.get(script, frozenset())

    # Determine which lines to check
    if seam_line_indices is not None and seam_line_indices:
        check_indices = set()
        for si in seam_line_indices:
            # Check the seam line and its immediate neighbours
            for offset in (-1, 0, 1):
                idx = si + offset
                if 0 <= idx < len(lines):
                    check_indices.add(idx)
    else:
        check_indices = set(range(len(lines)))

    if not check_indices:
        return 0.0

    fragments = 0
    checked = 0
    for idx in sorted(check_indices):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        checked += 1
        tokens = stripped.split()
        first = tokens[0] if tokens else ""
        last = tokens[-1] if tokens else ""

        has_leading = False
        has_trailing = False

        # Leading fragment check
        if first and first[0] in "-–—":
            has_leading = True
        elif first and script in ("latin", "greek", "cyrillic"):
            lw = first.lower()
            if lw not in func_words:
                if (len(first) == 1 and first.isalpha()
                        and first.lower() not in _VOWEL_SETS.get(script, set())):
                    has_leading = True
                elif len(first) == 2 and first.isalpha() and not first[0].isupper():
                    has_leading = True

        # Trailing fragment check (broken word at end)
        if last and last[-1] == "-":
            has_trailing = True
        elif last and len(last) == 1 and last.isalpha():
            has_trailing = True

        if has_leading or has_trailing:
            fragments += 1

    return fragments / max(checked, 1)


def trailing_fragment_ratio(lines: Sequence[str]) -> float:
    """Fraction of lines ending with a likely broken-off word.

    Indicators: line ends with a hyphen, or a single consonant.
    """
    if not lines:
        return 0.0

    fragments = 0
    total = 0
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        total += 1
        if stripped.endswith("-"):
            fragments += 1
        elif len(stripped) >= 2:
            last_word = stripped.split()[-1] if stripped.split() else ""
            if len(last_word) == 1 and last_word.isalpha() and last_word.lower() not in _VOWELS_LATIN:
                fragments += 1

    return fragments / max(total, 1)


# ═══════════════════════════════════════════════════════════════════════
# Cross-pass stability scoring
# ═══════════════════════════════════════════════════════════════════════

def token_stability_score(
    pass1_tokens: Sequence[str],
    pass2_tokens: Sequence[str],
) -> float:
    """Measure how stable tokens are across two OCR passes (0..1).

    1.0 = identical token sets.
    0.0 = completely different.

    Uses Jaccard similarity of token sets (case-insensitive).
    """
    if not pass1_tokens and not pass2_tokens:
        return 1.0
    s1 = {t.lower() for t in pass1_tokens if t.strip()}
    s2 = {t.lower() for t in pass2_tokens if t.strip()}
    if not s1 and not s2:
        return 1.0
    union = s1 | s2
    if not union:
        return 1.0
    return len(s1 & s2) / len(union)


def normalized_levenshtein_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity between two strings (0..1).

    Uses standard DP edit-distance.  1.0 = identical, 0.0 = completely
    different.  Normalised by the length of the longer string.
    """
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    # Standard two-row DP
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def compute_cross_pass_stability(text_a: str, text_b: str) -> float:
    """Robust cross-pass stability combining Jaccard + normalised Levenshtein.

    - Normalize whitespace before comparison.
    - Average of line-sequence Levenshtein similarity and token-level Jaccard.
    - Returns 0..1 (1.0 = identical).
    """
    def _norm(t: str) -> str:
        return re.sub(r"\\s+", " ", t.strip().lower())

    na, nb = _norm(text_a), _norm(text_b)
    if na == nb:
        return 1.0
    if not na or not nb:
        return 0.0

    # Line-level normalised Levenshtein
    lev_sim = normalized_levenshtein_similarity(na, nb)

    # Token-level Jaccard
    tok_a = na.split()
    tok_b = nb.split()
    jaccard = token_stability_score(tok_a, tok_b)

    return round((lev_sim + jaccard) / 2.0, 4)


# ═══════════════════════════════════════════════════════════════════════
# Uncertainty marker density
# ═══════════════════════════════════════════════════════════════════════

def uncertainty_density(text: str) -> float:
    """Fraction of characters that are uncertainty markers (?, …, [...]).

    High values (> 0.10) indicate very low confidence OCR.
    """
    if not text:
        return 0.0
    markers = text.count("?") + text.count("…") + text.count("[…]") * 3
    return markers / max(len(text), 1)


# ═══════════════════════════════════════════════════════════════════════
# Gibberish detection (composite score)
# ═══════════════════════════════════════════════════════════════════════

def gibberish_score(text: str, script: str = "latin") -> float:
    """Composite gibberish detector (0 = clean, 1 = total gibberish).

    Language-agnostic: uses character-level signals only.
    """
    if not text or not text.strip():
        return 0.0

    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if not tokens:
        return 0.0

    # 1. Non-wordlike token fraction
    nwl_scores = [non_wordlike_score(t, script) for t in tokens if len(t) >= 3]
    nwl_frac = (
        sum(1 for s in nwl_scores if s > 0.45) / max(len(nwl_scores), 1)
        if nwl_scores else 0.0
    )

    # 2. Entropy anomaly
    ent = char_entropy(text)
    ent_penalty = 0.0
    if ent < 2.0:
        ent_penalty = 0.3
    elif ent > 5.5:
        ent_penalty = 0.2

    # 3. Rare bigram load
    rbr = rare_bigram_ratio(text, script)

    # 4. Uncertainty marker density
    unc = uncertainty_density(text)

    # Weighted combination
    score = (
        0.40 * nwl_frac
        + 0.20 * ent_penalty
        + 0.20 * min(1.0, rbr * 5)
        + 0.20 * min(1.0, unc * 5)
    )
    return round(min(1.0, score), 4)


# ═══════════════════════════════════════════════════════════════════════
# Per-token quality report
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TokenQuality:
    """Quality assessment for a single token."""
    token: str
    index: int
    non_wordlike: float = 0.0
    vowel_ratio: float = 0.0
    rare_bigram: bool = False
    is_anomalous: bool = False


@dataclass
class LineQuality:
    """Quality assessment for a single line."""
    line_idx: int
    text: str
    token_count: int = 0
    gibberish_score: float = 0.0
    non_wordlike_frac: float = 0.0
    is_fragment_start: bool = False
    is_fragment_end: bool = False


# ═══════════════════════════════════════════════════════════════════════
# OCR Quality Report (main output)
# ═══════════════════════════════════════════════════════════════════════

# ── backward-compat aliases (thresholds now live in ocr_quality_config) ──
# Re-exported so that ``from ocr_quality import LEADING_FRAG_HARD_LIMIT``
# still works without touching every importer.
# The canonical values live in ``ocr_quality_config``.
CROSS_PASS_INSTABILITY_LIMIT = CROSS_PASS_STABILITY_MIN  # alias


@dataclass
class OCRQualityReport:
    """Structured OCR quality assessment for a single run/pass.

    Stored in DB (as JSON) for auditing and gating decisions.
    """
    # Identification
    run_id: str = ""
    pass_idx: int = 0    # 0=baseline, 1=overlap, 2=high-recall

    # Script detection
    script_family: str = "unknown"

    # Global signals
    char_entropy: float = 0.0
    gibberish_score: float = 0.0
    uncertainty_density: float = 0.0
    rare_bigram_ratio: float = 0.0

    # Token-level stats
    token_count: int = 0
    non_wordlike_frac: float = 0.0
    token_length_mean: float = 0.0
    token_length_stddev: float = 0.0
    very_short_token_frac: float = 0.0
    very_long_token_frac: float = 0.0

    # Fragment stats
    leading_fragment_ratio: float = 0.0
    seam_fragment_ratio: float = 0.0
    trailing_fragment_ratio: float = 0.0
    line_count: int = 0

    # Cross-pass stability (populated only after multi-pass)
    cross_pass_stability: float = -1.0   # -1 = not computed yet

    # Sanitization
    sanitized_token_count: int = 0

    # Derived label
    quality_label: str = "OK"           # HIGH | OK | RISKY | UNRELIABLE

    # Gate decisions
    token_search_allowed: bool = True
    ner_allowed: bool = True
    seam_retry_required: bool = False

    # Per-line detail (optional, for DB storage)
    line_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Don't serialize massive line_details to top-level
        if len(d.get("line_details", [])) > 100:
            d["line_details"] = d["line_details"][:100]
            d["line_details_truncated"] = True
        return d


def compute_quality_report(
    text: str,
    *,
    run_id: str = "",
    pass_idx: int = 0,
    previous_pass_tokens: Sequence[str] | None = None,
) -> OCRQualityReport:
    """Compute a full OCR quality report for a text.

    This is the main entry point for quality assessment.

    Args:
        text: The OCR output text.
        run_id: Pipeline run identifier.
        pass_idx: Which OCR pass (0=baseline, 1=overlap, 2=high-recall).
        previous_pass_tokens: Tokens from a previous pass for stability scoring.

    Returns:
        OCRQualityReport with all signals computed and quality_label set.
    """
    report = OCRQualityReport(run_id=run_id, pass_idx=pass_idx)

    if not text or not text.strip():
        report.quality_label = "UNRELIABLE"
        report.token_search_allowed = False
        report.ner_allowed = False
        return report

    # ── Script detection ───────────────────────────────────────────
    report.script_family = detect_script_family(text)

    # ── Tokenize ───────────────────────────────────────────────────
    lines = [ln for ln in text.splitlines() if ln.strip()]
    report.line_count = len(lines)

    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    report.token_count = len(tokens)

    # ── Character entropy ──────────────────────────────────────────
    report.char_entropy = round(char_entropy(text), 4)

    # ── Token length distribution ──────────────────────────────────
    tl_stats = token_length_stats(tokens)
    report.token_length_mean = tl_stats["mean"]
    report.token_length_stddev = tl_stats["stddev"]
    report.very_short_token_frac = tl_stats["very_short_frac"]
    report.very_long_token_frac = tl_stats["very_long_frac"]

    # ── Non-wordlike token fraction ────────────────────────────────
    if tokens:
        nwl_scores = [non_wordlike_score(t, report.script_family) for t in tokens if len(t) >= 3]
        if nwl_scores:
            report.non_wordlike_frac = round(
                sum(1 for s in nwl_scores if s > 0.45) / len(nwl_scores), 4
            )

    # ── Rare bigram ratio ──────────────────────────────────────────
    report.rare_bigram_ratio = round(rare_bigram_ratio(text, report.script_family), 4)

    # ── Gibberish score ────────────────────────────────────────────
    report.gibberish_score = gibberish_score(text, report.script_family)

    # ── Uncertainty density ────────────────────────────────────────
    report.uncertainty_density = round(uncertainty_density(text), 4)

    # ── Fragment ratios ────────────────────────────────────────────
    report.leading_fragment_ratio = round(
        leading_fragment_ratio(lines, report.script_family), 4
    )
    report.seam_fragment_ratio = round(
        seam_fragment_ratio(lines, script=report.script_family), 4
    )
    report.trailing_fragment_ratio = round(trailing_fragment_ratio(lines), 4)

    # ── Cross-pass stability ───────────────────────────────────────
    if previous_pass_tokens is not None:
        report.cross_pass_stability = round(
            token_stability_score(previous_pass_tokens, tokens), 4
        )

    # ── Sanitized token count ──────────────────────────────────────
    report.sanitized_token_count = sum(
        1 for t in tokens
        if "?" in t or "…" in t or "[" in t
    )

    # ── Per-line details ───────────────────────────────────────────
    lfr_lines = leading_fragment_ratio(lines, report.script_family)
    for idx, line in enumerate(lines):
        line_tokens = [t for t in re.split(r"\s+", line.strip()) if t]
        line_gs = gibberish_score(line, report.script_family) if len(line.strip()) >= 5 else 0.0
        nwl_line = 0.0
        if line_tokens:
            nwl_line_scores = [non_wordlike_score(t, report.script_family) for t in line_tokens if len(t) >= 3]
            if nwl_line_scores:
                nwl_line = sum(1 for s in nwl_line_scores if s > 0.45) / len(nwl_line_scores)

        report.line_details.append({
            "line_idx": idx,
            "text": line.strip()[:120],
            "token_count": len(line_tokens),
            "gibberish_score": round(line_gs, 4),
            "non_wordlike_frac": round(nwl_line, 4),
        })

    # ── Derive quality label ───────────────────────────────────────
    report.quality_label = _derive_quality_label(report)

    # ── Gate decisions ─────────────────────────────────────────────
    report.token_search_allowed = report.quality_label in ("HIGH", "OK")
    report.ner_allowed = report.quality_label in ("HIGH", "OK")
    # Seam retry: prefer geometry-aware seam_fragment_ratio; fall back to
    # leading_fragment_ratio only when seam_fragment_ratio is unavailable.
    report.seam_retry_required = (
        report.seam_fragment_ratio >= SEAM_FRAG_HARD_LIMIT
        or report.leading_fragment_ratio >= LEADING_FRAG_HARD_LIMIT
    )

    return report


def _derive_quality_label(r: OCRQualityReport) -> str:
    """Derive quality label from computed signals.

    Hierarchy:
      UNRELIABLE → RISKY → OK → HIGH
    """
    # UNRELIABLE conditions (any one triggers)
    if r.gibberish_score >= GIBBERISH_HARD_LIMIT:
        return "UNRELIABLE"
    if r.uncertainty_density >= UNCERTAINTY_HARD_LIMIT:
        return "UNRELIABLE"
    if r.cross_pass_stability >= 0 and r.cross_pass_stability < CROSS_PASS_STABILITY_MIN:
        return "UNRELIABLE"
    if r.non_wordlike_frac >= NON_WORDLIKE_GATE_LIMIT:
        return "UNRELIABLE"
    if r.char_entropy < 1.5:
        return "UNRELIABLE"

    # RISKY conditions
    if r.gibberish_score >= GIBBERISH_SOFT_LIMIT:
        return "RISKY"
    if r.non_wordlike_frac >= NWL_TOKEN_HARD_LIMIT:
        return "RISKY"
    # Use seam_fragment_ratio (geometry-aware) for seam-related RISKY.
    # leading_fragment_ratio alone does NOT trigger RISKY — it may be
    # caused by legitimate short function words in medieval verse.
    if r.seam_fragment_ratio >= SEAM_FRAG_HARD_LIMIT:
        return "RISKY"
    if r.leading_fragment_ratio >= LEADING_FRAG_HARD_LIMIT:
        return "RISKY"
    if r.char_entropy < ENTROPY_LOW_LIMIT or r.char_entropy > ENTROPY_HIGH_LIMIT:
        return "RISKY"
    if r.uncertainty_density >= UNCERTAINTY_RISKY_LIMIT:
        return "RISKY"

    # HIGH conditions
    if (r.gibberish_score < 0.10
        and r.non_wordlike_frac < 0.15
        and r.leading_fragment_ratio < 0.06
        and r.seam_fragment_ratio < 0.05
        and r.uncertainty_density < 0.03
        and 2.5 <= r.char_entropy <= 5.0):
        return "HIGH"

    return "OK"


# ═══════════════════════════════════════════════════════════════════════
# Mention recall gating
# ═══════════════════════════════════════════════════════════════════════

def check_mention_recall(
    text: str,
    mentions_total: int,
    quality_label: str,
) -> dict[str, Any]:
    """Check if mention extraction meets minimum expectations.

    Returns:
        {
            "mention_recall_ok": bool,
            "reason": str,
            "trigger_high_recall": bool,  # should we re-run with relaxed params?
        }
    """
    if quality_label not in ("HIGH", "OK"):
        return {
            "mention_recall_ok": True,
            "reason": f"quality_label={quality_label}, mention recall check skipped",
            "trigger_high_recall": False,
        }

    text_len = len(text.strip())
    if text_len < 200:
        return {
            "mention_recall_ok": True,
            "reason": "text too short for recall check",
            "trigger_high_recall": False,
        }

    expected_min = max(
        MENTION_ABSOLUTE_MIN,
        int(text_len / 1000 * MENTION_MIN_PER_1K_CHARS),
    )

    if mentions_total >= expected_min:
        return {
            "mention_recall_ok": True,
            "reason": f"mentions={mentions_total} >= expected_min={expected_min}",
            "trigger_high_recall": False,
        }

    return {
        "mention_recall_ok": False,
        "reason": (
            f"MENTION_RECALL_LOW: mentions={mentions_total} < expected_min={expected_min} "
            f"(text_len={text_len}, quality={quality_label})"
        ),
        "trigger_high_recall": True,
    }


# ═══════════════════════════════════════════════════════════════════════
# Format for report / DB
# ═══════════════════════════════════════════════════════════════════════

def format_quality_report_summary(report: OCRQualityReport) -> str:
    """Format a quality report as a human-readable summary string."""
    L = [
        "=== OCR QUALITY REPORT ===",
        f"run_id: {report.run_id}",
        f"pass_idx: {report.pass_idx}",
        f"script_family: {report.script_family}",
        f"quality_label: {report.quality_label}",
        f"",
        f"char_entropy: {report.char_entropy:.4f}",
        f"gibberish_score: {report.gibberish_score:.4f}",
        f"non_wordlike_frac: {report.non_wordlike_frac:.4f}",
        f"rare_bigram_ratio: {report.rare_bigram_ratio:.4f}",
        f"uncertainty_density: {report.uncertainty_density:.4f}",
        f"",
        f"token_count: {report.token_count}",
        f"token_length_mean: {report.token_length_mean:.2f}",
        f"token_length_stddev: {report.token_length_stddev:.2f}",
        f"very_short_token_frac: {report.very_short_token_frac:.3f}",
        f"very_long_token_frac: {report.very_long_token_frac:.3f}",
        f"",
        f"leading_fragment_ratio: {report.leading_fragment_ratio:.4f}",
        f"seam_fragment_ratio: {report.seam_fragment_ratio:.4f}",
        f"trailing_fragment_ratio: {report.trailing_fragment_ratio:.4f}",
        f"line_count: {report.line_count}",
        f"",
        f"cross_pass_stability: {report.cross_pass_stability:.4f}",
        f"sanitized_token_count: {report.sanitized_token_count}",
        f"",
        f"=== GATE DECISIONS ===",
        f"token_search_allowed: {report.token_search_allowed}",
        f"ner_allowed: {report.ner_allowed}",
        f"seam_retry_required: {report.seam_retry_required}",
    ]
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════
# Unified EffectiveQuality — single source of truth
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EffectiveQuality:
    """Single unified quality object used by UI, DB, and gate enforcement.

    This eliminates the mismatch between v1 ``quality_label`` (sanity-based)
    and v2 ``quality_label_v2`` (hardened) by providing ONE authoritative
    source of truth.

    Fields:
      label         — "HIGH" | "OK" | "RISKY" | "UNRELIABLE"
      downstream    — "token_based" | "vision_fallback"
      confidence    — 0..1 overall OCR confidence
      ner_allowed   — can downstream NER/chunking run?
      token_search_allowed — can downstream word-search/linking run?
      seam_retry_required — should we retry with shifted tiles?
      metrics       — dict of numeric quality signals
    """
    label: str = "OK"
    downstream: str = "token_based"
    confidence: float = 0.0
    ner_allowed: bool = True
    token_search_allowed: bool = True
    seam_retry_required: bool = False
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_effective_quality(
    report: OCRQualityReport,
    gate_decisions: dict[str, Any],
    confidence: float = 0.0,
) -> EffectiveQuality:
    """Build a single EffectiveQuality from a quality report + gate decisions.

    This is the canonical way to produce the quality object that flows
    through UI, DB, and gating.
    """
    return EffectiveQuality(
        label=report.quality_label,
        downstream=gate_decisions.get("downstream_mode", "vision_fallback"),
        confidence=confidence,
        ner_allowed=gate_decisions.get("ner_allowed", report.ner_allowed),
        token_search_allowed=gate_decisions.get(
            "token_search_allowed", report.token_search_allowed,
        ),
        seam_retry_required=gate_decisions.get(
            "seam_retry_required", report.seam_retry_required,
        ),
        metrics={
            "gibberish_score": report.gibberish_score,
            "non_wordlike_frac": report.non_wordlike_frac,
            "leading_fragment_ratio": report.leading_fragment_ratio,
            "seam_fragment_ratio": report.seam_fragment_ratio,
            "char_entropy": report.char_entropy,
            "uncertainty_density": report.uncertainty_density,
            "cross_pass_stability": report.cross_pass_stability,
            "trailing_fragment_ratio": report.trailing_fragment_ratio,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# Conservative uncertainty enforcement
# ═══════════════════════════════════════════════════════════════════════

_SUSPICIOUS_HYPHEN_RE = re.compile(
    r"(?<=[a-z])(?=[A-Z])-|"          # mixed-case around hyphen
    r"-(?=[A-Z][a-z]{0,2}$)|"          # hyphen before very short suffix
    r"(?<=[A-Z])-(?=[a-z])",           # upper-lower boundary
    re.UNICODE,
)

_UNLIKELY_BIGRAMS_SET = frozenset({
    "zx", "xz", "qx", "xq", "jx", "xj", "zq", "qz",
    "bx", "xb", "vx", "xv", "kx", "xk", "wx", "xw",
    "jq", "qj", "jz", "zj", "bk", "kb", "fq", "qf",
    "vq", "qv",
})


def _token_has_unlikely_bigrams(token: str) -> bool:
    low = token.lower()
    for i in range(len(low) - 1):
        if low[i:i + 2] in _UNLIKELY_BIGRAMS_SET:
            return True
    return False


def _looks_like_hallucinated_hyphen(token: str, is_unstable: bool) -> bool:
    """Conservative: only flag when multiple hallucination signals coincide."""
    if "-" not in token:
        return False
    parts = token.split("-")
    if len(parts) != 2:
        return False
    left, right = parts
    if not left or not right:
        return False
    # Mixed case + hyphen + short suffix + instability
    has_mixed_case = (left[-1:].islower() and right[0:1].isupper()) or \
                     (left[-1:].isupper() and right[0:1].islower())
    has_unlikely = _token_has_unlikely_bigrams(left + right)
    has_short_part = len(left) <= 2 or len(right) <= 2
    enough_signals = sum([has_mixed_case, has_unlikely, has_short_part, is_unstable])
    return enough_signals >= 3


def apply_uncertainty_markers(
    text: str,
    cross_pass_text: str | None = None,
    cross_pass_stability: float = -1.0,
    frag_gate_val: float = 0.0,
    uncertainty_dens: float = 0.0,
) -> tuple[str, int]:
    """Apply conservative uncertainty markers to unstable/suspicious tokens.

    This function NEVER corrects text into plausible words — it only
    downgrades uncertain tokens to ``?`` (character) or ``[…]`` (span).

    Returns:
        (processed_text, markers_inserted_count)
    """
    from app.services.ocr_quality_config import (
        UNCERTAINTY_ENFORCEMENT_STABILITY_THRESHOLD,
        UNCERTAINTY_ENFORCEMENT_FRAG_THRESHOLD,
        UNCERTAINTY_ENFORCEMENT_DENSITY_THRESHOLD,
    )

    # Decide whether enforcement is needed
    should_enforce = False
    if 0 <= cross_pass_stability < UNCERTAINTY_ENFORCEMENT_STABILITY_THRESHOLD:
        should_enforce = True
    if frag_gate_val >= UNCERTAINTY_ENFORCEMENT_FRAG_THRESHOLD:
        should_enforce = True
    if uncertainty_dens >= UNCERTAINTY_ENFORCEMENT_DENSITY_THRESHOLD:
        should_enforce = True

    if not should_enforce or not text.strip():
        return text, 0
    if cross_pass_text is None:
        return text, 0

    # Build set of unstable tokens by comparing the two passes line-by-line
    lines_a = text.splitlines()
    lines_b = cross_pass_text.splitlines()
    # Build unstable-token set from differing lines
    unstable_tokens: set[str] = set()
    max_lines = max(len(lines_a), len(lines_b))
    for i in range(max_lines):
        la = lines_a[i].strip() if i < len(lines_a) else ""
        lb = lines_b[i].strip() if i < len(lines_b) else ""
        if la == lb:
            continue
        toks_a = set(la.split())
        toks_b = set(lb.split())
        # Tokens that appear in one pass but not the other are unstable
        unstable_tokens |= toks_a.symmetric_difference(toks_b)

    if not unstable_tokens:
        return text, 0

    # Apply markers - replace unstable tokens in-place
    markers_count = 0
    result_lines: list[str] = []
    for line in lines_a:
        tokens = line.split()
        new_tokens: list[str] = []
        for tok in tokens:
            if tok in unstable_tokens:
                # Check for hallucinated hyphen
                is_unstable = True
                if _looks_like_hallucinated_hyphen(tok, is_unstable):
                    new_tokens.append("[…]")
                    markers_count += 1
                elif _token_has_unlikely_bigrams(tok) and non_wordlike_score(tok) > 0.45:
                    # Non-wordlike + unlikely bigrams → replace chars
                    replaced = "".join(
                        "?" if c.isalpha() and non_wordlike_score(c, "latin") > 0.3 else c
                        for c in tok
                    )
                    if replaced != tok:
                        new_tokens.append(replaced)
                        markers_count += 1
                    else:
                        new_tokens.append("[…]")
                        markers_count += 1
                elif non_wordlike_score(tok) > 0.55:
                    new_tokens.append("[…]")
                    markers_count += 1
                else:
                    # Token differs across passes but isn't clearly garbage →
                    # leave it, don't hallucinate a correction
                    new_tokens.append(tok)
            else:
                new_tokens.append(tok)
        result_lines.append(" ".join(new_tokens))

    return "\n".join(result_lines), markers_count
