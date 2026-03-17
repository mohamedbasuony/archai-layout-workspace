"""Robust text normalization for medieval OCR output.

Provides Unicode normalization, OCR-confusion fixes, token quality
scoring, a unified ``normalize_for_search()`` pipeline, and a
common-noun blacklist that prevents short everyday words from
being treated as entity mentions.

All functions are stateless and operate on plain strings.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


# ── Common-noun / function-word blacklist ──────────────────────────────
# Tokens in this set must NEVER become "person" or "work" mentions.
# They are ordinary Old-/Middle-French words, modern French words, or
# Latin words commonly produced by OCR on medieval manuscripts.
# The list is intentionally broad to prevent false-positive entity
# mentions that waste Wikidata API calls.

COMMON_WORD_BLACKLIST: frozenset[str] = frozenset({
    # Modern French function words
    "a", "au", "aux", "car", "ce", "ces", "comme", "dans", "de", "des",
    "du", "en", "est", "et", "il", "ils", "je", "la", "le", "les",
    "leur", "lui", "mais", "me", "mes", "mon", "ne", "ni", "nos",
    "nous", "on", "ou", "par", "pas", "plus", "pour", "que", "qui",
    "sa", "se", "ses", "si", "son", "sur", "te", "tes", "ton", "tu",
    "un", "une", "vous",
    # Old French function words
    "al", "com", "con", "del", "d", "el", "li", "lo", "lor",
    "molt", "moult", "ot", "ont", "uns", "vont",
    # Common Old French verbs / adverbs / conjunctions
    "ala", "alerent", "aussi", "avoit", "bien", "dit", "dist",
    "donc", "dont", "estoit", "fait", "firent", "fist", "font",
    "fut", "lors", "puis", "sont", "tant", "tout", "vint",
    # Short common French/Latin nouns that OCR produces regularly
    "main", "port", "ente", "pont", "tour", "mont", "fort",
    "camp", "part", "mort", "sort", "vent", "voie", "aide",
    "arme", "cent", "droit", "lieu", "mise", "note", "page",
    "rien", "sens", "site", "terre", "temps", "voir", "vers",
    "cors", "cors", "char", "cuer", "cour", "dame", "dieu",
    # Latin common words
    "ante", "post", "inter", "super", "amen",
    # Tokens observed as false positives in run 81ae066e
    "voin", "elain", "ament",
})

# ── Unicode normalization ──────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """NFC-normalize, fold long-s (ſ) → s, collapse whitespace.

    This is the *first* step in every normalization pipeline.  It ensures
    that combining diacritics are composed and historical glyphs are
    mapped to modern equivalents.

    Also: NFKC normalization, NBSP → regular space, strip bracket
    artifacts like "[…]".
    """
    # NFKC first (compatibility decomposition + canonical composition)
    text = unicodedata.normalize("NFKC", text)
    # Then NFC for composed diacritics
    text = unicodedata.normalize("NFC", text)
    # Long-s → s
    text = text.replace("\u017f", "s")
    # NBSP + other Unicode whitespace → regular space
    text = text.replace("\u00a0", " ")   # NBSP
    text = text.replace("\u2007", " ")   # figure space
    text = text.replace("\u202f", " ")   # narrow NBSP
    # Strip bracket artifacts: […], (...), etc.
    text = re.sub(r"\[\s*…\s*\]", " ", text)
    text = re.sub(r"\[\s*\.{2,}\s*\]", " ", text)
    # Collapse whitespace (newlines → space, multiple spaces → one)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── OCR confusion fixes ───────────────────────────────────────────────

# Common medieval OCR substitution pairs (source → target).
# Applied left-to-right; order matters for overlapping patterns.
_OCR_CONFUSION_MAP: list[tuple[str, str]] = [
    # Long-s / f confusion
    ("ſ", "s"),
    # u / v interchange (common in medieval texts)
    ("vn", "un"),
    ("vne", "une"),
    # i / j / y interchange
    ("ieh", "jeh"),
    # Common ligature artifacts
    ("œ", "oe"),
    ("æ", "ae"),
    ("ﬁ", "fi"),
    ("ﬂ", "fl"),
    ("ﬀ", "ff"),
    ("ﬃ", "ffi"),
    ("ﬄ", "ffl"),
    # Accent / diacritic misreads
    ("ë", "e"),
    ("ï", "i"),
    # Typographic quotes → plain
    ("\u2018", "'"),
    ("\u2019", "'"),
    ("\u201c", '"'),
    ("\u201d", '"'),
]

# Regex-based replacements for common OCR patterns
_OCR_REGEX_FIXES: list[tuple[re.Pattern[str], str]] = [
    # Double-i → ii (not "u" misread)
    (re.compile(r"(?<=[a-z])ii(?=[a-z])"), "ii"),
    # Terminal -ct → -ct (common misread of -et)
    # Only remove truly garbage non-alpha chars within words
    (re.compile(r"(?<=\w)[|¦](?=\w)"), ""),
]


def ocr_confusion_fixes(text: str) -> str:
    """Apply common medieval-OCR confusion substitutions.

    These are deliberately conservative — only well-attested patterns are
    included to avoid introducing new errors.
    """
    for src, tgt in _OCR_CONFUSION_MAP:
        text = text.replace(src, tgt)
    for pat, repl in _OCR_REGEX_FIXES:
        text = pat.sub(repl, text)
    return text


# ── Token quality features ────────────────────────────────────────────

_VOWELS = set("aeiouyàâäéèêëïîôùûüœæ")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE = re.compile(r"\s+")


def _vowel_ratio(token: str) -> float:
    """Fraction of characters that are vowels (0..1)."""
    if not token:
        return 0.0
    return sum(1 for c in token.lower() if c in _VOWELS) / len(token)


def _max_consonant_run(token: str) -> int:
    """Length of the longest consecutive consonant run."""
    best = 0
    run = 0
    for c in token.lower():
        if c in _VOWELS or not c.isalpha():
            run = 0
        else:
            run += 1
            best = max(best, run)
    return best


def _is_alpha_token(token: str) -> bool:
    """True if the token is purely alphabetic (allows diacritics)."""
    return bool(token) and all(c.isalpha() for c in token)


def token_quality_score(token: str) -> float:
    """Score a single token for OCR quality (0..1).

    Factors:
      - Length in [3, 20] → bonus
      - Vowel ratio in [0.20, 0.80] → bonus
      - No consonant run ≥ 4 → bonus
      - Alphabetic → bonus

    Returns a float in [0, 1].
    """
    if not token or len(token) < 2:
        return 0.0

    score = 0.50  # base
    low = token.lower()

    # Length
    if 3 <= len(low) <= 20:
        score += 0.15
    elif len(low) < 3:
        score -= 0.20
    else:
        score -= 0.10

    # Vowel ratio
    vr = _vowel_ratio(low)
    if 0.25 <= vr <= 0.65:
        score += 0.20  # ideal
    elif 0.15 <= vr <= 0.80:
        score += 0.05
    else:
        score -= 0.25

    # Consonant clusters
    mcr = _max_consonant_run(low)
    if mcr >= 4:
        score -= 0.35
    elif mcr >= 3:
        score -= 0.05

    # Alpha
    if _is_alpha_token(low):
        score += 0.10
    else:
        score -= 0.15

    return round(max(0.0, min(1.0, score)), 3)


def text_quality_label(text: str) -> str:
    """Classify overall OCR quality of a text as HIGH / MEDIUM / LOW.

    Based on average token quality and fraction of low-quality tokens.
    """
    if not text or not text.strip():
        return "LOW"

    tokens = _PUNCT_RE.sub(" ", text)
    tokens = [t for t in _MULTI_SPACE.sub(" ", tokens).split() if len(t) >= 2]
    if not tokens:
        return "LOW"

    scores = [token_quality_score(t) for t in tokens]
    avg = sum(scores) / len(scores)
    low_frac = sum(1 for s in scores if s < 0.30) / len(scores)

    if avg >= 0.55 and low_frac < 0.15:
        return "HIGH"
    elif avg >= 0.35 and low_frac < 0.40:
        return "MEDIUM"
    else:
        return "LOW"


# ── Normalize for search ──────────────────────────────────────────────

def normalize_for_search(text: str) -> str:
    """Full normalization pipeline for Wikidata/authority search.

    Steps:
      1. Unicode NFKC+NFC normalization + long-s fold + NBSP/bracket strip
      2. OCR confusion fixes
      3. Strip diacritics (NFKD decompose + drop combining marks)
      4. Lowercase, remove punctuation, collapse whitespace
    """
    text = normalize_unicode(text)
    text = ocr_confusion_fixes(text)
    # Strip diacritics
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    # Lowercase + strip punct
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


# ── OCR-confusion-aware matching ─────────────────────────────────

# Character equivalence classes for medieval OCR confusion.
# Used ONLY for matching, never to mutate stored text.
_OCR_CONFUSION_CLASSES: dict[str, str] = {
    "i": "I", "l": "I", "1": "I",  # i/l/1 confusion
    "u": "U", "v": "U",             # u/v interchange
    "c": "C", "e": "C",             # c/e confusion
}


def _ocr_confuse_normalize(text: str) -> str:
    """Normalize a string by collapsing OCR-confusion equivalence classes.

    - i/l/1 → I
    - u/v → U
    - rn → m  (applied as substring)
    - c/e → C

    Used only for *matching purposes*, not for storing text.
    """
    text = text.lower()
    # rn → m (must happen before single-char replacements)
    text = text.replace("rn", "m")
    result: list[str] = []
    for ch in text:
        result.append(_OCR_CONFUSION_CLASSES.get(ch, ch))
    return "".join(result)


def ocr_aware_similarity(a: str, b: str) -> float:
    """Similarity (0..1) accounting for common OCR confusions.

    Normalizes both strings with OCR-confusion classes, then computes
    Dice coefficient of character bigrams.  This is more tolerant of
    medieval OCR errors (i/l/1, u/v, rn/m, c/e) than raw edit distance.
    """
    na = _ocr_confuse_normalize(normalize_for_search(a))
    nb = _ocr_confuse_normalize(normalize_for_search(b))
    if na == nb:
        return 1.0
    ba = _char_bigrams(na)
    bb = _char_bigrams(nb)
    if not ba or not bb:
        return 0.0
    return 2 * len(ba & bb) / (len(ba) + len(bb))


# ── Character bigram utilities ─────────────────────────────────────────

def _char_bigrams(s: str) -> set[str]:
    """Return the set of character bigrams in *s*."""
    return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else set()


def bigram_overlap(a: str, b: str) -> float:
    """Dice coefficient of character bigrams between *a* and *b*.

    Returns 0.0–1.0.  Higher means more similar.
    """
    ba = _char_bigrams(a.lower())
    bb = _char_bigrams(b.lower())
    if not ba or not bb:
        return 0.0
    return 2 * len(ba & bb) / (len(ba) + len(bb))


def normalized_edit_distance(a: str, b: str) -> float:
    """Levenshtein edit distance normalized by the length of the longer string.

    Returns 0.0–1.0.  Lower means more similar.
    """
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return 1.0
    if m == 0:
        return 1.0
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m] / max(n, m)


def is_blacklisted_token(token: str) -> bool:
    """Return True if *token* is in the common-word blacklist."""
    return token.lower().strip() in COMMON_WORD_BLACKLIST
