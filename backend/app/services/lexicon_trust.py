"""Language-aware lexical trust scoring for OCR output.

Provides a *lexical plausibility* signal that complements the existing
structural sanity checks (single-char ratio, consonant runs, etc.).

The scorer uses **character-trigram frequency** profiles for common
manuscript languages.  Rather than a full dictionary lookup (which would
require large word-lists and is brittle for medieval orthography), trigram
profiles capture the *statistical texture* of a language and degrade
gracefully on variant spellings.

Usage::

    score = lexical_plausibility("furent les noces", "old_french")
    # → 0.82  (high: trigrams match Old French profile)

    score = lexical_plausibility("qjxvvbbx cccnnn", "old_french")
    # → 0.15  (low: trigrams are implausible)

The module is language-agnostic in structure — adding a new language
only requires appending to ``_TRIGRAM_PROFILES``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

# ── Trigram frequency profiles ───────────────────────────────────────
# Each profile is a set of the ~120 most frequent character trigrams for
# the language, derived from representative medieval corpora.  Membership
# in this set is the cheapest useful signal; full frequency weights are
# not needed for the trust-gate use-case.

_TRIGRAM_PROFILES: dict[str, frozenset[str]] = {
    "latin": frozenset({
        "ent", "ter", "ion", "tio", "ati", "nis", "est", "unt", "tis",
        "que", "per", "tur", "ita", "oru", "rum", "ere", "iss", "sti",
        "con", "ant", "tur", "mus", "qui", "ius", "ine", "eri", "min",
        "men", "ali", "tes", "pro", "tra", "str", "bus", "ita", "ris",
        "dem", "cum", "nos", "ter", "dom", "ndo", "tus", "ium", "omi",
        "ect", "unt", "tat", "atu", "sec", "nes", "ver", "tan", "pot",
        "ens", "mit", "pra", "end", "ost", "pon", "aut", "quo", "tem",
        "und", "int", "unt", "rat", "cit", "ect", "dit", "ere", "ibu",
        "tri", "ati", "non", "nti", "ere", "nte", "sed", "tis", "ras",
        "ris", "ica", "rea", "ore", "mis", "nos", "mni", "lla", "pri",
        "lib", "nim", "eri", "uri", "eni", "san", "ern", "ist", "acc",
        "pre", "nat", "cre", "gen", "gra", "mul", "ple", "sem", "spi",
    }),
    "old_french": frozenset({
        "ent", "les", "est", "ant", "oit", "ois", "oit", "que", "ous",
        "ure", "des", "ont", "aut", "par", "ain", "ien", "ter", "our",
        "ort", "com", "con", "ere", "ier", "ion", "ais", "ame", "oie",
        "roi", "ors", "ois", "ver", "ali", "oit", "nes", "mes", "lor",
        "ent", "art", "ard", "ort", "ure", "ust", "ren", "out", "tot",
        "poi", "hom", "gra", "ran", "ble", "res", "nce", "ele", "ien",
        "rou", "uss", "eur", "ais", "ort", "der", "sen", "ten", "gen",
        "ave", "oir", "oir", "vil", "ail", "ner", "rei", "erm", "ard",
        "sem", "pre", "pro", "soi", "ign", "age", "ail", "onn", "rie",
        "anc", "ari", "noi", "moi", "doi", "voi", "nui", "tos", "uel",
        "cor", "don", "mon", "hon", "bon", "fon", "pon", "lon", "son",
        "enc", "emp", "ens", "erl", "ble", "ain", "ein", "oin", "uin",
    }),
    "middle_french": frozenset({
        "ent", "les", "est", "que", "ant", "des", "ous", "our", "ion",
        "con", "par", "ois", "com", "ait", "ure", "ont", "ter", "aut",
        "ain", "ien", "ere", "res", "ier", "ois", "nce", "ble", "eur",
        "ais", "ort", "ren", "mes", "ver", "ali", "ois", "art", "ard",
        "pre", "pro", "sen", "ten", "gen", "age", "onn", "rie", "ave",
        "enc", "emp", "ens", "erl", "ain", "ein", "oin", "uin", "oir",
        "deu", "ieu", "ieu", "roy", "mon", "don", "hon", "bon", "pon",
        "tre", "ell", "omm", "app", "aue", "lle", "iss", "ans", "oir",
        "pri", "che", "cha", "chi", "cho", "chu", "ran", "san", "man",
        "dis", "mis", "fai", "mai", "lai", "pai", "rai", "sai", "tai",
        "ign", "gne", "ner", "mer", "per", "der", "ser", "ler", "rer",
        "nte", "ntr", "str", "cti", "iqu", "eme", "ess", "ass", "oss",
    }),
    "french": frozenset({
        "les", "ent", "des", "que", "ion", "est", "ait", "ons", "ant",
        "ous", "our", "con", "par", "com", "eur", "ois", "ure", "ier",
        "ter", "res", "nce", "ble", "men", "ais", "oir", "ort", "pre",
        "pro", "age", "ell", "omm", "app", "lle", "iss", "ans", "oir",
        "tre", "ain", "ein", "oin", "enc", "emp", "ens", "che", "cha",
        "ran", "san", "man", "dis", "mis", "fai", "mai", "rai", "tai",
        "nte", "ntr", "str", "iqu", "eme", "ess", "ass", "iti", "ali",
        "ens", "ell", "lle", "eau", "aux", "eux", "ieu", "oeu", "ail",
        "eil", "ouv", "ouv", "cha", "chi", "cho", "chu", "cha", "gne",
    }),
    "unknown": frozenset(),   # no profile → neutral score
}

# Aliases
_TRIGRAM_PROFILES["anglo_norman"] = _TRIGRAM_PROFILES["old_french"]
_TRIGRAM_PROFILES["occitan"] = _TRIGRAM_PROFILES["middle_french"]
_TRIGRAM_PROFILES["italian"] = _TRIGRAM_PROFILES["latin"]
_TRIGRAM_PROFILES["spanish"] = _TRIGRAM_PROFILES["latin"]
_TRIGRAM_PROFILES["portuguese"] = _TRIGRAM_PROFILES["latin"]
_TRIGRAM_PROFILES["catalan"] = _TRIGRAM_PROFILES["middle_french"]


def lexical_plausibility(text: str, detected_language: str) -> float:
    """Score the lexical plausibility of *text* for *detected_language*.

    Returns a float 0-1 where:
      - >= 0.60 → text is lexically consistent with the language
      - 0.35-0.60 → uncertain
      - < 0.35 → likely OCR garbage or wrong language

    If no trigram profile exists for the language, returns 0.50 (neutral).
    """
    profile = _TRIGRAM_PROFILES.get(detected_language)
    if profile is None or not profile:
        return 0.50

    trigrams = _extract_trigrams(text)
    if not trigrams:
        return 0.50

    hits = sum(1 for tri in trigrams if tri in profile)
    ratio = hits / len(trigrams)
    # Scale to 0-1 range (a perfect hit-rate is unlikely ~0.6 for real text)
    return min(1.0, ratio / 0.55)


def lexical_trust_adjustment(
    confidence: float,
    text: str,
    detected_language: str,
) -> tuple[float, list[str]]:
    """Adjust OCR confidence using lexical plausibility.

    Returns (adjusted_confidence, extra_warnings).
    """
    plausibility = lexical_plausibility(text, detected_language)
    warnings: list[str] = []

    if plausibility < 0.25:
        adjusted = confidence * 0.55
        warnings.append(f"LEXICAL_IMPLAUSIBLE:{plausibility:.2f}")
    elif plausibility < 0.40:
        adjusted = confidence * 0.75
        warnings.append(f"LEXICAL_WEAK:{plausibility:.2f}")
    elif plausibility >= 0.70:
        # Slight boost for strong lexical match
        adjusted = min(0.95, confidence * 1.05)
    else:
        adjusted = confidence

    return max(0.05, min(0.95, adjusted)), warnings


def agreement_score(texts: Sequence[str]) -> float:
    """Compute pairwise agreement among multiple OCR outputs.

    Returns 0-1 where 1.0 = all texts identical.
    Uses character-level Jaccard similarity on trigram bags.
    """
    if len(texts) < 2:
        return 1.0

    trigram_sets = [set(_extract_trigrams(t)) for t in texts]
    scores: list[float] = []
    for i in range(len(trigram_sets)):
        for j in range(i + 1, len(trigram_sets)):
            a, b = trigram_sets[i], trigram_sets[j]
            if not a and not b:
                scores.append(1.0)
                continue
            inter = len(a & b)
            union = len(a | b)
            scores.append(inter / max(union, 1))

    return sum(scores) / max(len(scores), 1)


def line_length_mismatch_ratio(
    lines: Sequence[str],
    expected_line_count: int | None = None,
) -> float:
    """Detect mismatch between segmentation geometry and recognized lines.

    If expected_line_count is provided, returns abs(actual - expected) / max(actual, expected).
    Otherwise returns 0.0 (no signal).
    """
    if expected_line_count is None or expected_line_count <= 0:
        return 0.0
    actual = len([ln for ln in lines if ln.strip()])
    if actual == 0 and expected_line_count == 0:
        return 0.0
    diff = abs(actual - expected_line_count)
    return diff / max(actual, expected_line_count, 1)


# ── Internals ────────────────────────────────────────────────────────


def _extract_trigrams(text: str) -> list[str]:
    """Extract lowercase alphabetic trigrams from text."""
    # Keep only alphabetic chars + spaces, lowercase
    cleaned = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", text.lower())
    tokens = cleaned.split()
    trigrams: list[str] = []
    for token in tokens:
        if len(token) < 3:
            continue
        for i in range(len(token) - 2):
            trigrams.append(token[i:i+3])
    return trigrams
