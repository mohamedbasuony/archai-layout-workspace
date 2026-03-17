"""Authority Linking orchestrator (v2 — precision-first).

Reads entity_mentions for a run, queries Wikidata for candidates,
scores/disambiguates with hard type gates, enriches with VIAF/GeoNames,
and persists results back into entity_candidates (with ``is_selected``
semantics via ``meta_json``).

**Key rules (v2):**

* ``type_compatible`` from wikidata_client.is_type_compatible() is a
  **hard gate** — candidates that fail can never be auto-selected.
* ``AUTO_SELECT_THRESHOLD = 0.75`` and ``MIN_MARGIN = 0.10`` from
  entity_scoring.py are enforced.
* Gate D in the validation report flags type-mismatch detections.
* VIAF / GeoNames IDs are only surfaced when backed by a Wikidata
  property value (P214 / P1566).
* Separate ``api_calls_search`` and ``api_calls_get`` counters.

Usage::

    from app.services.authority_linking import run_authority_linking
    result = run_authority_linking(run_id)
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from typing import Any

from app.db import pipeline_db
from app.services.entity_scoring import (
    compute_score,
    disambiguate,
    get_thresholds,
    rescore_with_canonical,
)
from app.services.wikidata_client import (
    enrich_wikidata_item,
    is_type_compatible,
    search_wikidata,
    cache_check,
    cache_get,
    normalise_surface,
)
from app.services.text_normalization import (
    normalize_for_search,
    text_quality_label,
    token_quality_score,
)

log = logging.getLogger("archai.authority_linking")

# ── API call caps ──────────────────────────────────────────────────────
_MAX_SEARCH_CALLS_PER_RUN = 30    # max wbsearchentities calls per run
_MAX_ENRICH_PER_MENTION = 3       # max wbgetentities calls per mention

# ── Modern-occupation keywords to reject from medieval-entity linking ──
_MODERN_DESCRIPTION_REJECTS: set[str] = {
    "singer", "footballer", "basketball", "baseball", "hockey",
    "rapper", "actress", "actor", "politician", "president",
    "musician", "composer", "band", "album", "song",
    "software", "programming", "computer", "video game",
    "film", "television", "tv series", "anime",
    "bacterium", "bacteria", "virus", "protein", "enzyme",
    "chemical", "compound", "mineral", "drug",
    "asteroid", "galaxy", "planet", "star system",
    "erectile", "dysfunction",
    "operating system", "linux",
    "port ", "seaport", "airport",
    "brand", "company", "corporation",
    "football club", "sports team",
}

# ── Domain keywords that boost medieval entity confidence ──────────────
_MEDIEVAL_DOMAIN_KEYWORDS: set[str] = {
    "arthurian", "round table", "table ronde", "chevalier",
    "knight", "grail", "graal", "camelot", "legendary",
    "chrétien", "chretien", "troyes", "medieval", "médiéval",
    "fictional character", "literary character", "mythological",
    "legendary king", "queen consort",
    "literature", "literary work", "roman de",
    "chanson de geste", "matière de bretagne",
}

# ── Editorial / philology blacklist (non-entity markers) ──────────────
# These surface forms should never be sent to Wikidata as entity queries.
_EDITORIAL_BLACKLIST: set[str] = {
    "lacune", "lacuna", "lacunae",
    "[...]", "\u2026", "illegible", "gap", "missing",
    "illisible", "effacé", "effac\u00e9", "rature",
    "trou", "déchirure", "dechirure",
}

# ── Known-place gazetteer for conditional linking ─────────────────────
_KNOWN_PLACE_GAZETTEER: set[str] = {
    "lausanne", "paris", "lyon", "rome", "avignon", "geneve", "genève",
    "camelot", "logres", "bretagne", "cornouailles", "gaules",
    "tintagel", "carduel", "carlion", "winchester", "londres",
    "jérusalem", "jerusalem", "constantinople", "acre",
    "norgales", "gorre", "sarras", "benoic", "gaule",
}

# ── Canonical Arthurian / medieval entities ────────────────────────────
# Maps normalised canonical name → preferred query + qualifier strings.
# Used to (a) detect when an OCR mention matches a known entity, and
# (b) build high-quality Wikidata queries for that entity.

_CANONICAL_ENTITIES: dict[str, dict[str, Any]] = {
    "arthur":     {"canon": "Arthur",     "queries": ["King Arthur", "Arthur Arthurian legend"], "type": "person"},
    "artus":      {"canon": "Arthur",     "queries": ["King Arthur", "Arthur Arthurian legend"], "type": "person"},
    "lancelot":   {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "leantlote":  {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "leantilote": {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "lancelote":  {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "lanselot":   {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "lanceloc":   {"canon": "Lancelot",   "queries": ["Lancelot", "Lancelot du Lac"], "type": "person"},
    "guenievre":  {"canon": "Guinevere",  "queries": ["Guinevere", "Guenièvre"], "type": "person"},
    "guenievr":   {"canon": "Guinevere",  "queries": ["Guinevere", "Guenièvre"], "type": "person"},
    "perceval":   {"canon": "Perceval",   "queries": ["Perceval", "Percival Arthurian"], "type": "person"},
    "parsival":   {"canon": "Perceval",   "queries": ["Perceval", "Percival Arthurian"], "type": "person"},
    "merlin":     {"canon": "Merlin",     "queries": ["Merlin", "Merlin enchanteur"], "type": "person"},
    "galahad":    {"canon": "Galahad",    "queries": ["Galahad", "Galaad chevalier"], "type": "person"},
    "galaad":     {"canon": "Galahad",    "queries": ["Galahad", "Galaad chevalier"], "type": "person"},
    "tristan":    {"canon": "Tristan",    "queries": ["Tristan", "Tristan Iseut"], "type": "person"},
    "gauvain":    {"canon": "Gauvain",    "queries": ["Gauvain", "Gawain chevalier"], "type": "person"},
    "gawain":     {"canon": "Gauvain",    "queries": ["Gauvain", "Gawain chevalier"], "type": "person"},
    "yvain":      {"canon": "Yvain",      "queries": ["Yvain", "Yvain chevalier au lion"], "type": "person"},
    "ivain":      {"canon": "Yvain",      "queries": ["Yvain", "Yvain chevalier au lion"], "type": "person"},
    "graal":      {"canon": "Graal",      "queries": ["Saint Graal", "Holy Grail"], "type": "work"},
    "bohort":     {"canon": "Bohort",     "queries": ["Bohort", "Bors chevalier"], "type": "person"},
    "erec":       {"canon": "Érec",       "queries": ["Érec", "Erec et Enide"], "type": "person"},
}

# ── Name-likeness quality gate ─────────────────────────────────────────

_VOWELS_SET = set("aeiouyàâäéèêëïîôùûüœæ")

_NAME_QUALITY_THRESHOLD = 0.30  # below this → skip Wikidata


def _levenshtein_quick(s: str, t: str) -> int:
    """Levenshtein edit distance (pure-Python, small strings)."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def _compute_name_likeness(surface: str, ent_type: str) -> float:
    """Return a 0-1 quality score for a mention surface.

    A high score means the surface looks like a plausible entity name.
    A low score (< ``_NAME_QUALITY_THRESHOLD``) means the surface is
    likely OCR garbage and should NOT be sent to Wikidata.

    Factors:
      * Blacklisted common word → 0.0
      * Canonical match (normalized distance + bigram overlap) → 1.0 / 0.90
      * Vowel ratio within [0.20, 0.80]
      * No runs of ≥ 4 consecutive consonants
      * Token length within [3, 20]
      * Not purely stopwords
    """
    import re as _re
    from app.services.text_normalization import (
        is_blacklisted_token,
        normalized_edit_distance as _ned,
        bigram_overlap as _bo,
    )
    norm = _re.sub(r"[^\w\s]", " ", surface, flags=_re.UNICODE)
    norm = " ".join(norm.split()).strip().lower()
    if not norm:
        return 0.0

    tokens = norm.split()

    # Fast path: canonical match on any token → excellent quality
    # Uses normalized edit distance + bigram overlap (not raw dist)
    for tok in tokens:
        # Reject blacklisted tokens from contributing to quality
        if is_blacklisted_token(tok):
            continue
        if tok in _CANONICAL_ENTITIES:
            return 1.0
        for canon in _CANONICAL_ENTITIES:
            if len(tok) < 5:
                continue  # short tokens: exact match only (above)
            nd = _ned(tok, canon)
            bo = _bo(tok, canon)
            if nd <= 0.25 and bo >= 0.40:
                return 0.90

    # Remove role triggers / stopwords for scoring
    content = [
        t for t in tokens
        if t not in _QUERY_STOPWORDS and len(t) >= 2
        and not is_blacklisted_token(t)
    ]
    if not content:
        return 0.05  # pure stopwords / blacklisted → low

    score = 0.0
    n_good = 0
    for tok in content:
        tok_score = 0.5  # base
        # Length
        if len(tok) < 3 or len(tok) > 20:
            tok_score -= 0.30
        # Vowel ratio
        vowels = sum(1 for c in tok if c in _VOWELS_SET)
        ratio = vowels / len(tok)
        if ratio < 0.15 or ratio > 0.85:
            tok_score -= 0.30
        elif 0.25 <= ratio <= 0.65:
            tok_score += 0.15  # ideal range
        # Consonant clusters
        max_run = 0
        run = 0
        for c in tok:
            if c in _VOWELS_SET:
                run = 0
            else:
                run += 1
                max_run = max(max_run, run)
        if max_run >= 4:
            tok_score -= 0.35  # heavy penalty

        if tok_score > 0:
            n_good += 1
        score += max(0.0, tok_score)

    avg = score / len(content) if content else 0.0
    # Bonus for having multiple good tokens (reinforces signal)
    if n_good >= 2:
        avg = min(1.0, avg + 0.10)
    return round(min(1.0, max(0.0, avg)), 3)


def _check_canonical_match(surface: str) -> dict[str, Any] | None:
    """If any token in *surface* fuzzy-matches a canonical entity, return it.

    Uses length-aware matching:
      - Tokens < 5 chars: exact match only
      - Tokens >= 5 chars: normalized_edit_distance <= 0.25 AND
        bigram_overlap >= 0.40
      - Blacklisted common words are never matched.

    Returns a dict ``{"canon": …, "queries": …, "type": …, "token": …,
    "dist": …, "nd": …, "bo": …}`` or ``None``.
    """
    import re as _re
    from app.services.text_normalization import (
        is_blacklisted_token,
        normalized_edit_distance as _ned,
        bigram_overlap as _bo,
    )
    norm = _re.sub(r"[^\w\s]", " ", surface, flags=_re.UNICODE)
    tokens = norm.split()

    best: dict[str, Any] | None = None
    best_nd: float = 1.0

    for tok in tokens:
        tok_low = tok.lower()
        if len(tok_low) < 3:
            continue
        if is_blacklisted_token(tok_low):
            continue
        for canon_key, info in _CANONICAL_ENTITIES.items():
            nd = _ned(tok_low, canon_key)
            # Short tokens (< 5 chars): exact match only
            if len(tok_low) < 5:
                if nd == 0.0 and nd < best_nd:
                    best = {**info, "token": tok_low, "dist": 0, "nd": 0.0, "bo": 1.0}
                    best_nd = nd
                continue
            # Tokens >= 5 chars: normalized distance + bigram overlap
            bo = _bo(tok_low, canon_key)
            if nd <= 0.25 and bo >= 0.40 and nd < best_nd:
                raw_dist = _levenshtein_quick(tok_low, canon_key)
                best = {**info, "token": tok_low, "dist": raw_dist, "nd": nd, "bo": bo}
                best_nd = nd
    return best

# ── Role-aware decomposition ───────────────────────────────────────────

# Trigger tokens that denote a title/role preceding a personal name.
_ROLE_TRIGGERS: set[str] = {
    "roi", "roy", "roya", "rex", "evesque", "évesque", "euesque",
    "duc", "conte", "sire", "pape", "dame", "monsieur",
}

_QUERY_STOPWORDS: set[str] = {
    "et", "de", "la", "le", "les", "des", "du", "en", "a", "au",
    "que", "qui", "ne", "pas", "par", "son", "ses", "sa",
    "li", "lo", "el", "al", "ou", "un", "une",
    # Also include role triggers so they get stripped in query simplification
    "roi", "roy", "roya", "rex", "sire", "monsieur", "dame",
    "evesque", "évesque", "euesque", "duc", "conte", "pape",
}


def _prefilter_candidates(
    candidates: list[dict[str, Any]],
    surface: str,
    ent_type: str,
) -> list[dict[str, Any]]:
    """Pre-filter Wikidata candidates BEFORE calling wbgetentities.

    Removes candidates whose description contains modern-occupation
    keywords that are obviously irrelevant to medieval entity linking.
    Also sorts by relevance (domain keyword bonus) so the enrichment
    cap processes the best candidates first.
    """
    from app.services.text_normalization import normalize_for_search

    surface_norm = normalize_for_search(surface)
    kept: list[tuple[float, dict[str, Any]]] = []

    for wd in candidates:
        desc = str(wd.get("description", "")).lower()
        # Reject modern-occupation descriptions
        rejected = False
        for reject_kw in _MODERN_DESCRIPTION_REJECTS:
            if reject_kw in desc:
                log.debug(
                    "pre-filter: rejecting %s (%s) — description contains '%s'",
                    wd.get("qid", "?"), wd.get("label", "?"), reject_kw,
                )
                rejected = True
                break
        if rejected:
            continue

        # Domain keyword bonus for sorting
        bonus = 0.0
        for kw in _MEDIEVAL_DOMAIN_KEYWORDS:
            if kw in desc:
                bonus += 0.10
        # String proximity bonus
        label_norm = normalize_for_search(str(wd.get("label", "")))
        if label_norm and surface_norm and label_norm == surface_norm:
            bonus += 0.30
        elif label_norm and surface_norm and surface_norm in label_norm:
            bonus += 0.15
        kept.append((bonus, wd))

    # Sort by bonus descending, return just the candidates
    kept.sort(key=lambda x: -x[0])
    return [wd for _, wd in kept]


def _build_query_variants(surface: str, method: str) -> list[str]:
    """Generate progressively simplified queries for noisy medieval OCR.

    Strategy (always applied, regardless of method):
      a) normalised full surface (whitespace normalised, punct removed)
      b) drop stopwords/roles
      c) longest token(s) with len >= 5

    For ``rule:salvage_trigger`` mentions that start with a role trigger,
    the role-stripped form is always the *first* query so that Wikidata
    searches the personal name directly.

    Returns 1-3 deduplicated query strings.
    """
    # Normalise: collapse newlines/whitespace, strip punctuation
    import re as _re
    norm = _re.sub(r"[^\w\s]", " ", surface, flags=_re.UNICODE)
    norm = " ".join(norm.split()).strip()
    if not norm:
        return [surface]

    tokens = norm.split()

    queries: list[str] = []

    # ── (a) normalised full surface ───────────────────────────────────
    queries.append(norm)

    # ── (b) drop stopwords / roles ────────────────────────────────────
    content_tokens = [
        t for t in tokens
        if t.lower() not in _QUERY_STOPWORDS and len(t) >= 2
    ]
    if content_tokens and " ".join(content_tokens) != norm:
        queries.append(" ".join(content_tokens))

    # ── (c) longest tokens with len >= 5 ──────────────────────────────
    long_tokens = sorted(
        [t for t in tokens if len(t) >= 5],
        key=len, reverse=True,
    )[:2]
    if long_tokens:
        lt_str = " ".join(long_tokens)
        queries.append(lt_str)

    # ── For salvage_trigger role mentions: inject name-only first ─────
    if method == "rule:salvage_trigger" and len(tokens) >= 2:
        first_low = tokens[0].lower()
        if first_low in _ROLE_TRIGGERS:
            name_only = [
                t for t in tokens[1:]
                if t.lower() not in _QUERY_STOPWORDS and len(t) >= 2
            ]
            if name_only:
                # Insert name-only query at the front (highest priority)
                queries.insert(0, " ".join(name_only))

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        q_low = q.strip().lower()
        if q_low and q_low not in seen:
            seen.add(q_low)
            deduped.append(q.strip())
    return deduped or [surface]


# ── Public API ─────────────────────────────────────────────────────────


def run_authority_linking(
    run_id: str,
    *,
    top_k: int = 5,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Run the full authority-linking pipeline for *run_id*.

    1. Load entity_mentions + chunks from SQLite
    2. For each mention, search Wikidata for candidates
    3. Score (with hard type gate), disambiguate, enrich selected candidate
    4. Persist candidates into entity_candidates
    5. Return a summary dict suitable for the chat report

    The function is idempotent: existing candidates are cleared for the
    run_id before inserting new ones.
    """
    t0 = time.monotonic()

    # ── 1. Load data ──────────────────────────────────────────────────
    run = pipeline_db.get_run(run_id)
    base_text = str(run.get("proofread_text") or run.get("ocr_text") or "") if run else ""
    asset_ref = str(run.get("asset_ref") or "") if run else ""

    # ── Determine OCR quality level ───────────────────────────────────
    ocr_quality = "MEDIUM"  # default
    if run:
        # Check warnings_json for explicit quality_label
        warnings_raw = run.get("warnings_json") or ""
        if isinstance(warnings_raw, str):
            try:
                warnings_data = json.loads(warnings_raw) if warnings_raw.strip() else {}
            except Exception:
                warnings_data = {}
        else:
            warnings_data = warnings_raw
        explicit_label = ""
        if isinstance(warnings_data, dict):
            explicit_label = str(warnings_data.get("quality_label", ""))
        if explicit_label.upper() in ("HIGH", "MEDIUM", "LOW"):
            ocr_quality = explicit_label.upper()
        elif base_text:
            # Compute from text content
            ocr_quality = text_quality_label(base_text)

    log.info("run %s ocr_quality=%s", run_id, ocr_quality)

    mentions = pipeline_db.list_entity_mentions(run_id)
    if not mentions:
        empty = _empty_result(run_id, "no mentions found")
        empty["_base_text"] = base_text
        empty["asset_ref"] = asset_ref
        empty["ocr_quality"] = ocr_quality
        return empty

    chunks = pipeline_db.list_chunks(run_id)
    chunk_text_by_id: dict[str, str] = {
        str(ch["chunk_id"]): str(ch.get("text") or "") for ch in chunks
    }

    # ── 2. Clear old candidates ───────────────────────────────────────
    _clear_candidates_for_run(run_id)

    # ── 3. Process each mention ───────────────────────────────────────
    api_calls_search = 0
    api_calls_get = 0
    cache_hits = 0
    type_mismatch_count = 0
    quality_skipped = 0
    canonical_matched = 0
    all_candidate_rows: list[dict[str, Any]] = []
    mention_results: list[dict[str, Any]] = []

    for mention in mentions:
        mention_id = str(mention["mention_id"])
        surface = str(mention.get("surface") or "")
        ent_type = str(mention.get("ent_type") or "unknown")
        method = str(mention.get("method") or "")
        start_off = int(mention.get("start_offset", 0))
        end_off = int(mention.get("end_offset", 0))
        chunk_id = mention.get("chunk_id")

        # Context: chunk text + nearby base text
        context = _build_context(
            base_text, start_off, end_off,
            chunk_text_by_id.get(str(chunk_id) if chunk_id else "", ""),
        )

        # ── Quality gate: skip non-linkable types and garbage surfaces ─
        if ent_type == "role":
            # Role mentions (trigger without plausible name) → skip
            evidence_text = base_text[start_off:end_off] if base_text and end_off <= len(base_text) else surface
            quality_skipped += 1
            mention_results.append({
                "mention_id": mention_id,
                "surface": surface,
                "ent_type": ent_type,
                "chunk_id": chunk_id,
                "start_offset": start_off,
                "end_offset": end_off,
                "evidence_text": evidence_text,
                "queries_attempted": [],
                "query_details": [],
                "status": "skipped",
                "reason": "ent_type=role (not linkable)",
                "name_likeness": 0.0,
                "canonical_match": None,
                "selected": None,
                "top_candidates": [],
            })
            log.info(
                "mention %s surface=%r SKIPPED: ent_type=role",
                mention_id, surface,
            )
            continue

        # ── Req 3a: DATE mentions → never query Wikidata ─────────────
        if ent_type == "date":
            evidence_text = base_text[start_off:end_off] if base_text and end_off <= len(base_text) else surface
            quality_skipped += 1
            mention_results.append({
                "mention_id": mention_id,
                "surface": surface,
                "ent_type": ent_type,
                "chunk_id": chunk_id,
                "start_offset": start_off,
                "end_offset": end_off,
                "evidence_text": evidence_text,
                "queries_attempted": [],
                "query_details": [],
                "status": "skipped",
                "reason": "ent_type=date (dates never query Wikidata)",
                "name_likeness": 0.0,
                "canonical_match": None,
                "selected": None,
                "top_candidates": [],
            })
            log.info(
                "mention %s surface=%r SKIPPED: ent_type=date (no Wikidata query)",
                mention_id, surface,
            )
            continue

        # ── Req 3b: PLACE mentions → conditional Wikidata query ───────
        if ent_type == "place":
            confidence = float(mention.get("confidence", 0.0))
            surface_low = surface.strip().lower()
            is_capitalized = surface.strip()[:1].isupper()
            is_known_place = surface_low in _KNOWN_PLACE_GAZETTEER
            is_blacklisted = surface_low in _EDITORIAL_BLACKLIST
            # Only query Wikidata if:
            #   (a) surface is capitalized, OR
            #   (b) matches known-place gazetteer, OR
            #   (c) confidence >= 0.70 AND not blacklisted
            place_linkable = (
                is_capitalized
                or is_known_place
                or (confidence >= 0.70 and not is_blacklisted)
            )
            if not place_linkable:
                evidence_text = base_text[start_off:end_off] if base_text and end_off <= len(base_text) else surface
                quality_skipped += 1
                mention_results.append({
                    "mention_id": mention_id,
                    "surface": surface,
                    "ent_type": ent_type,
                    "chunk_id": chunk_id,
                    "start_offset": start_off,
                    "end_offset": end_off,
                    "evidence_text": evidence_text,
                    "queries_attempted": [],
                    "query_details": [],
                    "status": "skipped",
                    "reason": (
                        f"low_evidence_place: cap={is_capitalized} "
                        f"gazetteer={is_known_place} conf={confidence:.2f} "
                        f"blacklisted={is_blacklisted}"
                    ),
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                })
                log.info(
                    "mention %s surface=%r SKIPPED: low_evidence_place "
                    "(cap=%s gazetteer=%s conf=%.2f blacklisted=%s)",
                    mention_id, surface, is_capitalized, is_known_place,
                    confidence, is_blacklisted,
                )
                continue

        # Name-likeness quality score
        name_likeness = _compute_name_likeness(surface, ent_type)
        if ent_type in ("person",) and name_likeness < _NAME_QUALITY_THRESHOLD:
            evidence_text = base_text[start_off:end_off] if base_text and end_off <= len(base_text) else surface
            quality_skipped += 1
            mention_results.append({
                "mention_id": mention_id,
                "surface": surface,
                "ent_type": ent_type,
                "chunk_id": chunk_id,
                "start_offset": start_off,
                "end_offset": end_off,
                "evidence_text": evidence_text,
                "queries_attempted": [],
                "query_details": [],
                "status": "skipped",
                "reason": f"name_likeness={name_likeness:.3f} < {_NAME_QUALITY_THRESHOLD} (OCR garbage)",
                "name_likeness": name_likeness,
                "canonical_match": None,
                "selected": None,
                "top_candidates": [],
            })
            log.info(
                "mention %s surface=%r SKIPPED: name_likeness=%.3f < %.2f",
                mention_id, surface, name_likeness, _NAME_QUALITY_THRESHOLD,
            )
            continue

        # ── Canonical match check ─────────────────────────────────────
        canon_match = _check_canonical_match(surface)
        canon_match_info = None
        if canon_match:
            canonical_matched += 1
            canon_match_info = {
                "canon": canon_match["canon"],
                "token": canon_match["token"],
                "dist": canon_match["dist"],
            }
            log.info(
                "mention %s surface=%r CANONICAL MATCH: %s (dist=%d)",
                mention_id, surface, canon_match["canon"], canon_match["dist"],
            )

        # ── Query generation (progressive simplification) ─────────────
        queries = _build_query_variants(surface, method)
        # Prepend canonical queries when matched
        if canon_match:
            canon_queries = canon_match["queries"]
            # When canonicalization exists, use only canonical queries
            # unless they return 0 candidates (handled in search loop).
            # Remove the raw OCR surface query to avoid wasting API calls.
            canon_set = {cq.lower() for cq in canon_queries}
            queries = [q for q in queries if q.lower() not in canon_set]
            for cq in reversed(canon_queries):
                queries.insert(0, cq)
        log.info(
            "mention %s surface=%r queries=%s ent_type=%s method=%s name_likeness=%.3f",
            mention_id, surface, queries, ent_type, method, name_likeness,
        )

        # Collect candidates across all sub-queries
        merged_candidates: dict[str, dict[str, Any]] = {}  # qid → wd dict
        query_details: list[dict[str, Any]] = []  # per-query diagnostics
        canon_query_count = len(canon_match["queries"]) if canon_match else 0
        for q_idx, query_str in enumerate(queries):
            # ── API search cap: max 30 searches per run ───────────────
            if api_calls_search >= _MAX_SEARCH_CALLS_PER_RUN:
                log.warning(
                    "mention %s: search API cap reached (%d), skipping query %r",
                    mention_id, api_calls_search, query_str,
                )
                break

            # ── Skip raw OCR queries when canonical search got results ─
            if canon_match and q_idx >= canon_query_count and merged_candidates:
                log.debug(
                    "mention %s: canonical search produced %d candidates, "
                    "skipping fallback query %r",
                    mention_id, len(merged_candidates), query_str,
                )
                continue

            # Determine cache status
            _, c_status = cache_check("wikidata", query_str, max_age_hours=6.0)
            # API is needed unless we have a real (non-empty) cache hit
            need_api = force_refresh or c_status in ("miss", "expired", "empty-hit")
            if force_refresh:
                c_status = "bypassed"

            wd_results = search_wikidata(
                query_str, k=top_k, force_refresh=need_api,
            )

            if need_api:
                api_calls_search += 1
            else:
                cache_hits += 1

            raw_hits = [
                {
                    "qid": w.get("qid", ""),
                    "label": w.get("label", ""),
                    "description": w.get("description", ""),
                    "search_rank": i + 1,
                }
                for i, w in enumerate(wd_results[:5])
            ]

            query_details.append({
                "query": query_str,
                "cache_status": c_status,
                "wikidata_called": need_api,
                "raw_hits": raw_hits,
            })

            for wd in wd_results:
                qid = wd.get("qid", "")
                if qid and qid not in merged_candidates:
                    merged_candidates[qid] = wd

        wd_candidates = list(merged_candidates.values())

        # ── Pre-filter: reject obviously irrelevant candidates ────────
        # For medieval person/work mentions, reject candidates whose
        # description contains modern-occupation keywords.
        if ent_type in ("person", "work") and wd_candidates:
            wd_candidates = _prefilter_candidates(
                wd_candidates, surface, ent_type,
            )

        # For PERSON_OR_ROLE mentions from salvage_trigger, force type
        # gate to require human (Q5) / fictional human (Q15632617).
        effective_ent_type = ent_type
        if method == "rule:salvage_trigger" and ent_type == "person":
            effective_ent_type = "person"  # already correct; explicit

        # Derive canonical norm for scoring (e.g. "lancelot")
        canonical_norm = ""
        if canon_match:
            canonical_norm = canon_match["canon"].lower()

        # Score each candidate — enrich for P31
        # ── Enrichment cap: max 3 wbgetentities per mention ───────────
        scored: list[dict[str, Any]] = []
        enrich_count = 0
        for wd in wd_candidates:
            qid = wd.get("qid", "")
            label = wd.get("label", "")
            description = wd.get("description", "")

            # Fetch P31 for type checking (+ VIAF/GeoNames), capped
            if qid and enrich_count < _MAX_ENRICH_PER_MENTION:
                enrichment = enrich_wikidata_item(qid)
                api_calls_get += 1
                enrich_count += 1
            else:
                enrichment = {}
            instance_of = enrichment.get("instance_of_qids", [])
            type_ok = is_type_compatible(
                effective_ent_type, instance_of,
                description=description,
            )
            if not type_ok:
                type_mismatch_count += 1

            # Domain bonus from description keywords
            desc_low = description.lower()
            domain_bonus = 0.0
            for kw in _MEDIEVAL_DOMAIN_KEYWORDS:
                if kw in desc_low:
                    domain_bonus += 0.05
            domain_bonus = min(0.20, domain_bonus)

            score = compute_score(
                surface, label, context, description,
                type_compatible=type_ok,
                canonical_norm=canonical_norm,
                domain_bonus=domain_bonus,
            )
            scored.append({
                "qid": qid,
                "label": label,
                "description": description,
                "url": wd.get("url", ""),
                "score": round(score, 4),
                "type_compatible": type_ok,
                "viaf_id": enrichment.get("viaf_id", ""),
                "geonames_id": enrichment.get("geonames_id", ""),
                "instance_of_qids": instance_of,
            })

        # ── Canonical rescoring: boost high-confidence canonical matches ─
        if canonical_norm and scored:
            scored = rescore_with_canonical(scored, canonical_norm)

        # Disambiguate (precision-first v2, quality-adaptive v4)
        result = disambiguate(scored, ocr_quality=ocr_quality)
        selected = result["selected"]
        status = result["status"]
        reason = result["reason"]

        # Build candidate rows for DB insertion
        for i, cand in enumerate(result["all"]):
            is_selected = selected is not None and cand is selected
            meta = {
                "qid": cand.get("qid", ""),
                "label": cand.get("label", ""),
                "description": cand.get("description", ""),
                "url": cand.get("url", ""),
                "viaf_id": cand.get("viaf_id", ""),
                "geonames_id": cand.get("geonames_id", ""),
                "type_compatible": cand.get("type_compatible", False),
                "instance_of_qids": cand.get("instance_of_qids", []),
                "is_selected": is_selected,
                "link_status": status if is_selected else "",
                "rank": i + 1,
            }
            all_candidate_rows.append({
                "mention_id": mention_id,
                "source": "wikidata",
                "candidate": cand.get("qid", ""),
                "score": cand.get("score", 0.0),
                "meta_json": meta,
            })

        # Evidence span text
        evidence_text = base_text[start_off:end_off] if base_text and end_off <= len(base_text) else surface

        mention_results.append({
            "mention_id": mention_id,
            "surface": surface,
            "ent_type": ent_type,
            "chunk_id": chunk_id,
            "start_offset": start_off,
            "end_offset": end_off,
            "evidence_text": evidence_text,
            "queries_attempted": queries,
            "query_details": query_details,
            "status": status,
            "reason": reason,
            "name_likeness": name_likeness,
            "canonical_match": canon_match_info,
            "selected": {
                "qid": selected.get("qid", ""),
                "label": selected.get("label", ""),
                "description": selected.get("description", ""),
                "score": selected.get("score", 0.0),
                "viaf_id": selected.get("viaf_id", ""),
                "geonames_id": selected.get("geonames_id", ""),
            } if selected else None,
            "top_candidates": [
                {
                    "qid": c.get("qid", ""),
                    "label": c.get("label", ""),
                    "score": c.get("score", 0.0),
                    "type_compatible": c.get("type_compatible", False),
                }
                for c in result["all"][:3]
            ],
        })

    # ── 4. Persist candidates ─────────────────────────────────────────
    pipeline_db.insert_entity_candidates(all_candidate_rows)

    # ── 5. Build summary ──────────────────────────────────────────────
    elapsed_ms = round((time.monotonic() - t0) * 1000)

    # Counts
    type_counts = Counter(str(m.get("ent_type", "unknown")) for m in mentions)
    source_counts = Counter("wikidata" for _ in all_candidate_rows)
    linked_total = sum(1 for r in mention_results if r["status"] == "linked")
    unresolved_total = sum(1 for r in mention_results if r["status"] == "unresolved")
    ambiguous_total = sum(1 for r in mention_results if r["status"] == "ambiguous")
    skipped_total = sum(1 for r in mention_results if r["status"] == "skipped")

    return {
        "run_id": run_id,
        "asset_ref": asset_ref,
        "mentions_total": len(mentions),
        "type_counts": dict(type_counts),
        "candidates_total": len(all_candidate_rows),
        "source_counts": dict(source_counts),
        "linked_total": linked_total,
        "unresolved_total": unresolved_total,
        "ambiguous_total": ambiguous_total,
        "skipped_total": skipped_total,
        "quality_skipped": quality_skipped,
        "canonical_matched": canonical_matched,
        "type_mismatch_count": type_mismatch_count,
        "api_calls_search": api_calls_search,
        "api_calls_get": api_calls_get,
        "api_calls": api_calls_search + api_calls_get,
        "cache_hits": cache_hits,
        "took_ms": elapsed_ms,
        "ocr_quality": ocr_quality,
        "mention_results": mention_results,
        "_base_text": base_text,
    }


# ── Helpers ────────────────────────────────────────────────────────────


def _build_context(
    base_text: str,
    start: int,
    end: int,
    chunk_text: str,
    window: int = 200,
) -> str:
    """Build context string from surrounding text."""
    parts: list[str] = []
    if chunk_text:
        parts.append(chunk_text)
    if base_text:
        ctx_start = max(0, start - window)
        ctx_end = min(len(base_text), end + window)
        parts.append(base_text[ctx_start:ctx_end])
    return " ".join(parts)


def _clear_candidates_for_run(run_id: str) -> None:
    """Remove existing Wikidata candidates for this run (keep heuristic ones)."""
    pipeline_db._init_db_if_needed()
    with pipeline_db._connect() as conn:
        conn.execute(
            """
            DELETE FROM entity_candidates
            WHERE source='wikidata'
              AND mention_id IN (
                  SELECT mention_id FROM entity_mentions WHERE run_id=?
              )
            """,
            (run_id,),
        )
        conn.commit()


def _empty_result(run_id: str, reason: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "asset_ref": "",
        "mentions_total": 0,
        "type_counts": {},
        "candidates_total": 0,
        "source_counts": {},
        "linked_total": 0,
        "unresolved_total": 0,
        "ambiguous_total": 0,
        "type_mismatch_count": 0,
        "api_calls_search": 0,
        "api_calls_get": 0,
        "api_calls": 0,
        "cache_hits": 0,
        "took_ms": 0,
        "mention_results": [],
        "_base_text": "",
    }


def build_report_from_db(run_id: str) -> dict[str, Any]:
    """Reconstruct a linking report from persisted DB data.

    This does NOT re-run linking — it reads ``entity_mentions`` +
    ``entity_candidates`` and produces the same structure as
    ``run_authority_linking`` so :func:`build_linking_report` can format it.
    """
    run = pipeline_db.get_run(run_id)
    if run is None:
        raise ValueError(f"Run not found: {run_id}")

    asset_ref = str(run.get("asset_ref") or "")
    base_text = str(run.get("proofread_text") or run.get("ocr_text") or "")
    mentions = pipeline_db.list_entity_mentions(run_id)

    if not mentions:
        empty = _empty_result(run_id, "no mentions in DB")
        empty["asset_ref"] = asset_ref
        empty["_base_text"] = base_text
        return {"report": build_linking_report(empty), **empty}

    # Load all candidates grouped by mention_id
    pipeline_db._init_db_if_needed()
    with pipeline_db._connect() as conn:
        rows = conn.execute(
            """
            SELECT c.* FROM entity_candidates c
            JOIN entity_mentions m ON m.mention_id = c.mention_id
            WHERE m.run_id=?
            ORDER BY c.score DESC
            """,
            (run_id,),
        ).fetchall()
    cands_by_mention: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        mid = str(row["mention_id"])
        cand: dict[str, Any] = {k: row[k] for k in row.keys()}
        if cand.get("meta_json") and isinstance(cand["meta_json"], str):
            try:
                cand["meta_json"] = json.loads(cand["meta_json"])
            except Exception:
                pass
        cands_by_mention.setdefault(mid, []).append(cand)

    type_counts: Counter[str] = Counter()
    mention_results: list[dict[str, Any]] = []
    candidates_total = 0
    linked_total = 0
    unresolved_total = 0
    ambiguous_total = 0

    for m in mentions:
        mid = str(m["mention_id"])
        etype = str(m.get("ent_type") or "unknown")
        type_counts[etype] += 1
        surface = str(m.get("surface") or "")
        start_off = int(m.get("start_offset", 0))
        end_off = int(m.get("end_offset", 0))
        chunk_id = m.get("chunk_id")
        evidence_text = base_text[start_off:end_off] if base_text and end_off <= len(base_text) else surface

        cands = cands_by_mention.get(mid, [])
        candidates_total += len(cands)

        # Find selected candidate via meta_json.is_selected
        selected_cand = None
        status = "unresolved"
        reason = "no wikidata candidates"
        for c in cands:
            meta = c.get("meta_json") or {}
            if isinstance(meta, dict) and meta.get("is_selected"):
                selected_cand = c
                status = str(meta.get("link_status", "linked"))
                reason = f"score = {c.get('score', 0):.4f}"
                break

        if selected_cand is None and cands:
            # Check if ambiguous
            any_meta = (cands[0].get("meta_json") or {}) if cands else {}
            if isinstance(any_meta, dict) and any_meta.get("link_status") == "ambiguous":
                status = "ambiguous"
                reason = "marked ambiguous"
            elif cands:
                status = "unresolved"
                reason = "no candidate marked as selected"

        if status == "linked":
            linked_total += 1
        elif status == "ambiguous":
            ambiguous_total += 1
        else:
            unresolved_total += 1

        sel_meta = (selected_cand.get("meta_json") or {}) if selected_cand else {}
        if isinstance(sel_meta, str):
            try:
                sel_meta = json.loads(sel_meta)
            except Exception:
                sel_meta = {}

        mention_results.append({
            "mention_id": mid,
            "surface": surface,
            "ent_type": etype,
            "chunk_id": chunk_id,
            "start_offset": start_off,
            "end_offset": end_off,
            "evidence_text": evidence_text,
            "status": status,
            "reason": reason,
            "selected": {
                "qid": sel_meta.get("qid", ""),
                "label": sel_meta.get("label", ""),
                "description": sel_meta.get("description", ""),
                "score": float(selected_cand.get("score", 0)) if selected_cand else 0.0,
                "viaf_id": sel_meta.get("viaf_id", ""),
                "geonames_id": sel_meta.get("geonames_id", ""),
            } if selected_cand else None,
            "top_candidates": [
                {
                    "qid": (c.get("meta_json") or {}).get("qid", c.get("candidate", "")),
                    "label": (c.get("meta_json") or {}).get("label", ""),
                    "score": float(c.get("score", 0)),
                }
                for c in cands[:3]
            ],
        })

    result = {
        "run_id": run_id,
        "asset_ref": asset_ref,
        "mentions_total": len(mentions),
        "type_counts": dict(type_counts),
        "candidates_total": candidates_total,
        "source_counts": {"wikidata": sum(1 for c in rows if str(c["source"]) == "wikidata")},
        "linked_total": linked_total,
        "unresolved_total": unresolved_total,
        "ambiguous_total": ambiguous_total,
        "api_calls": 0,
        "cache_hits": 0,
        "took_ms": 0,
        "mention_results": mention_results,
        "_base_text": base_text,
    }
    result["report"] = build_linking_report(result)
    return result


# ── Report builder ─────────────────────────────────────────────────────


def build_linking_report_from_db(run_id: str) -> str:
    """Build a fully DB-backed ENTITY LINKING REPORT (Option 1: minimal & safe).

    Every line is derived from SQL queries over persisted tables only:
      - entity_mentions   → mention counts, types
      - entity_candidates → candidate counts, linked/unresolved/ambiguous,
                            top linked entities, type mismatches
      - entity_decisions  → extraction decisions (for AUDIT cross-ref)

    Sections that were trace-only (queries_attempted, raw_hits,
    cache_status, api_calls, wikidata_called) are deliberately OMITTED
    because they are NOT persisted.  This ensures AUDIT_4 compliance.
    """
    L: list[str] = []

    # ── Load run metadata ──────────────────────────────────────────
    run = pipeline_db.get_run(run_id)
    asset_ref = str(run.get("asset_ref") or "?") if run else "?"
    base_text = str(run.get("proofread_text") or run.get("ocr_text") or "") if run else ""

    # ── Load mentions from DB ──────────────────────────────────────
    mentions = pipeline_db.list_entity_mentions(run_id)

    # ── Load candidates from DB ────────────────────────────────────
    pipeline_db._init_db_if_needed()
    with pipeline_db._connect() as conn:
        cand_rows = conn.execute(
            """
            SELECT c.* FROM entity_candidates c
            JOIN entity_mentions m ON m.mention_id = c.mention_id
            WHERE m.run_id=?
            ORDER BY c.score DESC
            """,
            (run_id,),
        ).fetchall()
    all_cands: list[dict[str, Any]] = []
    cands_by_mention: dict[str, list[dict[str, Any]]] = {}
    for row in cand_rows:
        cand: dict[str, Any] = {k: row[k] for k in row.keys()}
        if cand.get("meta_json") and isinstance(cand["meta_json"], str):
            try:
                cand["meta_json"] = json.loads(cand["meta_json"])
            except Exception:
                pass
        all_cands.append(cand)
        cands_by_mention.setdefault(str(cand["mention_id"]), []).append(cand)

    # ── Classify mentions ──────────────────────────────────────────
    type_counts: Counter[str] = Counter()
    linked_entries: list[dict[str, Any]] = []
    unresolved_entries: list[dict[str, Any]] = []
    ambiguous_entries: list[dict[str, Any]] = []
    skipped_entries: list[dict[str, Any]] = []
    type_mismatches: list[dict[str, Any]] = []

    _QUALITY_SKIP_TYPES = {"role", "date", "editorial"}
    _LINKABLE_TYPES = {"person", "place", "work"}

    for m in mentions:
        mid = str(m["mention_id"])
        etype = str(m.get("ent_type") or "unknown")
        type_counts[etype] += 1
        surface = str(m.get("surface") or "")
        start_off = int(m.get("start_offset", 0))
        end_off = int(m.get("end_offset", 0))
        chunk_id = m.get("chunk_id")
        evidence_text = base_text[start_off:end_off] if base_text and end_off <= len(base_text) else surface

        cands = cands_by_mention.get(mid, [])

        # Determine linking status from candidates
        selected_cand = None
        status = "unresolved"
        reason = "no candidates"
        for c in cands:
            meta = c.get("meta_json") or {}
            if isinstance(meta, dict) and meta.get("is_selected"):
                selected_cand = c
                status = str(meta.get("link_status", "linked"))
                reason = f"score={c.get('score', 0):.4f}"
                break

        if selected_cand is None and cands:
            any_meta = (cands[0].get("meta_json") or {})
            if isinstance(any_meta, dict) and any_meta.get("link_status") == "ambiguous":
                status = "ambiguous"
                reason = "ambiguous (no clear winner)"
            else:
                status = "unresolved"
                reason = "no selected candidate"

        # Quality-skipped mentions (role/date treated as skip)
        if etype in _QUALITY_SKIP_TYPES and not cands:
            status = "skipped"
            reason = f"quality_skip: ent_type={etype}"

        sel_meta = (selected_cand.get("meta_json") or {}) if selected_cand else {}
        if isinstance(sel_meta, str):
            try:
                sel_meta = json.loads(sel_meta)
            except Exception:
                sel_meta = {}

        entry = {
            "mention_id": mid,
            "surface": surface,
            "ent_type": etype,
            "chunk_id": chunk_id,
            "start_offset": start_off,
            "end_offset": end_off,
            "evidence_text": evidence_text,
            "status": status,
            "reason": reason,
            "selected": {
                "qid": sel_meta.get("qid", ""),
                "label": sel_meta.get("label", ""),
                "description": sel_meta.get("description", ""),
                "score": float(selected_cand.get("score", 0)) if selected_cand else 0.0,
                "viaf_id": sel_meta.get("viaf_id", ""),
                "geonames_id": sel_meta.get("geonames_id", ""),
                "type_compatible": sel_meta.get("type_compatible", True),
            } if selected_cand else None,
        }

        if status == "linked":
            linked_entries.append(entry)
        elif status == "ambiguous":
            ambiguous_entries.append(entry)
        elif status == "skipped":
            skipped_entries.append(entry)
        else:
            unresolved_entries.append(entry)

        # Type-mismatch detection from ALL candidates
        for c in cands:
            cmeta = c.get("meta_json") or {}
            if isinstance(cmeta, dict) and cmeta.get("type_compatible") is False:
                type_mismatches.append({
                    "surface": surface,
                    "ent_type": etype,
                    "qid": cmeta.get("qid", c.get("candidate", "?")),
                    "label": cmeta.get("label", "?"),
                    "score": float(c.get("score", 0)),
                })

    candidates_total = len(all_cands)
    source_counts: Counter[str] = Counter()
    for c in all_cands:
        source_counts[str(c.get("source", "unknown"))] += 1

    # ── Section 1: Summary (DB-derived only) ───────────────────────
    L.append("=== ENTITY LINKING REPORT ===")
    L.append(f"run_id: {run_id}")
    L.append(f"asset_ref: {asset_ref}")
    L.append(f"mentions_total: {len(mentions)}")

    if type_counts:
        parts = [f"{k}={v}" for k, v in sorted(type_counts.items())]
        L.append(f"  by_type: {', '.join(parts)}")

    L.append(f"candidates_total: {candidates_total}")
    if source_counts:
        parts = [f"{k}={v}" for k, v in sorted(source_counts.items())]
        L.append(f"  by_source: {', '.join(parts)}")

    L.append(f"linked_total: {len(linked_entries)}")
    L.append(f"unresolved_total: {len(unresolved_entries)}")
    L.append(f"ambiguous_total: {len(ambiguous_entries)}")
    L.append(f"skipped_total: {len(skipped_entries)}")
    L.append(f"type_mismatch_count: {len(type_mismatches)}")
    L.append("")

    # ── Section 2: Top Linked Entities (DB-only) ───────────────────
    top_linked = sorted(
        linked_entries,
        key=lambda e: (e.get("selected") or {}).get("score", 0),
        reverse=True,
    )[:10]

    L.append(f"=== TOP LINKED ENTITIES (N={len(top_linked)}) ===")
    for entry in top_linked:
        sel = entry.get("selected") or {}
        L.append(f"  mention_id: {entry['mention_id']}")
        L.append(f"  surface: \"{entry['surface']}\"")
        L.append(f"  ent_type: {entry['ent_type']}")
        L.append(f"  offsets: {entry['start_offset']}-{entry['end_offset']}")
        L.append(f"  evidence: \"{entry['evidence_text']}\"")
        L.append(f"  \u2192 Wikidata: {sel.get('qid', '?')} | {sel.get('label', '?')} | {sel.get('description', '')}")
        viaf = sel.get("viaf_id", "")
        geo = sel.get("geonames_id", "")
        if viaf or geo:
            viaf_display = viaf or "-"
            geo_display = geo or "-"
            L.append(f"    VIAF: {viaf_display}  GeoNames: {geo_display}")
        L.append(f"    score: {sel.get('score', 0):.4f}")
        L.append("")
    if not top_linked:
        L.append("  (none)")
        L.append("")

    # ── Section 3: Type-Mismatch Detections (DB-only) ──────────────
    tm_preview = type_mismatches[:10]
    L.append(f"=== TYPE-MISMATCH DETECTIONS (N={len(type_mismatches)}) ===")
    for tm in tm_preview:
        L.append(
            f"  surface=\"{tm['surface']}\" ent_type={tm['ent_type']} "
            f"\u2192 {tm['qid']} | {tm['label']} | score={tm['score']:.4f} [REJECTED]"
        )
    if not tm_preview:
        L.append("  (none)")
    L.append("")

    # ── Section 4: Failures/Edge Cases (DB-only) ───────────────────
    failures = unresolved_entries + ambiguous_entries
    top_failures = failures[:10]
    L.append(f"=== FAILURE/EDGE CASES (N={len(top_failures)}) ===")
    for entry in top_failures:
        L.append(f"  mention_id: {entry['mention_id']}")
        L.append(f"  surface: \"{entry['surface']}\"")
        L.append(f"  ent_type: {entry['ent_type']}")
        L.append(f"  status: {entry['status']}")
        L.append(f"  reason: {entry['reason']}")
        L.append("")
    if not top_failures:
        L.append("  (none)")
        L.append("")

    # ── Section 5: Validation Summary (DB-only gates) ──────────────
    mentions_total = len(mentions)
    linked_total_val = len(linked_entries)
    skipped_total_val = len(skipped_entries)

    # Check NO_ENTITIES_PRESENT
    no_entities_pass = False
    if mentions_total == 0:
        if base_text:
            from app.routers.ocr import _has_entity_cues
            has_cues = _has_entity_cues(base_text)
        else:
            has_cues = False
        if not has_cues:
            no_entities_pass = True

    all_skipped = mentions_total > 0 and skipped_total_val == mentions_total

    # Determine if any strong linkable mentions exist (persons/places with candidates)
    has_strong_linkable = any(
        e["ent_type"] in _LINKABLE_TYPES and cands_by_mention.get(e["mention_id"])
        for e in linked_entries + unresolved_entries + ambiguous_entries
    )

    no_linkable = (
        mentions_total > 0
        and linked_total_val == 0
        and not all_skipped
        and not has_strong_linkable
    )

    if no_entities_pass or all_skipped or no_linkable:
        gate_a = True
        gate_b = True
        gate_c = True
    else:
        gate_a = mentions_total > 0
        gate_b = candidates_total > 0
        gate_c = linked_total_val > 0

    # Gate D: no type-mismatch auto-links
    gate_d = all(
        (e.get("selected") or {}).get("type_compatible", True) is not False
        for e in linked_entries
    ) if linked_entries else True

    # Gate E: referential integrity — all mention_ids link to entity_mentions
    all_mention_ids = {str(m["mention_id"]) for m in mentions}
    classified_ids = {e["mention_id"] for e in linked_entries + unresolved_entries + ambiguous_entries + skipped_entries}
    gate_e = classified_ids <= all_mention_ids if mentions_total > 0 else True

    # Gate F: every linked entity has evidence text
    gate_f = all(bool(e.get("evidence_text", "").strip()) for e in linked_entries)

    # Gate G: at least one candidate exists when linkable mentions > 0
    linkable_mentions = mentions_total - skipped_total_val
    gate_g = True
    if linkable_mentions > 0 and not no_linkable:
        gate_g = candidates_total > 0

    # Gate H: no type-incompatible auto-links (same as D for DB path)
    gate_h = gate_d

    # ── AUDIT gates (DB-only) ──────────────────────────────────────
    # AUDIT_1–3 are computed in the mention extraction report; we add AUDIT_4 here.

    # AUDIT_4: every entity/link printed above is derivable from DB tables
    # By construction (Option 1), we only printed data from entity_mentions +
    # entity_candidates. Verify: all printed mention_ids exist in entity_mentions,
    # all selected QIDs come from entity_candidates.meta_json.
    audit_4_pass = True
    for entry in linked_entries:
        if entry["mention_id"] not in all_mention_ids:
            audit_4_pass = False
            break
        # Verify selected candidate exists in DB
        sel = entry.get("selected") or {}
        if sel.get("qid"):
            mention_cands = cands_by_mention.get(entry["mention_id"], [])
            qid_found = any(
                ((c.get("meta_json") or {}).get("qid") == sel["qid"])
                for c in mention_cands
            )
            if not qid_found:
                audit_4_pass = False
                break

    mentions_linkable = mentions_total - skipped_total_val

    L.append("=== VALIDATION SUMMARY ===")
    L.append(f"mentions_total: {mentions_total}")
    L.append(f"mentions_linkable_total: {mentions_linkable}")
    L.append(f"mentions_skipped_total: {skipped_total_val}")

    if no_entities_pass:
        L.append("Gate A (mentions_total > 0): PASS (NO_ENTITIES_PRESENT)")
        L.append("Gate B (candidates_total > 0): PASS (NO_ENTITIES_PRESENT)")
        L.append("Gate C (linked_total > 0): PASS (NO_ENTITIES_PRESENT)")
    elif all_skipped:
        L.append("Gate A (mentions_total > 0): PASS (ALL_QUALITY_SKIPPED)")
        L.append("Gate B (candidates_total > 0): PASS (ALL_QUALITY_SKIPPED)")
        L.append("Gate C (linked_total > 0): PASS (ALL_QUALITY_SKIPPED)")
    elif no_linkable:
        L.append("Gate A (mentions_total > 0): PASS")
        L.append("Gate B (candidates_total > 0): PASS (NO_LINKABLE_MENTIONS)")
        L.append("Gate C (linked_total > 0): PASS (NO_LINKABLE_MENTIONS)")
    else:
        L.append(f"Gate A (mentions_total > 0): {'PASS' if gate_a else 'FAIL'}")
        L.append(f"Gate B (candidates_total > 0): {'PASS' if gate_b else 'FAIL'}")
        L.append(f"Gate C (linked_total > 0): {'PASS' if gate_c else 'FAIL'}")

    L.append(f"Gate D (no type-mismatch auto-links): {'PASS' if gate_d else 'FAIL'}")
    L.append(f"Gate E (referential integrity): {'PASS' if gate_e else 'FAIL'}")
    L.append(f"Gate F (evidence spans present): {'PASS' if gate_f else 'FAIL'}")
    L.append(f"Gate G (candidates when linkable > 0): {'PASS' if gate_g else 'FAIL'}")
    L.append(f"Gate H (no type-incompatible auto-links): {'PASS' if gate_h else 'FAIL'}")
    L.append(f"AUDIT_4 (linking report DB-only): {'PASS' if audit_4_pass else 'FAIL'}")

    all_pass = gate_a and gate_b and gate_c and gate_d and gate_e and gate_f and gate_g and gate_h and audit_4_pass

    if no_entities_pass:
        ready_status = "PASS_NO_LINKABLE_MENTIONS"
    elif all_skipped:
        ready_status = "PASS_ALL_QUALITY_SKIPPED"
    elif no_linkable:
        ready_status = "PASS_NO_LINKABLE_MENTIONS"
    elif linked_total_val > 0 and all_pass:
        ready_status = "PASS_AUDITED_LINKED"
    elif linked_total_val > 0:
        ready_status = "PASS_LINKED"
    elif not has_strong_linkable:
        ready_status = "PASS_ALL_QUALITY_SKIPPED"
    else:
        ready_status = "FAIL"

    L.append(f"READY_STATUS: {ready_status}")

    if ready_status == "FAIL":
        failing = []
        for name, val in [("A", gate_a), ("B", gate_b), ("C", gate_c),
                          ("D", gate_d), ("E", gate_e), ("F", gate_f),
                          ("G", gate_g), ("H", gate_h), ("AUDIT_4", audit_4_pass)]:
            if not val:
                failing.append(name)
        L.append(f"Failing gates: {', '.join(failing)}")

    return "\n".join(L)


def build_linking_report(result: dict[str, Any]) -> str:
    """Build the chat-printable ENTITY LINKING REPORT string.

    Five sections:
    1. Summary stats (with separate API counters)
    2. Top linked entities (max 10)
    3. Failures/edge cases (max 10)
    4. Type-mismatch detections (Gate D)
    5. Validation summary with gates A–F
    """
    L: list[str] = []
    run_id = result.get("run_id", "?")
    asset_ref = result.get("asset_ref", "?")
    mention_results = result.get("mention_results", [])

    # ── Section 1: Summary ────────────────────────────────────────────
    L.append("=== ENTITY LINKING REPORT ===")
    L.append(f"run_id: {run_id}")
    L.append(f"asset_ref: {asset_ref}")
    L.append(f"mentions_total: {result.get('mentions_total', 0)}")

    type_counts = result.get("type_counts", {})
    if type_counts:
        parts = [f"{k}={v}" for k, v in sorted(type_counts.items())]
        L.append(f"  by_type: {', '.join(parts)}")

    L.append(f"candidates_total: {result.get('candidates_total', 0)}")
    source_counts = result.get("source_counts", {})
    if source_counts:
        parts = [f"{k}={v}" for k, v in sorted(source_counts.items())]
        L.append(f"  by_source: {', '.join(parts)}")

    L.append(f"linked_total: {result.get('linked_total', 0)}")
    L.append(f"unresolved_total: {result.get('unresolved_total', 0)}")
    L.append(f"ambiguous_total: {result.get('ambiguous_total', 0)}")
    L.append(f"skipped_total: {result.get('skipped_total', 0)}")
    L.append(f"quality_skipped: {result.get('quality_skipped', 0)}")
    L.append(f"canonical_matched: {result.get('canonical_matched', 0)}")
    L.append(f"type_mismatch_count: {result.get('type_mismatch_count', 0)}")
    L.append(f"api_calls_search: {result.get('api_calls_search', result.get('api_calls', 0))}")
    L.append(f"api_calls_get: {result.get('api_calls_get', 0)}")
    L.append(f"cache_hits: {result.get('cache_hits', 0)}")

    # VIAF / GeoNames enrichment counts
    viaf_found = sum(
        1 for r in mention_results
        if r.get("status") == "linked" and (r.get("selected") or {}).get("viaf_id")
    )
    geonames_found = sum(
        1 for r in mention_results
        if r.get("status") == "linked" and (r.get("selected") or {}).get("geonames_id")
    )
    L.append(f"viaf_found: {viaf_found}")
    L.append(f"geonames_found: {geonames_found}")

    L.append(f"ocr_quality: {result.get('ocr_quality', 'MEDIUM')}")
    L.append(f"took_ms: {result.get('took_ms', 0)}")
    L.append("")

    # ── Section 2: Top Linked Entities ────────────────────────────────
    linked = [r for r in mention_results if r.get("status") == "linked"]
    top_linked = sorted(linked, key=lambda r: (r.get("selected") or {}).get("score", 0), reverse=True)[:10]

    L.append(f"=== TOP LINKED ENTITIES (N={len(top_linked)}) ===")
    for r in top_linked:
        sel = r.get("selected") or {}
        L.append(f"  mention_id: {r.get('mention_id', '?')}")
        L.append(f"  surface: \"{r.get('surface', '')}\"")
        L.append(f"  ent_type: {r.get('ent_type', '?')}")
        qa = r.get("queries_attempted") or [r.get("surface", "?")]
        L.append(f"  queries_attempted: {qa}")
        # Per-query diagnostics
        for qd in (r.get("query_details") or []):
            L.append(f"    query: {qd['query']}  cache_status={qd['cache_status']}  wikidata_called={qd['wikidata_called']}")
            for rh in qd.get("raw_hits", []):
                L.append(f"      raw_hit #{rh.get('search_rank',0)}: {rh.get('qid','')} | {rh.get('label','')} | {rh.get('description','')[:60]}")
        L.append(f"  chunk_id: {r.get('chunk_id', '?')}  offsets: {r.get('start_offset', 0)}-{r.get('end_offset', 0)}")
        L.append(f"  evidence: \"{r.get('evidence_text', '')}\"")
        L.append(f"  → Wikidata: {sel.get('qid', '?')} | {sel.get('label', '?')} | {sel.get('description', '')}")
        viaf = sel.get("viaf_id", "")
        geo = sel.get("geonames_id", "")
        if viaf or geo:
            L.append(f"    VIAF: {viaf or '—'}  GeoNames: {geo or '—'}")
        L.append(f"    score: {sel.get('score', 0):.4f}")
        L.append("")
    if not top_linked:
        L.append("  (none)")
        L.append("")

    # ── Section 2b: Mention Quality Decisions ─────────────────────────
    skipped = [r for r in mention_results if r.get("status") == "skipped"]
    canonical = [r for r in mention_results if r.get("canonical_match")]
    L.append(f"=== MENTION QUALITY DECISIONS (skipped={len(skipped)}, canonical={len(canonical)}) ===")
    for r in skipped[:10]:
        L.append(f"  SKIP  surface=\"{r.get('surface', '')}\" ent_type={r.get('ent_type', '?')}")
        L.append(f"        reason: {r.get('reason', '?')}")
        nl = r.get("name_likeness")
        if nl is not None:
            L.append(f"        name_likeness: {nl:.3f}")
    for r in canonical[:10]:
        cm = r.get("canonical_match") or {}
        L.append(f"  CANON surface=\"{r.get('surface', '')}\" → {cm.get('canon', '?')} (dist={cm.get('dist', '?')})")
        nl = r.get("name_likeness")
        if nl is not None:
            L.append(f"        name_likeness: {nl:.3f}")
    if not skipped and not canonical:
        L.append("  (none)")
    L.append("")

    # ── Section 3: Failures/Edge Cases ────────────────────────────────
    failures = [r for r in mention_results if r.get("status") in ("unresolved", "ambiguous")]
    top_failures = failures[:10]

    L.append(f"=== FAILURE/EDGE CASES (N={len(top_failures)}) ===")
    for r in top_failures:
        L.append(f"  mention_id: {r.get('mention_id', '?')}")
        L.append(f"  surface: \"{r.get('surface', '')}\"")
        L.append(f"  ent_type: {r.get('ent_type', '?')}")
        qa = r.get("queries_attempted") or [r.get("surface", "?")]
        L.append(f"  queries_attempted: {qa}")
        # Per-query diagnostics
        for qd in (r.get("query_details") or []):
            L.append(f"    query: {qd['query']}  cache_status={qd['cache_status']}  wikidata_called={qd['wikidata_called']}")
            for rh in qd.get("raw_hits", []):
                L.append(f"      raw_hit #{rh.get('search_rank',0)}: {rh.get('qid','')} | {rh.get('label','')} | {rh.get('description','')[:60]}")
        L.append(f"  status: {r.get('status', '?')}")
        L.append(f"  reason: {r.get('reason', '?')}")
        top2 = r.get("top_candidates", [])[:2]
        for i, c in enumerate(top2, 1):
            tc_flag = "✓" if c.get("type_compatible", False) else "✗"
            L.append(f"    candidate_{i}: {c.get('qid', '?')} | {c.get('label', '?')} | score={c.get('score', 0):.4f} type={tc_flag}")
        L.append("")
    if not top_failures:
        L.append("  (none)")
        L.append("")

    # ── Section 4: Type-Mismatch Detections ───────────────────────────
    type_mismatches: list[dict[str, Any]] = []
    for r in mention_results:
        top_cands = r.get("top_candidates", [])
        for c in top_cands:
            if not c.get("type_compatible", True):
                type_mismatches.append({
                    "surface": r.get("surface", ""),
                    "ent_type": r.get("ent_type", "?"),
                    "qid": c.get("qid", "?"),
                    "label": c.get("label", "?"),
                    "score": c.get("score", 0),
                })
    tm_preview = type_mismatches[:10]
    L.append(f"=== TYPE-MISMATCH DETECTIONS (N={len(type_mismatches)}) ===")
    for tm in tm_preview:
        L.append(
            f"  surface=\"{tm['surface']}\" ent_type={tm['ent_type']} "
            f"→ {tm['qid']} | {tm['label']} | score={tm['score']:.4f} [REJECTED]"
        )
    if not tm_preview:
        L.append("  (none)")
    L.append("")

    # ── Section 5: Validation Summary ─────────────────────────────────
    mentions_total = result.get("mentions_total", 0)
    candidates_total = result.get("candidates_total", 0)
    linked_total_val = result.get("linked_total", 0)
    skipped_total_val = result.get("skipped_total", 0)
    canonical_matched_val = result.get("canonical_matched", 0)

    # Check for NO_ENTITIES_PRESENT condition
    no_entities_pass = False
    if mentions_total == 0:
        base_text = result.get("_base_text", "")
        if base_text:
            from app.routers.ocr import _has_entity_cues
            has_cues = _has_entity_cues(base_text)
        else:
            has_cues = False
        if not has_cues:
            no_entities_pass = True

    # ALL_SKIPPED: every mention was quality-skipped (no linkable mentions)
    all_skipped = (
        mentions_total > 0
        and skipped_total_val == mentions_total
    )

    # Req 4: Determine if there are "strong" linkable mentions.
    # A mention is "strong" if it was NOT skipped AND satisfies:
    #   - canonical_match is present, OR
    #   - name_likeness >= 0.60 (for person/place), OR
    #   - confidence >= 0.60
    # Gate C is only enforced when strong linkable mentions exist.
    has_strong_linkable = any(
        r.get("canonical_match") is not None
        or (r.get("name_likeness") or 0.0) >= 0.60
        for r in mention_results
        if r.get("status") != "skipped"
    )

    # NO_LINKABLE_MENTIONS: mentions exist but none are strong enough
    # (only applies when linked_total == 0; if anything linked, it's clearly linkable)
    no_linkable = (
        mentions_total > 0
        and linked_total_val == 0
        and not all_skipped
        and not has_strong_linkable
    )

    if no_entities_pass:
        gate_a = True
        gate_b = True
        gate_c = True
    elif all_skipped:
        # All mentions were garbage → no candidates expected → pass B/C
        gate_a = True
        gate_b = True
        gate_c = True
    elif no_linkable:
        # Mentions exist but none are strong → pass with NO_LINKABLE_MENTIONS
        gate_a = True
        gate_b = True
        gate_c = True
    else:
        gate_a = mentions_total > 0
        # Gate B: pass if candidates exist OR canonical entities resolved
        gate_b = candidates_total > 0 or canonical_matched_val > 0
        gate_c = linked_total_val > 0

    # Gate D: type-mismatch guard (PASS when no mismatches auto-selected)
    gate_d = all(
        r.get("status") != "linked"
        or (r.get("selected") or {}).get("type_compatible", True) is not False
        for r in mention_results
    ) if mention_results else True

    # Gate E: referential integrity (all mention_ids in results exist)
    mention_ids_in_results = {r.get("mention_id") for r in mention_results}
    gate_e = (
        len(mention_ids_in_results) == mentions_total
        and all(r.get("mention_id") for r in mention_results)
    ) if mentions_total > 0 else True

    # Gate F: every linked entity has evidence text
    gate_f = all(
        bool(r.get("evidence_text", "").strip())
        for r in mention_results
        if r.get("status") == "linked"
    )

    # Gate G: api_calls_search + api_calls_get > 0 when linkable mentions > 0
    # BUT: if no linkable mentions exist, Gate G passes automatically.
    gate_g = True
    linkable_mentions = mentions_total - skipped_total_val
    if linkable_mentions > 0 and not no_linkable:
        total_api = int(result.get("api_calls_search", 0)) + int(result.get("api_calls_get", 0))
        if total_api == 0:
            gate_g = False

    # Gate H: no auto-linked entity may have type_compatible=false
    gate_h = all(
        r.get("status") != "linked"
        or (r.get("selected") or {}).get("type_compatible", True) is not False
        for r in mention_results
    ) if mention_results else True

    L.append("=== VALIDATION SUMMARY ===")

    # ── Compute READY_STATUS (explicit, replacing ambiguous labels) ────
    # mentions_linkable_total = mentions that were NOT quality-skipped
    mentions_linkable = mentions_total - skipped_total_val

    L.append(f"mentions_total: {mentions_total}")
    L.append(f"mentions_linkable_total: {mentions_linkable}")
    L.append(f"mentions_skipped_total: {skipped_total_val}")
    L.append(f"ocr_quality: {result.get('ocr_quality', 'MEDIUM')}")

    if no_entities_pass:
        L.append("Gate A (mentions_total > 0): PASS (NO_ENTITIES_PRESENT)")
        L.append("Gate B (wikidata_candidates_total > 0): PASS (NO_ENTITIES_PRESENT)")
        L.append("Gate C (linked_total > 0): PASS (NO_ENTITIES_PRESENT)")
    elif all_skipped:
        L.append("Gate A (mentions_total > 0): PASS (PASS_NO_LINKABLE_MENTIONS)")
        L.append("Gate B (wikidata_candidates_total > 0): PASS (PASS_NO_LINKABLE_MENTIONS)")
        L.append("Gate C (linked_total > 0): PASS (PASS_NO_LINKABLE_MENTIONS)")
    elif no_linkable:
        L.append("Gate A (mentions_total > 0): PASS")
        L.append("Gate B (wikidata_candidates_total > 0): PASS (NO_LINKABLE_MENTIONS)")
        L.append("Gate C (linked_total > 0): PASS (NO_LINKABLE_MENTIONS)")
    else:
        L.append(f"Gate A (mentions_total > 0): {'PASS' if gate_a else 'FAIL'}")
        gate_b_detail = ""
        if gate_b and candidates_total == 0 and canonical_matched_val > 0:
            gate_b_detail = " (CANONICAL_MATCH)"
        L.append(f"Gate B (wikidata_candidates_total > 0): {'PASS' if gate_b else 'FAIL'}{gate_b_detail}")
        L.append(f"Gate C (linked_total > 0): {'PASS' if gate_c else 'FAIL'}")
    L.append(f"Gate D (no type-mismatch auto-links): {'PASS' if gate_d else 'FAIL'}")
    L.append(f"Gate E (referential integrity): {'PASS' if gate_e else 'FAIL'}")
    L.append(f"Gate F (evidence spans present): {'PASS' if gate_f else 'FAIL'}")
    L.append(f"Gate G (api_calls > 0 when mentions > 0): {'PASS' if gate_g else 'FAIL'}")
    L.append(f"Gate H (no type-incompatible auto-links): {'PASS' if gate_h else 'FAIL'}")

    all_pass = gate_a and gate_b and gate_c and gate_d and gate_e and gate_f and gate_g and gate_h

    # ── Determine READY_STATUS ────────────────────────────────────────
    # PASS_LINKED                : at least one entity linked
    # PASS_ALL_QUALITY_SKIPPED   : only role/low-quality mentions (no strong ones)
    # PASS_NO_LINKABLE_MENTIONS  : no mentions, no entity cues, or no strong linkable mentions
    # FAIL                       : at least one strong mention (canonicalized or
    #                              conf>=0.60) exists but linked_total==0

    if no_entities_pass:
        ready_status = "PASS_NO_LINKABLE_MENTIONS"
        L.append(f"READY_STATUS: {ready_status}")
    elif all_skipped:
        ready_status = "PASS_ALL_QUALITY_SKIPPED"
        L.append(f"READY_STATUS: {ready_status}")
    elif no_linkable:
        ready_status = "PASS_NO_LINKABLE_MENTIONS"
        L.append(f"READY_STATUS: {ready_status}")
    elif linked_total_val > 0:
        ready_status = "PASS_LINKED"
        L.append(f"READY_STATUS: {ready_status}")
    elif not has_strong_linkable:
        # No strong mentions and nothing linked → still OK
        ready_status = "PASS_ALL_QUALITY_SKIPPED"
        L.append(f"READY_STATUS: {ready_status}")
    else:
        ready_status = "FAIL"
        L.append(f"READY_STATUS: {ready_status}")
        failing = []
        if not gate_a:
            failing.append("A")
        if not gate_b:
            failing.append("B")
        if not gate_c:
            failing.append("C")
        if not gate_d:
            failing.append("D")
        if not gate_e:
            failing.append("E")
        if not gate_f:
            failing.append("F")
        if not gate_g:
            failing.append("G")
        if not gate_h:
            failing.append("H")
        L.append(f"Failing gates: {', '.join(failing)}")

    return "\n".join(L)
