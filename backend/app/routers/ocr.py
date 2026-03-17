from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from langdetect import detect_langs

from app.agents.crop_agent import decode_image_bytes
from app.agents.ocr_agent import OCRAgentError, OcrAgent
from app.agents.ocr_proofreader_agent import OcrProofreaderAgent
from app.agents.saia_ocr_agent import SaiaOCRAgent, SaiaOCRAgentError
from app.agents.segmentation_agent import run_single_segmentation
from app.core.constants import FINAL_CLASSES
from app.db.pipeline_db import (
    clear_analysis_for_run,
    count_chunks,
    count_entity_mentions,
    create_run,
    get_run,
    insert_chunks,
    insert_entity_attempts,
    insert_entity_candidates,
    insert_entity_decisions,
    insert_entity_mentions,
    list_chunks,
    list_entity_attempts_for_run,
    list_entity_decisions,
    list_entity_mentions,
    list_events,
    log_event,
    make_span_key,
    table_view_for_chunks,
    table_view_for_entity_attempts,
    table_view_for_entity_candidates,
    table_view_for_entity_decisions,
    table_view_for_entity_mentions,
    table_view_for_events,
    table_view_for_run,
    update_run_fields,
)
from app.routers.predict import _prepare_image_for_segmentation
from app.schemas.agents_ocr import OCRExtractAnyResponse, OCRExtractRequest
from app.schemas.agents_ocr import (
    SaiaFullPageExtractRequest,
    SaiaFullPageExtractResponse,
    SaiaOCRLocationSuggestion,
    SaiaOCRRequest,
    SaiaOCRResponse,
)
from app.config import settings as _app_settings
from app.services.saia_client import SaiaConfigError
from app.services.lexicon_trust import lexical_plausibility as _lexical_plausibility
from app.services.ocr_quality import (
    compute_quality_report,
    check_mention_recall,
    format_quality_report_summary,
    OCRQualityReport,
    EffectiveQuality,
    build_effective_quality,
    compute_cross_pass_stability,
    apply_uncertainty_markers,
)
from app.services.ocr_quality_config import (
    CROSS_PASS_STABILITY_MIN,
    LEADING_FRAG_HARD_LIMIT,
    MAX_OCR_ATTEMPTS as _MAX_OCR_ATTEMPTS,
    SEAM_FRAG_HARD_LIMIT,
    frag_gate_value,
)
from app.services.pipeline_hardening import (
    decide_downstream_mode,
    enforce_quality_gates,
    extract_high_recall_mentions,
    format_gate_report,
    proofreading_quality_guard,
    should_use_shape_based_search,
)
from app.services.seam_strategies import (
    TilingPlan,
    default_plan_from_suggestions,
    is_noop_retry,
    plan_to_suggestions,
    select_retry_strategy,
)
from app.db.pipeline_db import (
    insert_ocr_quality_report,
    list_ocr_quality_reports,
    insert_ocr_attempt,
    list_ocr_attempts,
    get_best_ocr_attempt,
)

router = APIRouter(tags=["ocr"])
_ocr_agent_instance: OcrAgent | None = None
_saia_ocr_agent_instance: SaiaOCRAgent | None = None


def _get_ocr_agent() -> OcrAgent:
    global _ocr_agent_instance
    if _ocr_agent_instance is None:
        _ocr_agent_instance = OcrAgent()
    return _ocr_agent_instance


def _get_saia_ocr_agent() -> SaiaOCRAgent:
    global _saia_ocr_agent_instance
    if _saia_ocr_agent_instance is None:
        _saia_ocr_agent_instance = SaiaOCRAgent()
    return _saia_ocr_agent_instance


def _auto_index_run(run_id: str) -> None:
    """Best-effort auto-index into ChromaDB after pipeline success."""
    if not _app_settings.rag_auto_index:
        return
    try:
        from app.services.rag_store import index_run as _rag_index
        log_event(run_id, "INDEX_RUNNING", "START", "Auto-indexing into vector store.")
        result = _rag_index(run_id)
        log_event(
            run_id,
            "INDEX_DONE",
            "END",
            f"Indexed {result.get('chunks_indexed', 0)} chunks.",
        )
    except Exception as exc:  # noqa: BLE001
        log_event(run_id, "INDEX_ERROR", "ERROR", f"Auto-index failed: {exc}")


def _run_authority_linking_stage(run_id: str) -> dict[str, Any] | None:
    """Best-effort authority linking (Wikidata) after STORED stage."""
    try:
        from app.services.authority_linking import run_authority_linking

        log_event(run_id, "AUTHORITY_LINKING_RUNNING", "START", "Running authority linking (Wikidata).")
        update_run_fields(run_id, current_stage="AUTHORITY_LINKING_RUNNING")
        result = run_authority_linking(run_id)
        log_event(
            run_id,
            "AUTHORITY_LINKING_DONE",
            "END",
            f"Authority linking complete: linked={result.get('linked_total', 0)}, "
            f"unresolved={result.get('unresolved_total', 0)}, "
            f"ambiguous={result.get('ambiguous_total', 0)}.",
        )
        update_run_fields(run_id, current_stage="AUTHORITY_LINKING_DONE")
        return result
    except Exception as exc:  # noqa: BLE001
        log_event(run_id, "AUTHORITY_LINKING_ERROR", "ERROR", f"Authority linking failed: {exc}")
        return None

_TEXT_LABEL_INCLUDE_TOKENS = (
    "main script",
    "variant script",
    "plain initial",
    "historiated",
    "inhabited",
    "embellished",
    "dropcapital",
    "defaultline",
)
_TEXT_LABEL_EXCLUDE_TOKENS = (
    "border",
    "column",
    "table",
    "diagram",
    "illustration",
    "graphic",
    "music",
    "zone",
    "gloss",
    "header",
    "catchword",
    "page number",
    "quire",
)

_PARTIAL_WARNING_KEYS = {
    "LOW_TEXT_QUALITY",
    "INVALID_OCR_JSON",
    "JSON_REPAIR_RETRY",
    "SCRIPT_DRIFT",
    "HALLUCINATION_SUSPECTED",
    "OCR_FAILED_ALL_MODELS",
    "PROOFREAD_FAILED",
    "PROOFREAD_EMPTY_RESULT",
    "PROOFREAD_SKIPPED_LOW_QUALITY",
    "SEGMENTATION_FAILED",
}
_ALLOWED_DETECTED_LANGUAGES = {
    "latin",
    "old_english",
    "middle_english",
    "french",
    "old_french",
    "middle_french",
    "anglo_norman",
    "occitan",
    "old_high_german",
    "middle_high_german",
    "german",
    "dutch",
    "italian",
    "spanish",
    "portuguese",
    "catalan",
    "church_slavonic",
    "greek",
    "hebrew",
    "arabic",
    "mixed",
    "unknown",
}
_OLD_ENGLISH_MARKERS_RE = re.compile(r"[þðƿÞÐǷ]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
_LATIN_TOKEN_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿĀ-žſƀ-ɏ]+$")
_LANGID_MIN_LETTERS = 50
_LATIN_VOWELS = set("aeiouyàâäæéèêëîïôöœùûüÿ")
_FRENCH_STRONG_ANCHORS = {
    "tres",
    "tresreuerend",
    "pere",
    "euesque",
    "eusque",
    "prince",
    "lausanne",
    "laufanne",
    "grace",
    "considerat",
    "desirs",
    "chapellain",
    "seruiteur",
    "approchant",
}
_FRENCH_FUNCTION_WORDS = {"que", "et", "de", "la", "le", "des"}
_FRENCH_OLD_MARKERS = {"pur", "ki", "mei", "sun", "uostre", "nostre", "cil", "cest", "dunc"}
_FRENCH_MIDDLE_MARKERS = {
    "approchant",
    "considerat",
    "grace",
    "mille",
    "cens",
    "chapellain",
    "seruiteur",
    "desirs",
    "lausanne",
    "prince",
    "pere",
}
_ANGLO_NORMAN_MARKERS = {"anglo", "norman", "normand", "engleterre"}
_LATIN_STRONG_ANCHORS = {
    "domini",
    "amen",
    "quod",
    "canonici",
    "capitulum",
    "instrumentum",
    "publicum",
    "consensu",
    "confirmamus",
    "perpetuo",
}
_MENTION_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]*")
_PERSON_PARTICLES = {"de", "d'", "d’", "du", "des"}
_PLACE_ANCHORS = {"lausanne", "paris", "lyon", "rome", "avignon", "geneve", "genève"}
_TITLE_ANCHORS = {"euesque", "évesque", "evesque", "prince", "chapellain", "pere"}
_DATE_PATTERNS = (
    re.compile(r"\b(?:l['’]an|lan|an de grace)\b(?:\s+[A-Za-z0-9IVXLCDM]+){0,4}", re.IGNORECASE),
    re.compile(r"\b(?:mil|mille)\b(?:\s+[A-Za-z0-9IVXLCDM]+){0,4}", re.IGNORECASE),
    re.compile(r"\b1[0-9]{3}\b"),
    re.compile(r"\bM{1,4}[CDLXVI]{2,}\b"),
)

# ── Date anchor validation ────────────────────────────────────────────
# A span may be classified as DATE only if it contains at least one
# strong anchor.  "mil" or "cent" alone without anchors are rejected.
_DATE_STRONG_ANCHORS: set[str] = {
    "an", "l'an", "l’an", "lan", "en l'an", "en l’an",
    "ans", "apres", "après", "avant",
}
_MONTH_NAMES: set[str] = {
    "janvier", "fevrier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "aout", "septembre", "octobre", "novembre",
    "decembre", "décembre",
    "januarii", "februarii", "martii", "aprilis", "maii", "junii",
    "julii", "augusti", "septembris", "octobris", "novembris", "decembris",
}
_DATE_FORMULA_RE = re.compile(
    r"\b(?:l['’']an|lan|an\s+de\s+grace)\s+(?:mil|mille)\b",
    re.IGNORECASE,
)
_NUMERIC_YEAR_RE = re.compile(r"\b[0-9]{3,4}\b")
_ROMAN_YEAR_RE = re.compile(r"\bM{1,4}[CDLXVI]{2,}\b")

# ── Editorial / philology blacklist (non-entity markers) ──────────────
_EDITORIAL_BLACKLIST: set[str] = {
    "lacune", "lacuna", "lacunae",
    "[...]", "…", "illegible", "gap", "missing",
    "illisible", "effacé", "rature",
    "trou", "déchirure", "dechirure",
}


def _has_date_anchor(span: str) -> bool:
    """Return True if *span* contains a strong date anchor.

    Strong anchors:
      - One of _DATE_STRONG_ANCHORS tokens
      - A numeric year pattern (3-4 digits)
      - A roman-numeral year pattern (e.g. MCLXIIII)
      - A month name
      - A full date formula like "l'an mil ..."
    """
    low = span.lower()
    # Check full date formula ("l'an mil ...") first
    if _DATE_FORMULA_RE.search(low):
        return True
    # Numeric year
    if _NUMERIC_YEAR_RE.search(span):
        return True
    # Roman-numeral year
    if _ROMAN_YEAR_RE.search(span):
        return True
    # Tokenize and check anchors / month names
    tokens = re.findall(r"[A-Za-z\u00c0-\u00ff'’]+", low)
    for tok in tokens:
        if tok in _DATE_STRONG_ANCHORS:
            return True
        if tok in _MONTH_NAMES:
            return True
    # Check bigrams for multi-word anchors ("en l'an", "an de grace")
    text_joined = " ".join(tokens)
    for anchor in _DATE_STRONG_ANCHORS:
        if " " in anchor and anchor in text_joined:
            return True
    return False


def _norm_surface(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _is_roman_numeral_token(token: str) -> bool:
    value = str(token or "").upper()
    return bool(value) and re.fullmatch(r"[IVXLCDM]+", value) is not None


# Chunks that are pure stopword / single-char are not useful for analysis.
_CHUNK_STOPWORDS: set[str] = {
    "a", "e", "i", "o", "u", "y",
    "et", "de", "la", "le", "en", "il", "ne", "si", "ce",
}


def _build_line_chunks(base_text: str) -> list[dict[str, Any]]:
    value = str(base_text or "")
    if not value:
        return []
    chunks: list[dict[str, Any]] = []
    cursor = 0
    idx = 0
    for line in value.splitlines():
        start = cursor
        end = start + len(line)
        stripped = line.strip()
        if stripped:
            # Req 4: drop chunks that are pure stopword / single-char
            if len(stripped) < 2 or stripped.lower() in _CHUNK_STOPWORDS:
                cursor = end + 1
                continue
            chunks.append(
                {
                    "idx": idx,
                    "start_offset": start,
                    "end_offset": end,
                    "text": line,
                }
            )
            idx += 1
        cursor = end + 1
    if not chunks and value.strip():
        chunks.append({"idx": 0, "start_offset": 0, "end_offset": len(value), "text": value})
    return chunks


def _assign_chunk_id_for_span(
    start: int,
    end: int,
    chunk_rows: list[dict[str, Any]],
) -> str | None:
    for chunk in chunk_rows:
        c_start = int(chunk.get("start_offset", 0))
        c_end = int(chunk.get("end_offset", 0))
        if start >= c_start and end <= c_end:
            return str(chunk.get("chunk_id") or "")
    for chunk in chunk_rows:
        c_start = int(chunk.get("start_offset", 0))
        c_end = int(chunk.get("end_offset", 0))
        if start < c_end and end > c_start:
            return str(chunk.get("chunk_id") or "")
    return None


def _extract_person_mentions(base_text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    lines = base_text.splitlines()
    cursor = 0
    for line in lines:
        token_matches = list(_MENTION_TOKEN_RE.finditer(line))
        i = 0
        while i < len(token_matches):
            first = token_matches[i]
            token = first.group(0)
            if not token[:1].isupper() or _is_roman_numeral_token(token):
                i += 1
                continue
            best_end_idx = -1
            best_non_particle = 0
            used_particle = False
            for j in range(i + 1, min(len(token_matches), i + 7)):
                prev_end = token_matches[j - 1].end()
                cur_start = token_matches[j].start()
                between = line[prev_end:cur_start]
                if not re.fullmatch(r"[\s,.;:()\-–—]*", between):
                    break
                token_j = token_matches[j].group(0)
                token_j_norm = token_j.lower()
                if token_j_norm in _PERSON_PARTICLES:
                    used_particle = True
                    continue
                if not token_j[:1].isupper() or _is_roman_numeral_token(token_j):
                    break
                non_particle_tokens = 0
                for k in range(i, j + 1):
                    tok_k = token_matches[k].group(0).lower()
                    if tok_k not in _PERSON_PARTICLES:
                        non_particle_tokens += 1
                if 2 <= non_particle_tokens <= 5:
                    best_end_idx = j
                    best_non_particle = non_particle_tokens
            if best_end_idx == -1:
                i += 1
                continue
            local_start = token_matches[i].start()
            local_end = token_matches[best_end_idx].end()
            start = cursor + local_start
            end = cursor + local_end
            surface = base_text[start:end]
            conf = 0.65 + min(0.15, 0.05 * max(0, best_non_particle - 2))
            if used_particle:
                conf = min(0.85, conf + 0.05)
            mentions.append(
                {
                    "start_offset": start,
                    "end_offset": end,
                    "surface": surface,
                    "norm": _norm_surface(surface),
                    "label": None,
                    "ent_type": "person",
                    "confidence": conf,
                    "method": "rule:proper_name_sequence",
                    "notes": None,
                }
            )
            i = best_end_idx + 1
        cursor += len(line) + 1
    return mentions


def _extract_anchor_mentions(base_text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    lowered = base_text.lower()

    for place in sorted(_PLACE_ANCHORS):
        pattern = re.compile(rf"\b{re.escape(place)}\b", re.IGNORECASE)
        for match in pattern.finditer(lowered):
            start = match.start()
            end = match.end()
            surface = base_text[start:end]
            mentions.append(
                {
                    "start_offset": start,
                    "end_offset": end,
                    "surface": surface,
                    "norm": _norm_surface(surface),
                    "label": None,
                    "ent_type": "place",
                    "confidence": 0.84,
                    "method": "rule:toponym_anchor",
                    "notes": None,
                }
            )

    for title in sorted(_TITLE_ANCHORS):
        pattern = re.compile(rf"\b{re.escape(title)}\b", re.IGNORECASE)
        for match in pattern.finditer(lowered):
            start = match.start()
            end = match.end()
            surface = base_text[start:end]
            mentions.append(
                {
                    "start_offset": start,
                    "end_offset": end,
                    "surface": surface,
                    "norm": _norm_surface(surface),
                    "label": None,
                    "ent_type": "title",
                    "confidence": 0.64,
                    "method": "rule:title_anchor",
                    "notes": None,
                }
            )

    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(base_text):
            start = match.start()
            end = match.end()
            surface = base_text[start:end].strip()
            if not surface:
                continue
            # ── Req 1: Validate date anchor presence ──────────────
            if not _has_date_anchor(surface):
                continue
            # Reject spans that cross a newline unless an anchor
            # appears on BOTH sides or the full phrase matches a
            # known date formula.
            if "\n" in base_text[start:end]:
                parts = base_text[start:end].split("\n")
                if not _DATE_FORMULA_RE.search(base_text[start:end].lower()):
                    left_ok = _has_date_anchor(parts[0])
                    right_ok = _has_date_anchor(parts[-1])
                    if not (left_ok and right_ok):
                        continue
            mentions.append(
                {
                    "start_offset": start,
                    "end_offset": end,
                    "surface": base_text[start:end],
                    "norm": _norm_surface(surface),
                    "label": None,
                    "ent_type": "date",
                    "confidence": 0.72,
                    "method": "rule:date_pattern",
                    "notes": None,
                }
            )

    first_non_empty = next((line.strip() for line in base_text.splitlines() if line.strip()), "")
    if first_non_empty and len(first_non_empty) <= 80:
        lowered_line = first_non_empty.lower()
        if lowered_line.startswith(("la ", "le ", "l'", "l’")) and len(first_non_empty.split()) >= 2:
            start = base_text.find(first_non_empty)
            if start >= 0:
                end = start + len(first_non_empty)
                mentions.append(
                    {
                        "start_offset": start,
                        "end_offset": end,
                        "surface": base_text[start:end],
                        "norm": _norm_surface(first_non_empty),
                        "label": None,
                        "ent_type": "work",
                        "confidence": 0.56,
                        "method": "rule:first_line_title_heuristic",
                        "notes": None,
                    }
                )
    return mentions


def _dedupe_mentions(base_text: str, mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for item in mentions:
        start = int(item.get("start_offset", 0))
        end = int(item.get("end_offset", 0))
        ent_type = str(item.get("ent_type") or "unknown")
        if start < 0 or end <= start or end > len(base_text):
            continue
        key = (start, end, ent_type)
        prev = by_key.get(key)
        if prev is None or float(item.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
            by_key[key] = item

    ordered = sorted(by_key.values(), key=lambda row: (-float(row.get("confidence", 0.0)), -(int(row["end_offset"]) - int(row["start_offset"]))))
    selected: list[dict[str, Any]] = []
    for item in ordered:
        start = int(item["start_offset"])
        end = int(item["end_offset"])
        ent_type = str(item.get("ent_type") or "unknown")
        overlap = False
        for existing in selected:
            if str(existing.get("ent_type") or "unknown") != ent_type:
                continue
            ex_start = int(existing["start_offset"])
            ex_end = int(existing["end_offset"])
            if start < ex_end and end > ex_start:
                overlap = True
                break
        if overlap:
            continue
        selected.append(item)

    selected.sort(key=lambda row: (int(row["start_offset"]), -float(row.get("confidence", 0.0))))
    return selected


# ── Salvage extractor constants ─────────────────────────────────────────
_SALVAGE_TRIGGERS: dict[str, str] = {
    "roi": "PERSON_OR_ROLE", "roy": "PERSON_OR_ROLE", "roya": "PERSON_OR_ROLE",
    "evesque": "PERSON_OR_ROLE", "évesque": "PERSON_OR_ROLE", "euesque": "PERSON_OR_ROLE",
    "saint": "PERSON", "sainz": "PERSON",
    "pape": "PERSON_OR_ROLE", "duc": "PERSON_OR_ROLE",
    "conte": "PERSON_OR_ROLE", "dame": "PERSON_OR_ROLE", "sire": "PERSON_OR_ROLE",
}
_CANONICAL_WORKS: dict[str, str] = {
    "lancelot": "lancelot", "arthur": "arthur", "guenievre": "guenievre",
    "tristan": "tristan", "perceval": "perceval", "graal": "graal",
    "merlin": "merlin", "galahad": "galahad",
}
# Canonical person names used by the name-heuristic check in salvage Rule 1.
# Keys are normalised forms; values are the preferred canonical spelling.
_CANONICAL_PERSON_NAMES: dict[str, str] = {
    "arthur": "arthur", "artus": "arthur", "artur": "arthur",
    "lancelot": "lancelot", "lancelo": "lancelot",
    "lancelote": "lancelot", "leantlote": "lancelot",
    "lanselot": "lancelot", "lanceloc": "lancelot",
    "guenievre": "guenievre", "guenievr": "guenievre", "guenevere": "guenievre",
    "perceval": "perceval", "parsival": "perceval", "percevall": "perceval",
    "merlin": "merlin", "merlim": "merlin",
    "galahad": "galahad", "galaad": "galahad",
    "tristan": "tristan", "tristam": "tristan",
    "gauvain": "gauvain", "gawain": "gauvain",
    "yvain": "yvain", "ywain": "yvain", "ivain": "yvain",
    "erec": "erec", "enide": "enide",
    "bohort": "bohort", "bors": "bohort",
}
_SALVAGE_STOPWORDS = {
    "et", "de", "la", "le", "les", "des", "du", "en", "a", "au", "aux", "que",
    "qui", "ne", "pas", "par", "pour", "sur", "son", "ses", "sa", "ce",
    "est", "fut", "sont", "une", "uns", "si", "il", "li", "lor", "mais",
    "car", "con", "com", "tout", "molt", "bien", "fait", "dit",
    "del", "d'", "d\u2019", "mes", "ot", "ont", "vint", "fist", "dist",
    "estoit", "avoit", "ala", "alerent", "vont", "font", "firent",
    "puis", "lors", "tant", "moult", "aussi", "donc", "dont",
}
_SALVAGE_TOKEN_RE = re.compile(r"[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u017e\u017f\u0180-\u024f]+")
# Only prepositions that reliably signal a place name in medieval French.
# Deliberately excludes "a" (too common as a verb/article) to avoid noise.
_PLACE_PREP_RE = re.compile(
    r"\b(?:en|au|aux|de\s+la|de\s+l['\u2019]|de|du|des|del|d['\u2019])\s+",
    re.IGNORECASE,
)


def _levenshtein(s: str, t: str) -> int:
    """Compute Levenshtein edit distance (no external deps)."""
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


_VOWELS = set("aeiouyàâäéèêëïîôùûüœæ")


def _looks_like_medieval_name(token: str) -> bool:
    """Heuristic: does *token* look like a plausible medieval proper name?

    Used by salvage Rule 1 to decide whether the tokens following a role
    trigger (roi, dame, …) constitute a person name or are OCR garbage.

    Checks:
      0. Reject if token is in the common-word blacklist.
      1. Length-aware canonical matching: normalized edit distance <= 0.25
         AND bigram overlap >= 0.40 for tokens >= 5 chars.
      2. Length 3–18 characters.
      3. Vowel ratio between 0.20 and 0.80.
      4. No run of ≥ 4 consecutive consonants (unusual in French names).
      5. Proximity gate for non-canonical tokens ≥ 5 chars.

    Returns ``True`` if the token passes all applicable checks.
    """
    from app.services.text_normalization import (
        is_blacklisted_token,
        normalized_edit_distance,
        bigram_overlap,
    )
    if len(token) < 3 or len(token) > 18:
        return False
    low = token.lower()

    # Gate 0: reject common words — no common word is ever a name
    if is_blacklisted_token(low):
        return False

    # Gate 1: canonical match using normalized distance + bigram overlap
    best_norm_dist = 1.0
    for canon in _CANONICAL_PERSON_NAMES:
        nd = normalized_edit_distance(low, canon)
        if nd < best_norm_dist:
            best_norm_dist = nd
        # Short tokens (< 5 chars): require exact or 1-char-off match
        if len(low) < 5:
            if nd == 0.0:  # exact match only for very short tokens
                return True
            continue
        # Tokens >= 5 chars: normalized distance + bigram overlap
        bo = bigram_overlap(low, canon)
        if nd <= 0.25 and bo >= 0.40:
            return True

    # Vowel ratio
    vowels = sum(1 for c in low if c in _VOWELS)
    ratio = vowels / len(low) if low else 0
    if ratio < 0.20 or ratio > 0.80:
        return False

    # No long consonant runs (≥ 4 consecutive non-vowels)
    consonant_run = 0
    for c in low:
        if c in _VOWELS:
            consonant_run = 0
        else:
            consonant_run += 1
            if consonant_run >= 4:
                return False

    # Proximity gate: non-canonical tokens ≥ 5 chars must be within
    # a reasonable normalized distance of at least one canonical name.
    if len(low) >= 5:
        if best_norm_dist > 0.35:
            return False

    return True


def _extract_salvage_mentions(base_text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Salvage extractor for medieval lowercase text.

    Returns ``(mentions, debug)`` where *debug* contains counts per rule
    and rejected candidates.
    """
    from app.services.text_normalization import (
        is_blacklisted_token,
        normalized_edit_distance as _ned,
        bigram_overlap as _bo,
    )
    value = str(base_text or "")
    if not value.strip():
        return [], {"trigger": 0, "work_fuzzy": 0, "place_candidate": 0, "rejected": []}

    tokens_matches = list(_SALVAGE_TOKEN_RE.finditer(value))
    tokens_lower = [(m.group(0).lower(), m.start(), m.end()) for m in tokens_matches]

    mentions: list[dict[str, Any]] = []
    debug: dict[str, Any] = {
        "trigger": 0,
        "work_fuzzy": 0,
        "place_candidate": 0,
        "rejected": [],
    }
    seen_spans: set[tuple[int, int]] = set()

    # ── Rule 1: Trigger words ─────────────────────────────────────────
    for idx, (tok_low, tok_start, tok_end) in enumerate(tokens_lower):
        label = _SALVAGE_TRIGGERS.get(tok_low)
        if label is None:
            continue
        # Capture next 1–4 non-stopword tokens as the candidate name
        name_parts: list[str] = []
        name_end = tok_end
        for j in range(1, 5):
            nxt = idx + j
            if nxt >= len(tokens_lower):
                break
            nxt_tok, nxt_start, nxt_end = tokens_lower[nxt]
            # Stop on another trigger or punctuation gap > 3 chars
            gap = value[name_end:nxt_start]
            if len(gap) > 3 or any(ch in gap for ch in '.;:!?'):
                break
            if nxt_tok in _SALVAGE_STOPWORDS or len(nxt_tok) < 2:
                break
            if nxt_tok in _SALVAGE_TRIGGERS:
                break
            name_parts.append(value[nxt_start:nxt_end])
            name_end = nxt_end
        if not name_parts:
            continue
        # Filter out blacklisted common words from name_parts
        non_blacklisted = [p for p in name_parts if not is_blacklisted_token(p)]
        if not non_blacklisted:
            debug["rejected"].append({
                "surface": " ".join(name_parts),
                "start_offset": tok_start,
                "end_offset": name_end,
                "canonical": "(blacklisted)",
                "dist": 0,
                "max_dist": 0,
                "reason": f"all name_parts blacklisted: {name_parts}",
            })
            continue
        surf_start = tok_start
        surf_end = name_end
        if (surf_start, surf_end) in seen_spans:
            continue
        seen_spans.add((surf_start, surf_end))
        surface = value[surf_start:surf_end]

        # ── Name-quality heuristic: decide person vs role ─────────
        # If *any* non-blacklisted name_part looks like a real medieval
        # name → person.  Otherwise → role (not sent for person linking).
        any_name_like = any(_looks_like_medieval_name(p) for p in non_blacklisted)
        ent_type = "person" if any_name_like else "role"
        confidence = 0.55 if any_name_like else 0.25

        if not any_name_like:
            # SKIP_NON_LINKABLE: goes into skipped (not rejected) because
            # it IS emitted as a role mention — it's not a failed candidate.
            debug.setdefault("skipped", []).append({
                "surface": surface,
                "canonical": "(person heuristic)",
                "dist": 0,
                "max_dist": 0,
                "reason": f"name_parts={name_parts} failed medieval-name heuristic → role (SKIP_NON_LINKABLE)",
            })

        mentions.append({
            "start_offset": surf_start,
            "end_offset": surf_end,
            "surface": surface,
            "norm": _norm_surface(surface),
            "label": label,
            "ent_type": ent_type,
            "confidence": confidence,
            "method": "rule:salvage_trigger",
            "notes": f"trigger={tok_low} name_quality={'name' if any_name_like else 'role'}",
        })
        debug["trigger"] += 1

    # ── Rule 2: Fuzzy work/legend matching ────────────────────────────
    for tok_low, tok_start, tok_end in tokens_lower:
        if len(tok_low) < 4:
            continue
        if is_blacklisted_token(tok_low):
            continue
        for canon, canon_norm in _CANONICAL_WORKS.items():
            # Use normalized edit distance + bigram overlap
            nd = _ned(tok_low, canon)
            # Short tokens (< 5 chars): exact match only
            if len(tok_low) < 5:
                if nd > 0.0:
                    continue
            else:
                bo = _bo(tok_low, canon)
                if nd > 0.25 or bo < 0.40:
                    # Track close misses for debug
                    if nd <= 0.40:
                        debug["rejected"].append({
                            "surface": value[tok_start:tok_end],
                            "start_offset": tok_start,
                            "end_offset": tok_end,
                            "canonical": canon,
                            "dist": round(nd, 3),
                            "max_dist": 0.25,
                            "reason": f"nd={nd:.3f}>0.25 or bo={bo:.3f}<0.40",
                        })
                    continue
            if (tok_start, tok_end) in seen_spans:
                break
            seen_spans.add((tok_start, tok_end))
            surface = value[tok_start:tok_end]
            # Compute bo for provenance (may already have it from the gate above)
            if len(tok_low) >= 5:
                bo_val = _bo(tok_low, canon)
            else:
                bo_val = 1.0  # exact match for short tokens
            mentions.append({
                "start_offset": tok_start,
                "end_offset": tok_end,
                "surface": surface,
                "norm": canon_norm,
                "label": "WORK",
                "ent_type": "work",
                "confidence": max(0.35, 0.65 - nd),
                "method": "rule:salvage_work_fuzzy",
                "notes": f"canonical={canon} nd={nd:.3f} bo={bo_val:.3f}",
            })
            debug["work_fuzzy"] += 1
            break  # matched one canonical – stop checking others

    # ── Rule 3: Place candidate (preposition + token) ─────────────────
    # Uses place_likeness gate: token must have plausible phonotactics
    # AND meet stricter length/case requirements to avoid false positives
    # from random short OCR fragments like "ament".
    from app.services.text_normalization import token_quality_score as _tqs
    for m in _PLACE_PREP_RE.finditer(value):
        prep_end = m.end()
        # Get the next token after the preposition
        next_tok_m = _SALVAGE_TOKEN_RE.search(value, prep_end)
        if next_tok_m is None or next_tok_m.start() > prep_end + 2:
            continue
        tok_text = next_tok_m.group(0)
        tok_low_val = tok_text.lower()
        # ── Req 2: editorial/philology blacklist ─────────────────
        if tok_low_val in _EDITORIAL_BLACKLIST:
            debug["rejected"].append({
                "surface": tok_text,
                "start_offset": next_tok_m.start(),
                "end_offset": next_tok_m.end(),
                "canonical": "(place)",
                "dist": 0,
                "max_dist": 0,
                "reason": f"editorial_blacklist: '{tok_low_val}' is not an entity",
            })
            continue
        # Require length >= 5, alpha-only after norm, and not a stopword
        norm_tok = _norm_surface(tok_text)
        if len(tok_text) < 5 or tok_low_val in _SALVAGE_STOPWORDS:
            debug["rejected"].append({
                "surface": tok_text,
                "start_offset": next_tok_m.start(),
                "end_offset": next_tok_m.end(),
                "canonical": "(place)",
                "dist": 0,
                "max_dist": 0,
                "reason": f"too short ({len(tok_text)}<5) or stopword",
            })
            continue
        if not norm_tok.replace(" ", "").isalpha():
            debug["rejected"].append({
                "surface": tok_text,
                "start_offset": next_tok_m.start(),
                "end_offset": next_tok_m.end(),
                "canonical": "(place)",
                "dist": 0,
                "max_dist": 0,
                "reason": "non-alpha after normalisation",
            })
            continue
        # ── place_likeness gate ──────────────────────────────────────
        # A place candidate must satisfy EITHER:
        # (a) token starts with uppercase → likely intentional name, OR
        # (b) token is a known place anchor, OR
        # (c) token length >= 6 AND good phonotactic quality
        # This prevents short lowercase OCR garbage ("ament") from
        # slipping through.
        is_known_place = tok_low_val in _PLACE_ANCHORS
        is_capitalized = tok_text[:1].isupper()
        place_quality = _tqs(tok_low_val)

        if not is_known_place and not is_capitalized:
            # Lowercase non-anchor: needs stricter gate
            if len(tok_text) < 6 or place_quality < 0.50:
                debug["rejected"].append({
                    "surface": tok_text,
                    "start_offset": next_tok_m.start(),
                    "end_offset": next_tok_m.end(),
                    "canonical": "(place)",
                    "dist": 0,
                    "max_dist": 0,
                    "reason": f"place_likeness: lowercase, len={len(tok_text)}, quality={place_quality:.3f} (need len>=6 and quality>=0.50 for non-anchor lowercase)",
                })
                continue
        elif not is_known_place:
            # Capitalized but not a known place: basic quality check
            if place_quality < 0.40:
                debug["rejected"].append({
                    "surface": tok_text,
                    "start_offset": next_tok_m.start(),
                    "end_offset": next_tok_m.end(),
                    "canonical": "(place)",
                    "dist": 0,
                    "max_dist": 0,
                    "reason": f"place_likeness={place_quality:.3f} < 0.40 (OCR garbage)",
                })
                continue
        span = (next_tok_m.start(), next_tok_m.end())
        if span in seen_spans:
            continue
        seen_spans.add(span)
        mentions.append({
            "start_offset": next_tok_m.start(),
            "end_offset": next_tok_m.end(),
            "surface": tok_text,
            "norm": norm_tok,
            "label": "PLACE_CANDIDATE",
            "ent_type": "place",
            "confidence": 0.35,
            "method": "rule:salvage_place_candidate",
            "notes": f"prep='{m.group(0).strip()}' place_quality={place_quality:.3f} is_cap={is_capitalized} is_anchor={is_known_place}",
        })
        debug["place_candidate"] += 1

    # Trim rejected list
    debug["rejected"] = debug["rejected"][:20]
    return mentions, debug


def _has_entity_cues(base_text: str) -> bool:
    """Return True if the text has any cues suggesting named entities exist.

    Cues: trigger words, title anchors, place anchors, or any token >=8 chars
    not in stopwords.
    """
    value = str(base_text or "").lower()
    if not value.strip():
        return False
    tokens = set(_SALVAGE_TOKEN_RE.findall(value))
    # Check trigger words
    if tokens & set(_SALVAGE_TRIGGERS.keys()):
        return True
    # Check title / place anchors
    if tokens & _TITLE_ANCHORS:
        return True
    if tokens & _PLACE_ANCHORS:
        return True
    # Check canonical works (exact match only for cue detection)
    if tokens & set(_CANONICAL_WORKS.keys()):
        return True
    return False


def _build_mention_extraction_report(
    run_id: str,
    asset_ref: str,
    mentions: list[dict[str, Any]],
    salvage_debug: dict[str, Any],
) -> str:
    """Build a chat-printable === MENTION EXTRACTION REPORT ===.

    Primary path (auditable): queries ``entity_decisions`` and
    ``entity_attempts`` from SQLite so every printed line is backed by a
    persisted row.  Falls back to in-memory data only when no decisions
    exist in the DB (legacy unit-test path).

    Sections (DB-backed path):
      - top_mentions_preview:  status = ACCEPT_LINKABLE
      - skipped_preview:       status = SKIP_NON_LINKABLE
      - top_rejected_preview:  status = REJECT_LINKABLE
      - filtered_out_preview:  status = FILTERED_OUT (debug-only)
      - attempts_for_accepted: from entity_attempts table
    """
    _LINKABLE_TYPES = {"person", "place", "work", "date"}
    L: list[str] = []
    L.append("=== MENTION EXTRACTION REPORT ===")
    L.append(f"run_id: {run_id}")
    L.append(f"asset_ref: {asset_ref}")

    # ── Attempt DB-backed auditable path ────────────────────────────
    decisions = list_entity_decisions(run_id)

    # ── Backfill: older runs without entity_decisions ─────────────
    if not decisions:
        db_mentions_bf = list_entity_mentions(run_id)
        if db_mentions_bf:
            import uuid as _uuid_bf
            backfill_rows: list[dict[str, Any]] = []
            for m in db_mentions_bf:
                etype = str(m.get("ent_type") or "unknown")
                surface = str(m.get("surface") or "")
                s_off = int(m.get("start_offset", 0))
                e_off = int(m.get("end_offset", 0))
                if etype in _LINKABLE_TYPES:
                    status = "ACCEPT_LINKABLE"
                    method = str(m.get("method") or "unknown")
                else:
                    status = "SKIP_NON_LINKABLE"
                    method = str(m.get("method") or "unknown")
                backfill_rows.append({
                    "decision_id": str(_uuid_bf.uuid4()),
                    "chunk_id": m.get("chunk_id"),
                    "start_offset": s_off,
                    "end_offset": e_off,
                    "surface": surface,
                    "norm": m.get("norm"),
                    "ent_type_guess": etype,
                    "label": m.get("label"),
                    "status": status,
                    "method": method,
                    "reason": f"backfill_from_entity_mentions:{m.get('mention_id', '?')}",
                    "confidence": float(m.get("confidence", 0)),
                    "meta_json": None,
                })
            if backfill_rows:
                insert_entity_decisions(run_id, backfill_rows)
                decisions = list_entity_decisions(run_id)

    if decisions:
        # DB-backed mentions_total and mentions_by_method
        db_mentions = list_entity_mentions(run_id)
        L.append(f"mentions_total: {len(db_mentions)}")
        method_counts: dict[str, int] = {}
        for m in db_mentions:
            method = str(m.get("method") or "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1
        L.append("mentions_by_method:")
        if method_counts:
            for method, count in sorted(method_counts.items()):
                L.append(f"  {method}: {count}")
        else:
            L.append("  (none)")
        # Fetch all attempts for this run (joined with decision metadata)
        all_attempts = list_entity_attempts_for_run(run_id)
        # Build lookup: decision_id → list[attempt]
        attempts_by_decision: dict[str, list[dict[str, Any]]] = {}
        for a in all_attempts:
            attempts_by_decision.setdefault(a["decision_id"], []).append(a)

        accept_decisions = [d for d in decisions if d.get("status") == "ACCEPT_LINKABLE"]
        skip_decisions = [d for d in decisions if d.get("status") == "SKIP_NON_LINKABLE"]
        reject_decisions = [d for d in decisions if d.get("status") == "REJECT_LINKABLE"]
        filtered_decisions = [d for d in decisions if d.get("status") == "FILTERED_OUT"]

        # Top linkable mentions preview (up to 10) — status = ACCEPT_LINKABLE
        L.append("")
        top_n = min(10, len(accept_decisions))
        L.append(f"top_mentions_preview (N={top_n}):")
        sorted_accept = sorted(accept_decisions, key=lambda d: -float(d.get("confidence") or 0))
        for d in sorted_accept[:top_n]:
            label = d.get("label") or "-"
            # Get canonical/nd/bo from the ACCEPT attempt
            dec_attempts = attempts_by_decision.get(d["decision_id"], [])
            accept_att = next((a for a in dec_attempts if a.get("attempt_decision") == "ACCEPT"), None)
            notes_parts: list[str] = []
            if accept_att:
                notes_parts.append(f"canonical={accept_att.get('candidate', '?')}")
                if accept_att.get("nd") is not None:
                    notes_parts.append(f"nd={accept_att['nd']:.3f}")
                if accept_att.get("bo") is not None:
                    notes_parts.append(f"bo={accept_att['bo']:.3f}")
            notes = " ".join(notes_parts) if notes_parts else ""
            L.append(
                f"  surface=\"{d.get('surface', '')}\""
                f"  label={label}"
                f"  conf={float(d.get('confidence') or 0):.2f}"
                f"  method={d.get('method', '?')}"
                f"  notes={notes}"
            )
        if not accept_decisions:
            L.append("  (none)")

        # Skipped non-linkable preview — status = SKIP_NON_LINKABLE
        if skip_decisions:
            L.append("")
            skip_n = min(10, len(skip_decisions))
            L.append(f"skipped_preview (N={skip_n}):")
            for d in skip_decisions[:skip_n]:
                surface = d.get("surface", "")
                ent_type = d.get("ent_type_guess", "?")
                reason = d.get("reason", "SKIP_NON_LINKABLE")
                L.append(
                    f"  surface=\"{surface}\""
                    f"  type={ent_type}"
                    f"  reason={reason}"
                )

        # Rejected preview — status = REJECT_LINKABLE only
        L.append("")
        rej_n = min(10, len(reject_decisions))
        L.append(f"top_rejected_preview (N={rej_n}):")
        for d in reject_decisions[:rej_n]:
            dec_attempts = attempts_by_decision.get(d["decision_id"], [])
            canonical = "?"
            if dec_attempts:
                canonical = dec_attempts[-1].get("candidate", "?")
            L.append(
                f"  surface=\"{d.get('surface', '')}\""
                f"  canonical={canonical}"
                f"  reason={d.get('reason', '?')}"
            )
        if not reject_decisions:
            L.append("  (none)")

        # Filtered out preview — status = FILTERED_OUT (debug-only)
        if filtered_decisions:
            L.append("")
            filt_n = min(10, len(filtered_decisions))
            L.append(f"filtered_out_preview (N={filt_n}):")
            for d in filtered_decisions[:filt_n]:
                L.append(
                    f"  surface=\"{d.get('surface', '')}\""
                    f"  reason={d.get('reason', '?')}"
                )

        # Attempt history for accepted spans (from entity_attempts table)
        attempts_for_accepted: list[dict[str, Any]] = []
        for d in accept_decisions:
            dec_attempts = attempts_by_decision.get(d["decision_id"], [])
            for a in dec_attempts:
                a["_accepted_surface"] = d.get("surface", "")
                attempts_for_accepted.append(a)
        if attempts_for_accepted:
            L.append("")
            att_n = min(10, len(attempts_for_accepted))
            L.append(f"attempts_for_accepted (N={att_n}):")
            for a in attempts_for_accepted[:att_n]:
                L.append(
                    f"  surface=\"{a.get('_accepted_surface', '')}\""
                    f"  candidate={a.get('candidate', '?')}"
                    f"  attempt={a.get('attempt_decision', '?')}"
                    f"  reason={a.get('reason', '?')}"
                )

        return "\n".join(L)

    # ── Legacy path: derive from in-memory data ─────────────────────
    # (kept for backward-compatibility with unit tests that call the
    # function directly without persisted decisions)
    L.append(f"mentions_total: {len(mentions)}")

    # Counts by method
    method_counts_legacy: dict[str, int] = {}
    for m in mentions:
        method = str(m.get("method") or "unknown")
        method_counts_legacy[method] = method_counts_legacy.get(method, 0) + 1
    L.append("mentions_by_method:")
    if method_counts_legacy:
        for method, count in sorted(method_counts_legacy.items()):
            L.append(f"  {method}: {count}")
    else:
        L.append("  (none)")

    # Partition mentions into linkable (ACCEPT) and non-linkable (SKIP)
    linkable = [m for m in mentions if str(m.get("ent_type") or "unknown") in _LINKABLE_TYPES]
    non_linkable = [m for m in mentions if str(m.get("ent_type") or "unknown") not in _LINKABLE_TYPES]

    # Top linkable mentions preview (up to 10) — final status = ACCEPT
    L.append("")
    top_n = min(10, len(linkable))
    L.append(f"top_mentions_preview (N={top_n}):")
    sorted_linkable = sorted(linkable, key=lambda m: -float(m.get("confidence", 0)))
    for m in sorted_linkable[:top_n]:
        label = m.get("label") or "-"
        notes = str(m.get("notes", ""))
        L.append(
            f"  surface=\"{m.get('surface', '')}\""
            f"  label={label}"
            f"  conf={float(m.get('confidence', 0)):.2f}"
            f"  method={m.get('method', '?')}"
            f"  notes={notes}"
        )
    if not linkable:
        L.append("  (none)")

    # Skipped non-linkable preview — final status = SKIP_NON_LINKABLE
    # Deduplicate by surface (mention + debug entry may both exist)
    skipped_entries: list[dict[str, Any]] = []
    skipped_seen: set[str] = set()
    for src in [non_linkable, salvage_debug.get("skipped_non_linkable", [])]:
        for s in src:
            key = str(s.get("surface", "")).lower()
            if key and key not in skipped_seen:
                skipped_seen.add(key)
                skipped_entries.append(s)
    if skipped_entries:
        L.append("")
        skip_n = min(10, len(skipped_entries))
        L.append(f"skipped_preview (N={skip_n}):")
        for s in skipped_entries[:skip_n]:
            surface = s.get("surface", "")
            ent_type = s.get("ent_type", s.get("canonical", "?"))
            reason = s.get("reason", s.get("notes", "SKIP_NON_LINKABLE"))
            L.append(
                f"  surface=\"{surface}\""
                f"  type={ent_type}"
                f"  reason={reason}"
            )

    # Rejected preview from salvage debug (final REJECT decisions only)
    rejected = salvage_debug.get("rejected", [])
    L.append("")
    rej_n = min(10, len(rejected))
    L.append(f"top_rejected_preview (N={rej_n}):")
    for r in rejected[:rej_n]:
        L.append(
            f"  surface=\"{r.get('surface', '')}\""
            f"  canonical={r.get('canonical', '?')}"
            f"  reason={r.get('reason', '?')}"
        )
    if not rejected:
        L.append("  (none)")

    return "\n".join(L)


def _build_consolidated_report(
    run_id: str,
    asset_ref: str,
    mentions: list[dict[str, Any]],
    salvage_debug: dict[str, Any],
    linking_result: dict[str, Any] | None,
) -> str:
    """Build a single consolidated report combining mention extraction + entity linking.

    This is the ONE report that gets printed in chat after each pipeline run.
    Every section is DB-backed — no phantom outputs.
    """
    parts: list[str] = []

    # Part 1: Mention extraction (DB-backed)
    parts.append(_build_mention_extraction_report(
        run_id, asset_ref, mentions, salvage_debug,
    ))

    # Part 2: Entity linking (DB-only via build_linking_report_from_db)
    from app.services.authority_linking import build_linking_report_from_db
    parts.append("")
    parts.append(build_linking_report_from_db(run_id))

    # Part 3: AUDIT gates (mention extraction side)
    parts.append("")
    parts.append(_build_mention_audit_gates(run_id))

    return "\n".join(parts)


def _build_mention_audit_gates(run_id: str) -> str:
    """Compute AUDIT_1, AUDIT_2, AUDIT_3 gates from DB only.

    These validate that the mention extraction report is fully consistent
    with the persisted entity_decisions and entity_attempts tables.
    """
    from app.db.pipeline_db import count_entity_decisions

    L: list[str] = []
    L.append("=== MENTION EXTRACTION AUDIT ===")

    decisions = list_entity_decisions(run_id)
    all_attempts = list_entity_attempts_for_run(run_id)

    # ── AUDIT_1: No phantom surfaces ──────────────────────────────
    # Every surface in entity_decisions has a valid status.
    valid_statuses = {"ACCEPT_LINKABLE", "REJECT_LINKABLE", "SKIP_NON_LINKABLE", "FILTERED_OUT"}
    audit_1_pass = True
    for d in decisions:
        if d.get("status") not in valid_statuses:
            audit_1_pass = False
            break
        if not d.get("surface"):
            audit_1_pass = False
            break
    L.append(f"AUDIT_1 (no phantom surfaces): {'PASS' if audit_1_pass else 'FAIL'}")

    # ── AUDIT_2: Attempt trace consistency ────────────────────────
    # Every attempt row links to a valid decision_id in entity_decisions
    decision_ids = {d["decision_id"] for d in decisions}
    audit_2_pass = True
    if all_attempts:
        for a in all_attempts:
            if a["decision_id"] not in decision_ids:
                audit_2_pass = False
                break
            if a.get("attempt_decision") not in ("ACCEPT", "REJECT"):
                audit_2_pass = False
                break
    L.append(f"AUDIT_2 (attempt trace consistency): {'PASS' if audit_2_pass else 'FAIL'}")

    # ── AUDIT_3: Decision coverage & uniqueness ───────────────────
    total = count_entity_decisions(run_id)
    accepted = count_entity_decisions(run_id, status="ACCEPT_LINKABLE")
    rejected = count_entity_decisions(run_id, status="REJECT_LINKABLE")
    skipped = count_entity_decisions(run_id, status="SKIP_NON_LINKABLE")
    filtered = count_entity_decisions(run_id, status="FILTERED_OUT")
    sum_parts = accepted + rejected + skipped + filtered

    span_keys = [d["span_key"] for d in decisions]
    unique_keys = len(set(span_keys)) == len(span_keys)

    audit_3_pass = (total == sum_parts) and unique_keys
    detail = f"total={total}, A={accepted}+R={rejected}+S={skipped}+F={filtered}={sum_parts}, unique_keys={unique_keys}"
    L.append(f"AUDIT_3 (decision coverage & uniqueness): {'PASS' if audit_3_pass else 'FAIL'} ({detail})")

    return "\n".join(L)


def _extract_mentions_from_text(
    base_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Extract mentions using quality-aware pipeline.

    Strategy:
      1. Salvage extractor (trigger/work/place rules — works on ALL text)
      2. Person mentions from proper-name sequences (capitalization heuristic)
      3. Anchor mentions (place/title/date — case-insensitive)
      4. N-gram canonical scan (case-insensitive, catches matches that
         the capitalization-based extractor misses in lowercase OCR)

    Returns ``(mentions, heuristic_candidates, salvage_debug)``.
    """
    value = str(base_text or "")
    if not value.strip():
        return [], [], {"trigger": 0, "work_fuzzy": 0, "place_candidate": 0, "rejected": []}

    salvage_mentions, salvage_debug = _extract_salvage_mentions(value)
    candidates = [
        *_extract_person_mentions(value),
        *_extract_anchor_mentions(value),
        *salvage_mentions,
    ]

    # ── N-gram canonical scan (case-insensitive fallback) ─────────────
    # Catches canonical person/work names in fully-lowercase OCR output
    # that the capitalization-dependent _extract_person_mentions misses.
    ngram_mentions = _extract_ngram_canonical_mentions(value)
    candidates.extend(ngram_mentions)

    mentions = _dedupe_mentions(value, candidates)

    # ── Post-dedup reconciliation: single decision per span ────────
    # A surface that was rejected by one strategy (e.g. salvage Rule 2
    # fuzzy match against _CANONICAL_WORKS) may have been accepted by
    # another (e.g. ngram_canonical against _CANONICAL_PERSON_NAMES).
    # After dedup selects the winners, purge stale rejections so the
    # report cannot list the same surface as both accepted AND rejected.
    #
    # Linkable types (ACCEPT) vs non-linkable types (SKIP_NON_LINKABLE):
    _LINKABLE_TYPES = {"person", "place", "work", "date"}

    accepted_surfaces_lower: set[str] = {
        str(m.get("surface", "")).lower() for m in mentions
    }
    truly_rejected: list[dict[str, Any]] = []
    attempts_for_accepted: list[dict[str, Any]] = []
    for r in salvage_debug.get("rejected", []):
        surface_low = str(r.get("surface", "")).lower()
        if surface_low in accepted_surfaces_lower:
            attempts_for_accepted.append(r)
        else:
            truly_rejected.append(r)
    salvage_debug["rejected"] = truly_rejected
    salvage_debug["attempts_for_accepted"] = attempts_for_accepted

    # Partition accepted mentions into linkable (ACCEPT) and non-linkable
    # (SKIP_NON_LINKABLE, e.g. role/title with no entity to link).
    skipped_non_linkable: list[dict[str, Any]] = []
    for m in mentions:
        ent_type = str(m.get("ent_type") or "unknown")
        if ent_type not in _LINKABLE_TYPES:
            skipped_non_linkable.append(m)
    salvage_debug["skipped_non_linkable"] = (
        salvage_debug.get("skipped_non_linkable", []) + skipped_non_linkable
    )
    # Also migrate any "skipped" entries from Rule 1 into the list
    for s in salvage_debug.pop("skipped", []):
        salvage_debug["skipped_non_linkable"].append(s)

    # ── Assertion: accepted ∩ rejected = ∅ ────────────────────────
    rejected_surfaces_lower: set[str] = {
        str(r.get("surface", "")).lower() for r in salvage_debug["rejected"]
    }
    overlap = accepted_surfaces_lower & rejected_surfaces_lower
    assert not overlap, (
        f"BUG: surfaces in BOTH accepted and rejected: {overlap}"
    )

    heuristic_candidates: list[dict[str, Any]] = []
    for mention in mentions:
        ent_type = str(mention.get("ent_type") or "unknown")
        if ent_type not in {"person", "place", "work"}:
            continue
        heuristic_candidates.append(
            {
                "mention_id": mention.get("mention_id"),
                "start_offset": int(mention.get("start_offset", 0)),
                "end_offset": int(mention.get("end_offset", 0)),
                "ent_type": ent_type,
                "surface": str(mention.get("surface") or ""),
                "source": "heuristic",
                "candidate": str(mention.get("surface") or "").strip(),
                "score": float(mention.get("confidence", 0.0)),
                "meta_json": {"ent_type": ent_type, "method": mention.get("method")},
            }
        )
    return mentions, heuristic_candidates, salvage_debug


def _extract_ngram_canonical_mentions(base_text: str) -> list[dict[str, Any]]:
    """Case-insensitive scan for canonical entity names via fuzzy n-gram matching.

    This extracts mentions that the capitalization-dependent extractor would
    miss in fully lowercase OCR output.  It uses length-aware, normalized
    edit distance + bigram overlap to avoid false positives like
    "main" → Yvain or "port" → Bohort.

    Rules:
      - Tokens < 5 chars: NEVER fuzzy-match (too many false positives).
        Only exact canonical matches are accepted.
      - Tokens >= 5 chars: normalized_edit_distance <= 0.25 AND
        bigram_overlap >= 0.40.
      - Tokens in COMMON_WORD_BLACKLIST: always rejected.
    """
    from app.services.text_normalization import (
        is_blacklisted_token,
        normalized_edit_distance,
        bigram_overlap as _bigram_overlap,
    )
    mentions: list[dict[str, Any]] = []
    value = str(base_text or "")
    if not value.strip():
        return mentions

    token_matches = list(_SALVAGE_TOKEN_RE.finditer(value))
    for m in token_matches:
        tok = m.group(0)
        tok_low = tok.lower()
        if len(tok_low) < 4:
            continue
        # Blacklist gate: common words never match
        if is_blacklisted_token(tok_low):
            continue

        # Check against canonical person names
        matched_person = False
        for canon, canon_norm in _CANONICAL_PERSON_NAMES.items():
            nd = normalized_edit_distance(tok_low, canon)
            # Short tokens (< 5 chars): exact match only
            if len(tok_low) < 5:
                if nd == 0.0:
                    mentions.append({
                        "start_offset": m.start(),
                        "end_offset": m.end(),
                        "surface": tok,
                        "norm": canon_norm,
                        "label": None,
                        "ent_type": "person",
                        "confidence": 0.65,
                        "method": "rule:ngram_canonical_person",
                        "notes": f"canonical={canon} nd=0.000 bo=1.000",
                    })
                    matched_person = True
                    break
                continue
            # Tokens >= 5 chars: normalized distance + bigram overlap
            bo = _bigram_overlap(tok_low, canon)
            if nd <= 0.25 and bo >= 0.40:
                conf = max(0.45, 0.70 - nd)
                mentions.append({
                    "start_offset": m.start(),
                    "end_offset": m.end(),
                    "surface": tok,
                    "norm": canon_norm,
                    "label": None,
                    "ent_type": "person",
                    "confidence": round(conf, 3),
                    "method": "rule:ngram_canonical_person",
                    "notes": f"canonical={canon} nd={nd:.3f} bo={bo:.3f}",
                })
                matched_person = True
                break
        if matched_person:
            continue

        # Check against canonical works
        for canon, canon_norm in _CANONICAL_WORKS.items():
            nd = normalized_edit_distance(tok_low, canon)
            if len(tok_low) < 5:
                if nd == 0.0:
                    mentions.append({
                        "start_offset": m.start(),
                        "end_offset": m.end(),
                        "surface": tok,
                        "norm": canon_norm,
                        "label": "WORK",
                        "ent_type": "work",
                        "confidence": 0.60,
                        "method": "rule:ngram_canonical_work",
                        "notes": f"canonical={canon} nd=0.000 bo=1.000",
                    })
                    break
                continue
            bo = _bigram_overlap(tok_low, canon)
            if nd <= 0.25 and bo >= 0.40:
                conf = max(0.40, 0.65 - nd)
                mentions.append({
                    "start_offset": m.start(),
                    "end_offset": m.end(),
                    "surface": tok,
                    "norm": canon_norm,
                    "label": "WORK",
                    "ent_type": "work",
                    "confidence": round(conf, 3),
                    "method": "rule:ngram_canonical_work",
                    "notes": f"canonical={canon} nd={nd:.3f} bo={bo:.3f}",
                })
                break
    return mentions


def _parse_nd_bo_from_notes(notes: str | None) -> tuple[float | None, float | None, str | None]:
    """Extract nd, bo, canonical from notes strings like 'canonical=X nd=0.222 bo=0.750'."""
    if not notes:
        return None, None, None
    nd_val: float | None = None
    bo_val: float | None = None
    canon_val: str | None = None
    m_nd = re.search(r"nd=([\d.]+)", notes)
    if m_nd:
        nd_val = float(m_nd.group(1))
    m_bo = re.search(r"bo=([\d.]+)", notes)
    if m_bo:
        bo_val = float(m_bo.group(1))
    m_canon = re.search(r"canonical=(\S+)", notes)
    if m_canon:
        canon_val = m_canon.group(1)
    return nd_val, bo_val, canon_val


def _run_trace_analysis(
    run_id: str, base_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run chunk/entity analysis.

    Returns ``(chunk_rows, mention_rows, candidate_rows, salvage_debug)``.
    """
    clear_analysis_for_run(run_id)
    chunks_input = _build_line_chunks(base_text)
    chunk_rows = insert_chunks(run_id, chunks_input)

    mentions_input, candidate_input, salvage_debug = _extract_mentions_from_text(base_text)
    for mention in mentions_input:
        mention["chunk_id"] = _assign_chunk_id_for_span(
            int(mention.get("start_offset", 0)),
            int(mention.get("end_offset", 0)),
            chunk_rows,
        )
    mention_rows = insert_entity_mentions(run_id, mentions_input)

    mention_id_by_span = {
        (int(row["start_offset"]), int(row["end_offset"]), str(row["ent_type"]), str(row["surface"])): str(row["mention_id"])
        for row in mention_rows
    }
    for candidate in candidate_input:
        key = (
            int(candidate.get("start_offset", 0)),
            int(candidate.get("end_offset", 0)),
            str(candidate.get("ent_type", "unknown")),
            str(candidate.get("surface", "")),
        )
        mention_id = mention_id_by_span.get(key)
        if mention_id:
            candidate["mention_id"] = mention_id
    candidate_rows = insert_entity_candidates([row for row in candidate_input if row.get("mention_id")])

    # ── Persist ALL final decisions to entity_decisions + entity_attempts ─
    import uuid as _uuid_mod
    _LINKABLE_TYPES = {"person", "place", "work", "date"}
    decision_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []

    # Build lookup for attempts_for_accepted (keyed by surface lower)
    attempts_by_surface: dict[str, list[dict[str, Any]]] = {}
    for att in salvage_debug.get("attempts_for_accepted", []):
        key = str(att.get("surface", "")).lower()
        attempts_by_surface.setdefault(key, []).append(att)

    # 1) ACCEPT_LINKABLE decisions – from accepted linkable mentions
    for m in mention_rows:
        ent_type = str(m.get("ent_type") or "unknown")
        if ent_type not in _LINKABLE_TYPES:
            continue  # non-linkable go to SKIP below
        notes = str(m.get("notes") or "")
        nd, bo, canonical = _parse_nd_bo_from_notes(notes)
        surf_low = str(m.get("surface", "")).lower()
        decision_id = str(_uuid_mod.uuid4())
        decision_rows.append({
            "decision_id": decision_id,
            "chunk_id": m.get("chunk_id"),
            "start_offset": int(m.get("start_offset", 0)),
            "end_offset": int(m.get("end_offset", 0)),
            "surface": str(m.get("surface") or ""),
            "norm": m.get("norm"),
            "ent_type_guess": ent_type,
            "label": m.get("label"),
            "status": "ACCEPT_LINKABLE",
            "confidence": float(m.get("confidence", 0.0)),
            "method": str(m.get("method") or "unknown"),
            "reason": "accepted_by_extraction_pipeline",
            "meta_json": None,
        })
        # Persist prior REJECT attempts for this accepted surface
        att_idx = 0
        for att in attempts_by_surface.get(surf_low, []):
            att_reason = str(att.get("reason") or "rejected")
            att_nd, att_bo, att_canonical = _parse_nd_bo_from_notes(
                f"canonical={att.get('canonical', '?')} nd={att.get('dist', 0)} bo=0"
            )
            # Parse nd and bo from reason string like "nd=0.444>0.25 or bo=0.286<0.40"
            m_nd = re.search(r"nd=([\d.]+)", att_reason)
            m_bo = re.search(r"bo=([\d.]+)", att_reason)
            attempt_rows.append({
                "decision_id": decision_id,
                "attempt_idx": att_idx,
                "candidate_source": "rule:salvage_work_fuzzy",
                "candidate": str(att.get("canonical") or "?"),
                "candidate_label": None,
                "candidate_type": "work",
                "nd": float(m_nd.group(1)) if m_nd else att.get("dist"),
                "bo": float(m_bo.group(1)) if m_bo else None,
                "threshold_nd": float(att.get("max_dist", 0.25)),
                "threshold_bo": 0.40,
                "attempt_decision": "REJECT",
                "reason": att_reason,
                "meta_json": None,
            })
            att_idx += 1
        # Persist final ACCEPT attempt (from mention notes)
        if canonical:
            attempt_rows.append({
                "decision_id": decision_id,
                "attempt_idx": att_idx,
                "candidate_source": str(m.get("method") or "unknown"),
                "candidate": canonical,
                "candidate_label": None,
                "candidate_type": ent_type,
                "nd": nd,
                "bo": bo,
                "threshold_nd": 0.25,
                "threshold_bo": 0.40,
                "attempt_decision": "ACCEPT",
                "reason": "accepted_by_extraction_pipeline",
                "meta_json": None,
            })

    # 2) SKIP_NON_LINKABLE – non-linkable mentions + skipped_non_linkable debug
    seen_skip_surfaces: set[str] = set()
    for m in mention_rows:
        ent_type = str(m.get("ent_type") or "unknown")
        if ent_type in _LINKABLE_TYPES:
            continue
        surf_low = str(m.get("surface", "")).lower()
        if surf_low in seen_skip_surfaces:
            continue
        seen_skip_surfaces.add(surf_low)
        decision_rows.append({
            "chunk_id": m.get("chunk_id"),
            "start_offset": int(m.get("start_offset", 0)),
            "end_offset": int(m.get("end_offset", 0)),
            "surface": str(m.get("surface") or ""),
            "norm": m.get("norm"),
            "ent_type_guess": ent_type,
            "label": m.get("label"),
            "status": "SKIP_NON_LINKABLE",
            "confidence": float(m.get("confidence", 0.0)),
            "method": str(m.get("method") or "unknown"),
            "reason": f"ent_type={ent_type} not in linkable types",
            "meta_json": None,
        })
    # Also persist skipped entries from salvage debug that are NOT already
    # covered by a mention row (e.g. role entries created in Rule 1)
    for s in salvage_debug.get("skipped_non_linkable", []):
        surf_low = str(s.get("surface", "")).lower()
        if surf_low in seen_skip_surfaces:
            continue
        seen_skip_surfaces.add(surf_low)
        decision_rows.append({
            "chunk_id": None,
            "start_offset": int(s.get("start_offset", 0)),
            "end_offset": int(s.get("end_offset", 0)),
            "surface": str(s.get("surface") or ""),
            "norm": None,
            "ent_type_guess": str(s.get("ent_type", "role")),
            "label": s.get("label"),
            "status": "SKIP_NON_LINKABLE",
            "confidence": 0.0,
            "method": str(s.get("method") or "rule:salvage_trigger"),
            "reason": str(s.get("reason") or "SKIP_NON_LINKABLE"),
            "meta_json": None,
        })

    # 3) REJECT_LINKABLE / FILTERED_OUT – from salvage_debug["rejected"]
    _FILTER_REASON_PATTERNS = (
        "too short",
        "stopword",
        "blacklisted",
        "non-alpha",
        "editorial_blacklist",
        "place_likeness",
        "OCR garbage",
    )
    for r in salvage_debug.get("rejected", []):
        reason = str(r.get("reason") or "rejected_by_extraction_pipeline")
        canonical = r.get("canonical")
        is_linkable_reject = (
            canonical is not None
            and str(canonical) not in ("(blacklisted)", "(place)", "(person heuristic)")
            and "nd=" in reason
        )
        if not is_linkable_reject:
            is_filter = any(pat in reason.lower() for pat in _FILTER_REASON_PATTERNS)
            if not is_filter:
                is_filter = str(canonical or "").startswith("(")
        else:
            is_filter = False
        status = "FILTERED_OUT" if (not is_linkable_reject or is_filter) else "REJECT_LINKABLE"
        decision_id = str(_uuid_mod.uuid4())
        r_start = int(r.get("start_offset", 0))
        r_end = int(r.get("end_offset", 0))
        method_str = "filter:gate" if status == "FILTERED_OUT" else "rule:salvage_work_fuzzy"
        decision_rows.append({
            "decision_id": decision_id,
            "chunk_id": None,
            "start_offset": r_start,
            "end_offset": r_end,
            "surface": str(r.get("surface") or ""),
            "norm": None,
            "ent_type_guess": str(r.get("ent_type", "unknown")),
            "label": None,
            "status": status,
            "confidence": 0.0,
            "method": method_str,
            "reason": reason,
            "meta_json": None,
        })
        # For REJECT_LINKABLE, persist the failed comparison as an attempt
        if status == "REJECT_LINKABLE" and canonical:
            m_nd = re.search(r"nd=([\d.]+)", reason)
            m_bo = re.search(r"bo=([\d.]+)", reason)
            attempt_rows.append({
                "decision_id": decision_id,
                "attempt_idx": 0,
                "candidate_source": "rule:salvage_work_fuzzy",
                "candidate": str(canonical),
                "candidate_label": None,
                "candidate_type": "work",
                "nd": float(m_nd.group(1)) if m_nd else r.get("dist"),
                "bo": float(m_bo.group(1)) if m_bo else None,
                "threshold_nd": float(r.get("max_dist", 0.25)),
                "threshold_bo": 0.40,
                "attempt_decision": "REJECT",
                "reason": reason,
                "meta_json": None,
            })

    if decision_rows:
        insert_entity_decisions(run_id, decision_rows)
    if attempt_rows:
        insert_entity_attempts(attempt_rows)

    return chunk_rows, mention_rows, candidate_rows, salvage_debug


def _normalize_language_token(token: str) -> str:
    return str(token or "").lower().replace("ſ", "s").replace("v", "u").replace("j", "i")


def _token_vowel_ratio(token: str) -> float:
    value = str(token or "").lower()
    if not value:
        return 0.0
    vowels = sum(1 for char in value if char in _LATIN_VOWELS)
    return float(vowels) / float(len(value))


def _clean_text_for_langid(text: str, *, enforce_min_letters: bool) -> str:
    value = str(text or "")
    if not value.strip():
        return ""

    value = value.replace("&", " et ")
    value = re.sub(r"\[(?:…|\.\.\.)\]", " ", value)
    value = value.replace("?", " ")
    value = value.replace("ſ", "s")
    value = value.replace("\n", " ")

    tokens: list[str] = []
    for raw_token in re.split(r"\s+", value):
        token = str(raw_token or "").strip()
        if not token:
            continue
        letters_only = "".join(ch for ch in token if ch.isalpha())
        if not letters_only:
            continue
        non_letters = max(0, len(token) - len(letters_only))
        if len(token) > 0 and (float(non_letters) / float(len(token))) > 0.5:
            continue
        if len(letters_only) >= 4 and _LATIN_TOKEN_RE.match(letters_only):
            if _token_vowel_ratio(letters_only) < 0.20:
                continue
        tokens.append(letters_only.lower())

    cleaned = " ".join(tokens)
    if enforce_min_letters and len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žſƀ-ɏ]", cleaned)) < _LANGID_MIN_LETTERS:
        return ""
    return cleaned


def _langid_view(text: str) -> str:
    return _clean_text_for_langid(text, enforce_min_letters=True)


def _latin_anchor_hits(cleaned_text: str) -> int:
    tokens = {_normalize_language_token(token) for token in re.split(r"\s+", cleaned_text) if token}
    return sum(1 for anchor in _LATIN_STRONG_ANCHORS if anchor in tokens)


def _resolve_french_family_language(text: str, cleaned_text: str | None = None) -> str | None:
    cleaned = cleaned_text if cleaned_text is not None else _clean_text_for_langid(text, enforce_min_letters=False)
    if not cleaned:
        return None

    tokens = [_normalize_language_token(token) for token in re.split(r"\s+", cleaned) if token]
    if not tokens:
        return None

    strong_hits = 0
    function_hits = 0
    old_hits = 0
    middle_hits = 0
    anglo_hits = 0
    for token in tokens:
        if any(token == anchor or token.startswith(anchor) for anchor in _FRENCH_STRONG_ANCHORS):
            strong_hits += 1
        if token in _FRENCH_FUNCTION_WORDS:
            function_hits += 1
        if token in _FRENCH_OLD_MARKERS:
            old_hits += 1
        if token in _FRENCH_MIDDLE_MARKERS:
            middle_hits += 1
        if token in _ANGLO_NORMAN_MARKERS:
            anglo_hits += 1

    if not (strong_hits >= 2 or (strong_hits >= 1 and function_hits >= 2)):
        return None
    if anglo_hits >= 2:
        return "anglo_norman"
    if old_hits >= 2 and middle_hits == 0:
        return "old_french"
    return "middle_french"


def _is_french_family_language(value: str | None) -> bool:
    return str(value or "") in {"french", "old_french", "middle_french", "anglo_norman"}


def _is_relevant_text_label(label: str) -> bool:
    key = str(label or "").lower().strip()
    if not key:
        return False
    if any(token in key for token in _TEXT_LABEL_EXCLUDE_TOKENS):
        return False
    return any(token in key for token in _TEXT_LABEL_INCLUDE_TOKENS)


def _extract_location_suggestions(coco: dict[str, Any] | None) -> list[SaiaOCRLocationSuggestion]:
    if not isinstance(coco, dict):
        return []

    categories = coco.get("categories")
    annotations = coco.get("annotations")
    if not isinstance(categories, list) or not isinstance(annotations, list):
        return []

    category_by_id: dict[int, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            continue
        try:
            category_id = int(category.get("id"))
        except Exception:
            continue
        category_name = str(category.get("name") or "").strip()
        if category_name:
            category_by_id[category_id] = category_name

    suggestions: list[SaiaOCRLocationSuggestion] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            x, y, w, h = [float(bbox[idx]) for idx in range(4)]
        except Exception:
            continue
        if not all(value >= 0 for value in (w, h)):
            continue
        if w < 8 or h < 8:
            continue

        category_name = category_by_id.get(int(annotation.get("category_id") or -1), "")
        if not _is_relevant_text_label(category_name):
            continue

        region_id = str(annotation.get("id") or f"region-{len(suggestions) + 1}")
        suggestions.append(
            SaiaOCRLocationSuggestion(
                region_id=region_id,
                category=category_name,
                bbox_xywh=[x, y, w, h],
            )
        )

    suggestions.sort(key=lambda item: (item.bbox_xywh[1], item.bbox_xywh[0]))
    return suggestions[:80]


def _run_segmentation_for_suggestions(image_bytes: bytes) -> list[SaiaOCRLocationSuggestion]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        source_path = Path(tmp_dir) / "ocr_full_page_input"
        source_path.write_bytes(image_bytes)

        prepared_path, prepared_changed = _prepare_image_for_segmentation(str(source_path))
        try:
            annotated_path = str(Path(tmp_dir) / "annotated.jpg")
            coco, _stats = run_single_segmentation(
                prepared_path,
                confidence=0.25,
                iou=0.3,
                selected_classes=FINAL_CLASSES,
                annotated_output_path=annotated_path,
            )
        finally:
            if prepared_changed and prepared_path.endswith(".seg.jpg"):
                try:
                    Path(prepared_path).unlink(missing_ok=True)
                except Exception:
                    pass

    return _extract_location_suggestions(coco)


def _detect_language_metadata(text: str) -> tuple[str, float | None]:
    value = str(text or "")
    if not value.strip():
        return "unknown", None
    if _ARABIC_RE.search(value):
        return "arabic", 0.99
    if _HEBREW_RE.search(value):
        return "hebrew", 0.99

    cleaned = _langid_view(value)
    if not cleaned:
        return "unknown", None

    try:
        candidates = list(detect_langs(cleaned))
    except Exception:
        candidates = []

    best_language = "unknown"
    best_confidence: float | None = None
    for candidate in candidates:
        raw_lang = str(getattr(candidate, "lang", "unknown") or "unknown").strip().lower()
        normalized = _normalize_detected_language(raw_lang)
        if normalized == "unknown":
            continue
        try:
            probability = float(getattr(candidate, "prob", None))
        except Exception:
            probability = 0.0
        best_language = normalized
        best_confidence = max(0.0, min(1.0, probability))
        break

    french_family = _resolve_french_family_language(value, cleaned_text=cleaned)
    if french_family is not None:
        if best_language in {"unknown", "latin"}:
            promoted_confidence = max(0.60, best_confidence or 0.0)
            return french_family, min(0.75, promoted_confidence)
        if best_language == "french":
            promoted_confidence = max(0.62, best_confidence or 0.0)
            return french_family, min(0.75, promoted_confidence)
    elif best_language in {
        "french",
        "old_french",
        "middle_french",
        "anglo_norman",
        "italian",
        "spanish",
        "portuguese",
        "catalan",
        "occitan",
    } and _latin_anchor_hits(cleaned) >= 2:
        return "latin", max(0.55, best_confidence or 0.0)

    if best_language == "unknown":
        return "latin", 0.45
    return best_language, best_confidence


def _normalize_detected_language(value: str | None) -> str:
    language = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "la": "latin",
        "fr": "french",
        "fro": "old_french",
        "frm": "middle_french",
        "enm": "middle_english",
        "ang": "old_english",
        "de": "german",
        "it": "italian",
        "es": "spanish",
        "pt": "portuguese",
        "nl": "dutch",
        "ca": "catalan",
        "oc": "occitan",
        "he": "hebrew",
        "ar": "arabic",
        "el": "greek",
    }
    language = aliases.get(language, language)
    if language in _ALLOWED_DETECTED_LANGUAGES:
        return language
    return "unknown"


def _fallback_detected_language(script_hint: str | None, text: str | None) -> str:
    value = str(text or "")
    script = str(script_hint or "unknown").lower()

    if not value.strip():
        return "unknown"

    if _OLD_ENGLISH_MARKERS_RE.search(value):
        return "old_english"
    if _ARABIC_RE.search(value):
        return "arabic"
    if _HEBREW_RE.search(value):
        return "hebrew"
    french_family = _resolve_french_family_language(value)
    if french_family is not None:
        return french_family
    if script == "greek":
        return "greek"
    if script == "cyrillic":
        return "church_slavonic"
    if script == "mixed":
        return "mixed"
    return "latin"


def _resolve_full_page_status(*, text: str, confidence: float | None, warnings: list[str]) -> str:
    if not str(text or "").strip():
        return "EMPTY"

    upper_warnings = {str(item).upper() for item in warnings}
    if any(key in upper_warnings for key in _PARTIAL_WARNING_KEYS):
        return "PARTIAL"

    if confidence is not None and confidence < 0.4:
        return "PARTIAL"

    return "FULL"


def _latest_completed_stage(events: list[dict[str, Any]]) -> str | None:
    for item in reversed(events):
        if str(item.get("event") or "").upper() == "END":
            return str(item.get("stage") or "")
    return None


def _build_trace_snapshot(run_id: str) -> dict[str, Any] | None:
    run = get_run(run_id)
    if run is None:
        return None
    events = list_events(run_id)
    chunks_preview = list_chunks(run_id, limit=20)
    mentions_preview = list_entity_mentions(run_id, limit=20)
    chunks_count = int(run.get("chunks_count") or count_chunks(run_id))
    mentions_count = int(run.get("mentions_count") or count_entity_mentions(run_id))
    return {
        "run_id": run_id,
        "run": run,
        "events": events,
        "latest_completed_stage": _latest_completed_stage(events),
        "chunks_count": chunks_count,
        "mentions_count": mentions_count,
        "chunks_preview": chunks_preview,
        "top_mentions": mentions_preview,
    }


@router.post("/ocr/extract", response_model=OCRExtractAnyResponse)
async def ocr_extract(
    payload: OCRExtractRequest,
    mode: str | None = Query(default=None, pattern="^(full|simple)$"),
) -> OCRExtractAnyResponse:
    effective_payload = payload
    if mode in {"full", "simple"} and payload.mode != mode:
        effective_payload = payload.model_copy(update={"mode": mode})
    try:
        return _get_ocr_agent().run(effective_payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SaiaConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OCRAgentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"OCR extraction failed: {exc}") from exc


@router.post("/ocr/saia", response_model=SaiaOCRResponse)
async def ocr_saia(payload: SaiaOCRRequest) -> SaiaOCRResponse:
    try:
        return _get_saia_ocr_agent().extract(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SaiaConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SaiaOCRAgentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"SAIA OCR failed: {exc}") from exc


@router.post("/ocr/extract_full_page", response_model=SaiaFullPageExtractResponse)
async def ocr_extract_full_page(payload: SaiaFullPageExtractRequest) -> SaiaFullPageExtractResponse:
    base_warnings: list[str] = []

    try:
        image_bytes = decode_image_bytes(payload.image_b64 or "")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image_b64 payload: {exc}") from exc

    location_suggestions = list(payload.location_suggestions)
    if not location_suggestions:
        try:
            location_suggestions = await asyncio.to_thread(_run_segmentation_for_suggestions, image_bytes)
        except Exception as exc:
            base_warnings.append(f"SEGMENTATION_FAILED:{exc}")

    try:
        ocr_result = await asyncio.to_thread(
            _get_saia_ocr_agent().extract,
            SaiaOCRRequest(
                image_id=payload.image_id,
                page_id=payload.page_id,
                image_b64=payload.image_b64,
                script_hint_seed=payload.script_hint_seed,
                apply_proofread=payload.apply_proofread,
                location_suggestions=location_suggestions,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SaiaConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SaiaOCRAgentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Full-page OCR failed: {exc}") from exc

    warnings = list(dict.fromkeys([*base_warnings, *ocr_result.warnings]))
    fallbacks_used = list(dict.fromkeys(ocr_result.fallbacks_used or [item.model for item in ocr_result.fallbacks]))

    detected_language = _normalize_detected_language(ocr_result.detected_language)
    local_language, local_confidence = _detect_language_metadata(ocr_result.text)
    local_language = _normalize_detected_language(local_language)
    confidence_value = float(ocr_result.confidence) if ocr_result.confidence is not None else None
    language_confidence: float | None = confidence_value
    text_value = str(ocr_result.text or "")
    french_family_guess = _resolve_french_family_language(text_value)

    if not text_value.strip():
        detected_language = "unknown"
        language_confidence = None
    else:
        if detected_language == "unknown":
            if local_language != "unknown":
                detected_language = local_language
                language_confidence = local_confidence
            elif french_family_guess is not None:
                detected_language = french_family_guess
                language_confidence = 0.58
        elif local_language == detected_language and local_confidence is not None:
            baseline = language_confidence if language_confidence is not None else local_confidence
            language_confidence = min(1.0, max(baseline, local_confidence) + 0.06)
        else:
            if (
                french_family_guess is not None
                and detected_language in {"latin", "unknown", "french"}
                and not (
                    detected_language in {"old_french", "middle_french", "anglo_norman"}
                    and local_language == detected_language
                )
            ):
                detected_language = french_family_guess
                if _is_french_family_language(local_language) and local_confidence is not None:
                    language_confidence = min(0.75, max(local_confidence, language_confidence or 0.0))
                else:
                    language_confidence = min(0.70, max(language_confidence or 0.0, 0.58))
            elif (
                detected_language == "latin"
                and _is_french_family_language(local_language)
                and local_confidence is not None
                and local_confidence >= 0.55
            ):
                detected_language = "middle_french" if local_language == "french" else local_language
                language_confidence = min(0.75, max(0.55, local_confidence))

        if detected_language == "unknown":
            detected_language = _fallback_detected_language(ocr_result.script_hint, text_value)
            if detected_language != "unknown" and language_confidence is None:
                language_confidence = 0.55 if _is_french_family_language(detected_language) else 0.45

    if language_confidence is not None:
        language_confidence = max(0.0, min(1.0, float(language_confidence)))
    status = _resolve_full_page_status(text=ocr_result.text, confidence=confidence_value, warnings=warnings)

    return SaiaFullPageExtractResponse(
        status=status,  # type: ignore[arg-type]
        model_used=ocr_result.model_used,
        fallbacks_used=fallbacks_used,
        detected_language=detected_language,
        language_confidence=language_confidence,
        script_hint=ocr_result.script_hint,
        confidence=confidence_value,
        warnings=warnings,
        lines=list(ocr_result.lines),
        text=ocr_result.text,
        fallbacks=list(ocr_result.fallbacks),
    )


# ── Quality label ranking helper ──────────────────────────────────────

_QUALITY_RANK = {"HIGH": 0, "OK": 1, "RISKY": 2, "UNRELIABLE": 3}


def _quality_rank(label: str) -> int:
    """Return numeric rank for quality label (lower = better)."""
    return _QUALITY_RANK.get(label, 4)


@router.post("/ocr/page_with_trace")
async def ocr_page_with_trace(payload: SaiaFullPageExtractRequest) -> dict[str, Any]:
    """Run single-page OCR pipeline with trace logging.

    Implements hard quality gate enforcement with effective seam retry:
      - OCR runs in a retry loop (max_attempts configurable, default 3)
      - Attempt 0: normal OCR with standard tiling
      - Attempt 1+: seam-aware retry using strategy selection
        (grid_shift → seam_band_crop → expand_overlap)
      - NO-OP guard: if retry produces identical text_sha256 or tile_boxes,
        the retry is marked as failed/no-op and the next strategy is tried
      - After each attempt, quality gates are evaluated
      - If any blocking gate fails, downstream stages are SKIPPED
      - If all attempts fail gates, run is marked FAILED_QUALITY
    """
    # ── Configurable retry params ─────────────────────────────────
    MAX_OCR_ATTEMPTS = _MAX_OCR_ATTEMPTS  # from ocr_quality_config

    try:
        image_bytes = decode_image_bytes(payload.image_b64 or "")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image_b64 payload: {exc}") from exc

    # Get image dimensions for strategy computation
    import io
    from PIL import Image as _PILImage
    with _PILImage.open(io.BytesIO(image_bytes)) as _pil_img:
        img_w, img_h = _pil_img.size

    asset_ref = (
        str(payload.page_id or "").strip()
        or str(payload.image_id or "").strip()
        or str(payload.document_id or "").strip()
        or "inline:image_b64"
    )
    asset_sha256 = hashlib.sha256(image_bytes).hexdigest()
    run_id = create_run(asset_ref=asset_ref, asset_sha256=asset_sha256)
    log_event(run_id, "RECEIVED", "START", "OCR trace run received.")

    location_suggestions = list(payload.location_suggestions)
    base_warnings: list[str] = []
    if not location_suggestions:
        try:
            location_suggestions = await asyncio.to_thread(_run_segmentation_for_suggestions, image_bytes)
            log_event(run_id, "RECEIVED", "INFO",
                      f"Segmentation location suggestions prepared: {len(location_suggestions)}")
        except Exception as exc:
            base_warnings.append(f"SEGMENTATION_FAILED:{exc}")
            log_event(run_id, "RECEIVED", "ERROR", f"Segmentation failed: {exc}")
    else:
        log_event(run_id, "RECEIVED", "INFO",
                  f"Using provided location suggestions: {len(location_suggestions)}")

    try:
        # ══════════════════════════════════════════════════════════════
        # OCR RETRY LOOP — attempt 0..MAX_OCR_ATTEMPTS-1
        # Each attempt: run OCR → compute quality → evaluate gates
        # NO-OP guard: if text_sha256 or tile_boxes identical, switch strategy
        # If gates pass → proceed to downstream. If fail → retry or stop.
        # ══════════════════════════════════════════════════════════════
        from app.agents.saia_ocr_agent import (
            compute_sanity, _quality_label_from_sanity,
            quality_gate_enforce, sanity_adjust_confidence,
            force_uncertainty_markers, format_sanity_summary,
        )

        best_ocr_result: SaiaOCRResponse | None = None
        best_ocr_payload: dict[str, Any] | None = None
        best_quality_report: OCRQualityReport | None = None
        best_gate_decisions: dict[str, Any] | None = None
        best_effective_quality: EffectiveQuality | None = None
        best_attempt_idx: int = -1
        gates_ever_passed: bool = False

        # Track previous attempt for NO-OP detection
        prev_text_sha256: str = ""
        prev_tiling_plan: TilingPlan | None = None

        # Build initial tiling plan from location_suggestions
        suggestion_dicts = [
            {"region_id": getattr(s, "region_id", None),
             "category": getattr(s, "category", None),
             "bbox_xywh": list(getattr(s, "bbox_xywh", []))}
            for s in location_suggestions
        ]
        current_plan = default_plan_from_suggestions(suggestion_dicts, img_w, img_h)

        for attempt_idx in range(MAX_OCR_ATTEMPTS):
            # ── Determine tiling strategy for this attempt ────────────
            if attempt_idx == 0:
                effective_suggestions = location_suggestions
            else:
                # Select a retry strategy that changes tile_boxes
                assert prev_tiling_plan is not None
                current_plan = select_retry_strategy(
                    prev_tiling_plan, img_w, img_h,
                    prev_text_sha256=prev_text_sha256,
                    attempt_idx=attempt_idx,
                )
                # Convert TilingPlan → SaiaOCRLocationSuggestion objects
                retry_dicts = plan_to_suggestions(current_plan, img_w, img_h)
                effective_suggestions = [
                    SaiaOCRLocationSuggestion(**d) for d in retry_dicts
                ]
                log_event(run_id, "OCR_RUNNING", "INFO",
                          f"Retry strategy={current_plan.strategy} "
                          f"grid={current_plan.grid} "
                          f"tiles={len(current_plan.tile_boxes)} "
                          f"overlap={current_plan.overlap_pct:.0%}")

            log_event(run_id, "OCR_RUNNING", "START",
                      f"OCR attempt {attempt_idx}/{MAX_OCR_ATTEMPTS-1} "
                      f"strategy={current_plan.strategy}")
            update_run_fields(run_id, status="RUNNING",
                              current_stage="OCR_RUNNING")

            # ── Run OCR ───────────────────────────────────────────────
            ocr_result = await asyncio.to_thread(
                _get_saia_ocr_agent().extract,
                SaiaOCRRequest(
                    image_id=payload.image_id,
                    page_id=payload.page_id,
                    image_b64=payload.image_b64,
                    script_hint_seed=payload.script_hint_seed,
                    apply_proofread=False,
                    location_suggestions=effective_suggestions,
                ),
            )
            log_event(run_id, "OCR_DONE", "END",
                      f"OCR attempt {attempt_idx} complete, "
                      f"model={ocr_result.model_used}")

            # ── Log tile info ─────────────────────────────────────────
            raw_blob = ocr_result.raw_json if isinstance(ocr_result.raw_json, dict) else {}
            if "tile_count" in raw_blob:
                log_event(run_id, "OCR_DONE", "INFO",
                          f"Tile-based OCR: {raw_blob['tile_count']} tiles "
                          f"(strategy={current_plan.strategy}, grid={current_plan.grid}).")
            if raw_blob.get("column_split"):
                log_event(run_id, "OCR_DONE", "INFO",
                          f"Column-split OCR: {raw_blob.get('columns', 2)} cols.")

            # ── Legacy sanity metrics ─────────────────────────────────
            ocr_sanity = compute_sanity(str(ocr_result.text or ""))
            quality_label = raw_blob.get("quality_label") or _quality_label_from_sanity(ocr_sanity)
            sanity_metrics = raw_blob.get("sanity_metrics") or ocr_sanity
            sanity_summary = format_sanity_summary(sanity_metrics)

            log_event(run_id, "OCR_SANITY", "INFO",
                      f"attempt={attempt_idx} quality_label={quality_label} "
                      f"single_char={ocr_sanity.get('single_char_ratio',0):.3f} "
                      f"weird={ocr_sanity.get('weird_ratio',0):.3f} "
                      f"leading_frag={ocr_sanity.get('leading_fragment_ratio',0):.3f}")

            ocr_payload: dict[str, Any] = {
                "lines": list(ocr_result.lines),
                "text": str(ocr_result.text or ""),
                "script_hint": str(ocr_result.script_hint or "unknown"),
                "detected_language": _normalize_detected_language(ocr_result.detected_language),
                "confidence": float(ocr_result.confidence) if ocr_result.confidence is not None else 0.0,
                "warnings": list(dict.fromkeys([*base_warnings, *ocr_result.warnings, sanity_summary])),
                "quality_label": quality_label,
                "sanity_metrics": sanity_metrics,
            }

            # ── Compute text hash for NO-OP detection ─────────────────
            text_sha256 = hashlib.sha256(
                (ocr_payload["text"] or "").encode()
            ).hexdigest()
            current_plan.text_sha256 = text_sha256

            # ── NO-OP guard + auto-escalation ─────────────────────────
            noop_detected = False
            if attempt_idx > 0 and prev_tiling_plan is not None:
                noop_detected = is_noop_retry(
                    prev_text_sha256, text_sha256,
                    prev_tiling_plan, current_plan,
                )
                if noop_detected:
                    log_event(run_id, "OCR_DONE", "WARN",
                              f"NO-OP detected on attempt {attempt_idx}: "
                              f"text_sha256 {'identical' if text_sha256 == prev_text_sha256 else 'different'}, "
                              f"tile_boxes {'identical' if current_plan.boxes_signature() == prev_tiling_plan.boxes_signature() else 'different'}. "
                              f"Auto-escalating to next strategy.")
                    # Auto-escalate: immediately pick a different strategy
                    escalated_plan = select_retry_strategy(
                        current_plan, img_w, img_h,
                        prev_text_sha256=text_sha256,
                        attempt_idx=attempt_idx + 10,  # high idx to force fallback chain
                    )
                    if escalated_plan.boxes_signature() != current_plan.boxes_signature():
                        log_event(run_id, "OCR_DONE", "INFO",
                                  f"NOOP escalation: re-running OCR with "
                                  f"strategy={escalated_plan.strategy} "
                                  f"grid={escalated_plan.grid}")
                        current_plan = escalated_plan
                        retry_dicts = plan_to_suggestions(current_plan, img_w, img_h)
                        effective_suggestions = [
                            SaiaOCRLocationSuggestion(**d) for d in retry_dicts
                        ]
                        ocr_result = await asyncio.to_thread(
                            _get_saia_ocr_agent().extract,
                            SaiaOCRRequest(
                                image_id=payload.image_id,
                                page_id=payload.page_id,
                                image_b64=payload.image_b64,
                                script_hint_seed=payload.script_hint_seed,
                                apply_proofread=False,
                                location_suggestions=effective_suggestions,
                            ),
                        )
                        # Recompute text hash after escalation
                        ocr_payload["lines"] = list(ocr_result.lines)
                        ocr_payload["text"] = str(ocr_result.text or "")
                        text_sha256 = hashlib.sha256(
                            (ocr_payload["text"] or "").encode()
                        ).hexdigest()
                        current_plan.text_sha256 = text_sha256
                        noop_detected = False  # no longer a NOOP
                        log_event(run_id, "OCR_DONE", "INFO",
                                  f"NOOP escalation OCR complete, new text_hash={text_sha256[:12]}")
                    else:
                        log_event(run_id, "OCR_DONE", "WARN",
                                  f"NOOP escalation failed to change geometry. "
                                  f"Recording as NOOP attempt.")

            # ── Language-agnostic quality report ──────────────────────
            prev_tokens = None
            if best_ocr_payload is not None:
                prev_tokens = [t for t in (best_ocr_payload["text"] or "").split() if t]
            ocr_quality_report = compute_quality_report(
                ocr_payload["text"],
                run_id=run_id,
                pass_idx=attempt_idx,
                previous_pass_tokens=prev_tokens,
            )
            hardened_quality_label = ocr_quality_report.quality_label
            insert_ocr_quality_report(run_id, ocr_quality_report.to_dict())

            log_event(run_id, "QUALITY_REPORT", "INFO",
                      f"attempt={attempt_idx} quality_label_v2={hardened_quality_label} "
                      f"gibberish={ocr_quality_report.gibberish_score:.4f} "
                      f"lead_frag={ocr_quality_report.leading_fragment_ratio:.4f} "
                      f"seam_frag={ocr_quality_report.seam_fragment_ratio:.4f} "
                      f"nwl={ocr_quality_report.non_wordlike_frac:.4f} "
                      f"entropy={ocr_quality_report.char_entropy:.4f}")

            # ── Enforce quality gates ─────────────────────────────────
            _lex_lang = ocr_payload.get("detected_language", "unknown")
            _lex_score = _lexical_plausibility(ocr_payload["text"], _lex_lang) if _lex_lang != "unknown" else None
            gate_decisions = enforce_quality_gates(
                ocr_quality_report, run_id=run_id,
                lexical_plausibility=_lex_score,
            )
            downstream_mode = gate_decisions["downstream_mode"]
            log_event(run_id, "QUALITY_GATES", "INFO", format_gate_report(gate_decisions))

            has_blocking_failure = bool(gate_decisions.get("blocked_stages"))
            all_gates_passed = all(
                g["passed"] for g in gate_decisions.get("gates", {}).values()
            )

            # ── Build unified EffectiveQuality ────────────────────────
            effective_quality = build_effective_quality(
                ocr_quality_report, gate_decisions,
                confidence=ocr_payload["confidence"],
            )

            # ── Record attempt in DB ──────────────────────────────────
            eff_frag = frag_gate_value(
                ocr_quality_report.leading_fragment_ratio,
                ocr_quality_report.seam_fragment_ratio,
            )
            retry_reason = "initial" if attempt_idx == 0 else (
                "seam_retry" if gate_decisions.get("seam_retry_required") else "quality_retry"
            )
            insert_ocr_attempt(run_id, {
                "attempt_idx": attempt_idx,
                "tiling_strategy": current_plan.strategy,
                "tile_grid": current_plan.grid,
                "overlap_pct": current_plan.overlap_pct,
                "tile_count": len(current_plan.tile_boxes) or raw_blob.get("tile_count"),
                "tile_boxes_json": current_plan.tile_boxes,
                "preproc_json": current_plan.preproc,
                "model_used": ocr_result.model_used,
                "text_sha256": text_sha256,
                "text_hash": text_sha256[:16],
                "quality_label": hardened_quality_label,
                "effective_quality_json": effective_quality.to_dict(),
                "gibberish_score": ocr_quality_report.gibberish_score,
                "leading_fragment_ratio": ocr_quality_report.leading_fragment_ratio,
                "seam_fragment_ratio": ocr_quality_report.seam_fragment_ratio,
                "non_wordlike_frac": ocr_quality_report.non_wordlike_frac,
                "char_entropy": ocr_quality_report.char_entropy,
                "uncertainty_density": ocr_quality_report.uncertainty_density,
                "cross_pass_stability": ocr_quality_report.cross_pass_stability,
                "gates_passed": all_gates_passed,
                "noop_detected": noop_detected,
                "decision": "NOOP" if noop_detected else ("PASS" if all_gates_passed else "FAIL"),
                "ocr_text": ocr_payload["text"][:2000] if ocr_payload["text"] else None,
                "detail_json": {
                    "downstream_mode": downstream_mode,
                    "blocked_stages": gate_decisions.get("blocked_stages", []),
                    "seam_retry_required": gate_decisions.get("seam_retry_required", False),
                    "noop_detected": noop_detected,
                    "reason": retry_reason,
                    "lead_frag": ocr_quality_report.leading_fragment_ratio,
                    "seam_frag": ocr_quality_report.seam_fragment_ratio,
                    "max_frag_used_for_gate": eff_frag,
                    "cross_pass_stability": ocr_quality_report.cross_pass_stability,
                    "thresholds": {
                        "LEADING_FRAG_HARD_LIMIT": LEADING_FRAG_HARD_LIMIT,
                        "SEAM_FRAG_HARD_LIMIT": SEAM_FRAG_HARD_LIMIT,
                        "CROSS_PASS_STABILITY_MIN": CROSS_PASS_STABILITY_MIN,
                    },
                },
            })

            # ── Track best attempt ────────────────────────────────────
            is_better = (
                best_quality_report is None
                or _quality_rank(hardened_quality_label) < _quality_rank(best_quality_report.quality_label)
                or (
                    _quality_rank(hardened_quality_label) == _quality_rank(best_quality_report.quality_label)
                    and ocr_quality_report.gibberish_score < best_quality_report.gibberish_score
                )
            )
            if is_better:
                best_ocr_result = ocr_result
                best_ocr_payload = ocr_payload
                best_quality_report = ocr_quality_report
                best_gate_decisions = gate_decisions
                best_effective_quality = effective_quality
                best_attempt_idx = attempt_idx

            if all_gates_passed:
                gates_ever_passed = True
                log_event(run_id, "QUALITY_GATES", "INFO",
                          f"All gates PASSED on attempt {attempt_idx}. "
                          f"Proceeding to downstream stages.")
                break  # no need to retry

            # Gates failed — decide whether to retry
            prev_text_sha256 = text_sha256
            prev_tiling_plan = current_plan

            if attempt_idx < MAX_OCR_ATTEMPTS - 1:
                if gate_decisions.get("seam_retry_required"):
                    log_event(run_id, "QUALITY_GATES", "INFO",
                              f"Gates FAILED on attempt {attempt_idx} "
                              f"(seam_retry_required). Scheduling retry with "
                              f"strategy selection (grid_shift/seam_band_crop/expand).")
                    update_run_fields(run_id, current_stage="SEAM_RETRY_PENDING")
                elif has_blocking_failure:
                    log_event(run_id, "QUALITY_GATES", "INFO",
                              f"Gates FAILED on attempt {attempt_idx} "
                              f"(blocked_stages={gate_decisions['blocked_stages']}). "
                              f"Scheduling retry.")
                    update_run_fields(run_id, current_stage="QUALITY_RETRY_PENDING")
                else:
                    log_event(run_id, "QUALITY_GATES", "INFO",
                              f"Quality {hardened_quality_label} on attempt "
                              f"{attempt_idx}. Scheduling retry.")
                    update_run_fields(run_id, current_stage="QUALITY_RETRY_PENDING")
            else:
                log_event(run_id, "QUALITY_GATES", "INFO",
                          f"All {MAX_OCR_ATTEMPTS} attempts exhausted. "
                          f"Best attempt={best_attempt_idx}, "
                          f"quality={best_quality_report.quality_label if best_quality_report else 'UNKNOWN'}.")

        # ══════════════════════════════════════════════════════════════
        # POST-RETRY: Use best attempt results for the rest of the pipeline
        # ══════════════════════════════════════════════════════════════
        assert best_ocr_result is not None
        assert best_ocr_payload is not None
        assert best_quality_report is not None
        assert best_gate_decisions is not None
        assert best_effective_quality is not None

        ocr_result = best_ocr_result
        ocr_payload = best_ocr_payload
        ocr_quality_report = best_quality_report
        gate_decisions = best_gate_decisions
        effective_quality = best_effective_quality
        hardened_quality_label = ocr_quality_report.quality_label
        downstream_mode = gate_decisions["downstream_mode"]

        # ── Cross-pass stability: compute if not yet done and borderline ──
        # If attempt 0 is not HIGH, or near fragment thresholds, or
        # seam_retry_required, compute stability via a controlled perturbation.
        if ocr_quality_report.cross_pass_stability < 0:
            needs_stability = (
                hardened_quality_label not in ("HIGH",)
                or ocr_quality_report.seam_retry_required
                or frag_gate_value(
                    ocr_quality_report.leading_fragment_ratio,
                    ocr_quality_report.seam_fragment_ratio,
                ) >= SEAM_FRAG_HARD_LIMIT * 0.7
            )
            if needs_stability and best_attempt_idx >= 0 and prev_tiling_plan is not None:
                # Use a small perturbation plan for the stability pass
                from app.services.seam_strategies import expand_overlap as _expand_for_stability
                stability_plan = _expand_for_stability(
                    current_plan.tile_boxes, img_w, img_h,
                    expand_pct=0.10, attempt_idx=99,
                )
                stability_dicts = plan_to_suggestions(stability_plan, img_w, img_h)
                stability_suggestions = [
                    SaiaOCRLocationSuggestion(**d) for d in stability_dicts
                ]
                try:
                    stability_result = await asyncio.to_thread(
                        _get_saia_ocr_agent().extract,
                        SaiaOCRRequest(
                            image_id=payload.image_id,
                            page_id=payload.page_id,
                            image_b64=payload.image_b64,
                            script_hint_seed=payload.script_hint_seed,
                            apply_proofread=False,
                            location_suggestions=stability_suggestions,
                        ),
                    )
                    stability_text = str(stability_result.text or "")
                    cross_pass_stab = compute_cross_pass_stability(
                        ocr_payload["text"], stability_text,
                    )
                    ocr_quality_report.cross_pass_stability = cross_pass_stab
                    # Update effective quality and gate decisions with new stability
                    _stab_lex_lang = ocr_payload.get("detected_language", "unknown")
                    _stab_lex_score = _lexical_plausibility(ocr_payload["text"], _stab_lex_lang) if _stab_lex_lang != "unknown" else None
                    gate_decisions = enforce_quality_gates(
                        ocr_quality_report, run_id=run_id,
                        lexical_plausibility=_stab_lex_score,
                    )
                    effective_quality = build_effective_quality(
                        ocr_quality_report, gate_decisions,
                        confidence=ocr_payload["confidence"],
                    )
                    hardened_quality_label = ocr_quality_report.quality_label
                    downstream_mode = gate_decisions["downstream_mode"]
                    log_event(run_id, "CROSS_PASS_STABILITY", "INFO",
                              f"Cross-pass stability computed: {cross_pass_stab:.4f} "
                              f"(threshold={CROSS_PASS_STABILITY_MIN})")

                    # ── Apply uncertainty markers if warranted ────────
                    eff_frag_val = frag_gate_value(
                        ocr_quality_report.leading_fragment_ratio,
                        ocr_quality_report.seam_fragment_ratio,
                    )
                    processed_text, markers_count = apply_uncertainty_markers(
                        ocr_payload["text"],
                        cross_pass_text=stability_text,
                        cross_pass_stability=cross_pass_stab,
                        frag_gate_val=eff_frag_val,
                        uncertainty_dens=ocr_quality_report.uncertainty_density,
                    )
                    if markers_count > 0:
                        ocr_payload["text"] = processed_text
                        ocr_payload["lines"] = processed_text.splitlines()
                        log_event(run_id, "UNCERTAINTY_ENFORCEMENT", "INFO",
                                  f"Inserted {markers_count} uncertainty markers.")
                except Exception as stab_exc:
                    log_event(run_id, "CROSS_PASS_STABILITY", "WARN",
                              f"Stability pass failed: {stab_exc}")

        # Overlay hardened fields
        ocr_payload["quality_label_v2"] = hardened_quality_label
        ocr_payload["downstream_mode"] = downstream_mode
        # Add seam warning if relevant
        if gate_decisions.get("seam_retry_required"):
            ocr_payload["warnings"] = list(dict.fromkeys(
                [*ocr_payload["warnings"], "SEAM_RETRY_REQUIRED"]
            ))

        update_run_fields(
            run_id,
            current_stage="OCR_DONE",
            script_hint=ocr_payload["script_hint"],
            detected_language=ocr_payload["detected_language"],
            confidence=ocr_payload["confidence"],
            warnings_json=json.dumps({
                "warnings": ocr_payload["warnings"],
                "quality_label": quality_label,
                "quality_label_v2": hardened_quality_label,
                "sanity_metrics": ocr_payload.get("sanity_metrics", {}),
            }, ensure_ascii=False),
            ocr_lines_json=json.dumps(ocr_payload["lines"], ensure_ascii=False),
            ocr_text=ocr_payload["text"],
        )

        # ══════════════════════════════════════════════════════════════
        # HARD GATE CHECK — stop if any blocking gate failed
        # If gates never passed after all attempts, mark FAILED_QUALITY
        # and return WITHOUT running downstream stages.
        # ══════════════════════════════════════════════════════════════
        has_blocking = bool(gate_decisions.get("blocked_stages"))

        if not gates_ever_passed and has_blocking:
            blocked_stages = gate_decisions.get("blocked_stages", [])
            failure_reason = (
                f"Quality gates FAILED after {MAX_OCR_ATTEMPTS} attempts. "
                f"quality_label_v2={hardened_quality_label}, "
                f"blocked_stages={blocked_stages}, "
                f"gibberish={ocr_quality_report.gibberish_score:.4f}, "
                f"lead_frag={ocr_quality_report.leading_fragment_ratio:.4f}, "
                f"seam_frag={ocr_quality_report.seam_fragment_ratio:.4f}, "
                f"nwl={ocr_quality_report.non_wordlike_frac:.4f}"
            )
            log_event(run_id, "QUALITY_BLOCKED", "ERROR", failure_reason)
            update_run_fields(
                run_id,
                status="FAILED_QUALITY",
                current_stage="QUALITY_BLOCKED",
                error=failure_reason,
                proofread_text=ocr_payload["text"],  # keep best text
            )

            return {
                "run_id": run_id,
                "status": "FAILED_QUALITY",
                "ocr_result": ocr_payload,
                "proofread_text": None,
                "detected_language": ocr_payload["detected_language"],
                "final_confidence": ocr_payload["confidence"],
                "quality_label": ocr_payload.get("quality_label", quality_label),
                "quality_label_v2": hardened_quality_label,
                "effective_quality": effective_quality.to_dict(),
                "downstream_mode": downstream_mode,
                "sanity_metrics": ocr_payload.get("sanity_metrics", {}),
                "quality_gates": gate_decisions,
                "ocr_attempts": list_ocr_attempts(run_id),
                "failure_reason": failure_reason,
                # Downstream stages NOT executed
                "chunks_count": 0,
                "mentions_count": 0,
                "top_mentions": [],
                "mention_report": None,
                "mention_recall": None,
                "linking_result": None,
                "consolidated_report": None,
            }

        # ══════════════════════════════════════════════════════════════
        # DOWNSTREAM: PROOFREAD (only if gates allow)
        # ══════════════════════════════════════════════════════════════
        log_event(run_id, "PROOFREAD_RUNNING", "START", "Calling proofreader agent.")
        update_run_fields(run_id, current_stage="PROOFREAD_RUNNING")

        proofread_text = ocr_payload["text"]
        if payload.apply_proofread and ocr_payload["text"].strip():
            proofreader = OcrProofreaderAgent(
                client=_get_saia_ocr_agent().client,
                model_override=ocr_result.model_used or None,
            )
            proofread_text = await asyncio.to_thread(
                proofreader.proofread,
                ocr_payload["text"],
                ocr_payload["script_hint"],
                ocr_payload["detected_language"],
            )
            if not str(proofread_text or "").strip():
                proofread_text = ocr_payload["text"]
                log_event(run_id, "PROOFREAD_RUNNING", "INFO",
                          "Proofreader returned empty; using OCR text.")
        else:
            log_event(run_id, "PROOFREAD_RUNNING", "INFO",
                      "Proofreader skipped by request or empty OCR.")

        # ── Proofreading quality guard ────────────────────────────────
        if payload.apply_proofread and str(proofread_text or "").strip() and proofread_text != ocr_payload["text"]:
            guarded_text, proof_accepted, proof_reason = proofreading_quality_guard(
                ocr_payload["text"], proofread_text, ocr_quality_report,
            )
            if not proof_accepted:
                proofread_text = guarded_text
                ocr_payload["warnings"] = list(dict.fromkeys(
                    [*ocr_payload["warnings"], f"PROOFREAD_REJECTED:{proof_reason}"]
                ))
                log_event(run_id, "PROOFREAD_RUNNING", "INFO",
                          f"Proofreading rejected: {proof_reason}")
            else:
                log_event(run_id, "PROOFREAD_RUNNING", "INFO",
                          f"Proofreading accepted: {proof_reason}")

        # ── Post-proofread quality gate (legacy sanity) ───────────────
        final_text_for_gate = str(proofread_text or "").strip()
        if final_text_for_gate:
            post_sanity = compute_sanity(final_text_for_gate)
            if post_sanity["single_char_ratio"] > 0.10 or post_sanity["weird_ratio"] > 0.15:
                gated_lines, mask_count = quality_gate_enforce(
                    [ln for ln in final_text_for_gate.splitlines() if ln.strip()]
                )
                if mask_count > 0:
                    proofread_text = "\n".join(gated_lines).strip()
                    ocr_payload["warnings"] = list(dict.fromkeys(
                        [*ocr_payload["warnings"], f"QUALITY_GATE_MASKED:{mask_count}_lines"]
                    ))
                    log_event(run_id, "PROOFREAD_RUNNING", "INFO",
                              f"Quality gate masked {mask_count} suspicious lines.")
                    post_sanity = compute_sanity(proofread_text)

            forced_lines, forced_count = force_uncertainty_markers(
                [ln for ln in (proofread_text or "").splitlines() if ln.strip()],
                post_sanity,
            )
            if forced_count > 0:
                proofread_text = "\n".join(forced_lines).strip()
                ocr_payload["warnings"] = list(dict.fromkeys(
                    [*ocr_payload["warnings"], f"FORCED_UNCERTAINTY:{forced_count}"]
                ))
                log_event(run_id, "PROOFREAD_RUNNING", "INFO",
                          f"Forced {forced_count} uncertainty markers.")
                post_sanity = compute_sanity(proofread_text)
            quality_label = _quality_label_from_sanity(post_sanity)
            sanity_metrics = post_sanity
            ocr_payload["confidence"] = sanity_adjust_confidence(
                ocr_payload["confidence"], post_sanity
            )
            ocr_payload["quality_label"] = quality_label
            ocr_payload["sanity_metrics"] = sanity_metrics
            sanity_summary = format_sanity_summary(post_sanity)
            ocr_payload["warnings"] = [
                w for w in ocr_payload["warnings"]
                if not str(w).startswith("SANITY ")
            ]
            ocr_payload["warnings"] = list(dict.fromkeys(
                [*ocr_payload["warnings"], sanity_summary]
            ))

        log_event(run_id, "PROOFREAD_DONE", "END", "Proofreading complete.")

        # ── Post-proofread quality report update ──────────────────────
        final_pipeline_text = str(proofread_text or "").strip() or str(ocr_payload["text"] or "")
        if final_pipeline_text and final_pipeline_text != ocr_payload["text"]:
            post_proof_report = compute_quality_report(
                final_pipeline_text,
                run_id=run_id,
                pass_idx=best_attempt_idx + 10,  # offset to distinguish from OCR passes
                previous_pass_tokens=[
                    t for t in (ocr_payload["text"] or "").split() if t
                ],
            )
            hardened_quality_label = post_proof_report.quality_label
            insert_ocr_quality_report(run_id, post_proof_report.to_dict())
            _pp_lex_lang = ocr_payload.get("detected_language", "unknown")
            _pp_lex_score = _lexical_plausibility(final_pipeline_text, _pp_lex_lang) if _pp_lex_lang != "unknown" else None
            gate_decisions = enforce_quality_gates(
                post_proof_report, run_id=run_id,
                lexical_plausibility=_pp_lex_score,
            )
            downstream_mode = gate_decisions["downstream_mode"]
            ocr_payload["quality_label_v2"] = hardened_quality_label
            ocr_payload["downstream_mode"] = downstream_mode
            ocr_quality_report = post_proof_report
            log_event(run_id, "QUALITY_REPORT", "INFO",
                      f"post_proofread quality_label_v2={hardened_quality_label}")

        update_run_fields(
            run_id,
            current_stage="PROOFREAD_DONE",
            proofread_text=proofread_text,
            confidence=ocr_payload["confidence"],
            warnings_json=json.dumps({
                "warnings": ocr_payload["warnings"],
                "quality_label": ocr_payload.get("quality_label", quality_label),
                "quality_label_v2": hardened_quality_label,
            }, ensure_ascii=False),
        )

        # ══════════════════════════════════════════════════════════════
        # DOWNSTREAM: ANALYZE (check ner_allowed)
        # ══════════════════════════════════════════════════════════════
        if not gate_decisions.get("ner_allowed", True):
            log_event(run_id, "ANALYZE_SKIPPED", "INFO",
                      f"NER/analysis SKIPPED: ner_allowed=False "
                      f"(quality={hardened_quality_label})")
            update_run_fields(run_id, current_stage="ANALYZE_SKIPPED")
            chunks: list[dict[str, Any]] = []
            mentions: list[dict[str, Any]] = []
            salvage_debug: dict[str, Any] = {"skipped": True, "reason": "ner_not_allowed"}
            chunks_count = 0
            mentions_count = 0
        else:
            log_event(run_id, "ANALYZE_RUNNING", "START",
                      "Running chunk/entity analysis.")
            update_run_fields(run_id, current_stage="ANALYZE_RUNNING")

            base_text_source = "proofread" if str(proofread_text or "").strip() else "ocr"
            base_text = str(proofread_text or "").strip() if base_text_source == "proofread" else str(ocr_payload["text"] or "")
            chunks, mentions, _candidates, salvage_debug = _run_trace_analysis(
                run_id, base_text,
            )
            chunks_count = len(chunks)
            mentions_count = len(mentions)

            # ── Mention recall check + high-recall fallback ───────────
            mention_recall_result = check_mention_recall(
                base_text, mentions_count, hardened_quality_label,
            )
            if not mention_recall_result["mention_recall_ok"]:
                ocr_payload["warnings"] = list(dict.fromkeys(
                    [*ocr_payload["warnings"], mention_recall_result["reason"]]
                ))
                log_event(run_id, "QUALITY_GATES", "INFO",
                          mention_recall_result["reason"])

            if mention_recall_result.get("trigger_high_recall"):
                hr_mentions = extract_high_recall_mentions(
                    base_text,
                    script=ocr_quality_report.script_family,
                    quality_label=hardened_quality_label,
                )
                if hr_mentions:
                    existing_spans = {
                        (m.get("start_offset"), m.get("end_offset"))
                        for m in mentions
                    }
                    new_hr = [
                        m for m in hr_mentions
                        if (m.get("start_offset"), m.get("end_offset")) not in existing_spans
                    ]
                    if new_hr:
                        for m in new_hr:
                            m["chunk_id"] = _assign_chunk_id_for_span(
                                int(m.get("start_offset", 0)),
                                int(m.get("end_offset", 0)),
                                chunks,
                            )
                        new_rows = insert_entity_mentions(run_id, new_hr)
                        mentions.extend(new_rows)
                        mentions_count = len(mentions)
                        log_event(run_id, "ANALYZE_RUNNING", "INFO",
                                  f"High-recall extraction added {len(new_rows)} mentions.")
                        salvage_debug["high_recall_added"] = len(new_rows)

            update_run_fields(
                run_id,
                current_stage="ANALYZE_DONE",
                base_text_source=base_text_source,
                chunks_count=chunks_count,
                mentions_count=mentions_count,
            )
            log_event(run_id, "ANALYZE_DONE", "END",
                      f"chunks={chunks_count}, mentions={mentions_count}")

        # Build mention extraction report
        mention_report = _build_mention_extraction_report(
            run_id, asset_ref, mentions, salvage_debug,
        ) if mentions or not salvage_debug.get("skipped") else None

        # ── Mention recall (for response) ─────────────────────────────
        if gate_decisions.get("ner_allowed", True):
            base_text = str(proofread_text or "").strip() or str(ocr_payload["text"] or "")
            mention_recall = check_mention_recall(
                base_text, mentions_count, hardened_quality_label,
            )
        else:
            mention_recall = {
                "mention_recall_ok": True,
                "reason": "NER skipped (ner_allowed=False)",
                "trigger_high_recall": False,
            }

        log_event(run_id, "STORED", "END", "Run results persisted in SQLite.")
        update_run_fields(run_id, current_stage="STORED", proofread_text=proofread_text)

        # ══════════════════════════════════════════════════════════════
        # DOWNSTREAM: AUTHORITY LINKING (check token_search_allowed)
        # ══════════════════════════════════════════════════════════════
        if not gate_decisions.get("token_search_allowed", True):
            log_event(run_id, "LINKING_SKIPPED", "INFO",
                      f"Authority linking SKIPPED: token_search_allowed=False "
                      f"(quality={hardened_quality_label})")
            linking_result = None
        else:
            linking_result = _run_authority_linking_stage(run_id)

        # Build consolidated report
        consolidated_report = _build_consolidated_report(
            run_id, asset_ref, mentions, salvage_debug, linking_result,
        ) if mentions or linking_result else None

        # ══════════════════════════════════════════════════════════════
        # DOWNSTREAM: AUTO-INDEX (check token_search_allowed)
        # ══════════════════════════════════════════════════════════════
        if not gate_decisions.get("token_search_allowed", True):
            log_event(run_id, "INDEX_SKIPPED", "INFO",
                      f"RAG auto-index SKIPPED: token_search_allowed=False "
                      f"(quality={hardened_quality_label})")
        else:
            _auto_index_run(run_id)

        log_event(run_id, "DONE", "END", "Pipeline completed successfully.")
        update_run_fields(run_id, status="COMPLETED", current_stage="DONE", error=None)

        return {
            "run_id": run_id,
            "status": "COMPLETED",
            "ocr_result": ocr_payload,
            "proofread_text": proofread_text,
            "detected_language": ocr_payload["detected_language"],
            "final_confidence": ocr_payload["confidence"],
            "quality_label": ocr_payload.get("quality_label", quality_label),
            "quality_label_v2": hardened_quality_label,
            "effective_quality": effective_quality.to_dict(),
            "downstream_mode": downstream_mode,
            "sanity_metrics": ocr_payload.get("sanity_metrics", {}),
            "quality_gates": gate_decisions,
            "ocr_attempts": list_ocr_attempts(run_id),
            "mention_recall": mention_recall,
            "chunks_count": chunks_count,
            "mentions_count": mentions_count,
            "top_mentions": mentions[:20],
            "mention_report": mention_report,
            "linking_result": linking_result,
            "consolidated_report": consolidated_report,
        }
    except Exception as exc:
        err = str(exc)
        update_run_fields(run_id, status="FAILED", current_stage="FAILED", error=err)
        log_event(run_id, "FAILED", "ERROR", err)
        raise HTTPException(status_code=500, detail={"run_id": run_id, "error": err}) from exc


@router.get("/ocr/trace/{run_id}")
async def ocr_trace_snapshot(run_id: str) -> dict[str, Any]:
    snapshot = _build_trace_snapshot(run_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return snapshot


@router.get("/ocr/trace/stream/{run_id}")
async def ocr_trace_stream(run_id: str) -> StreamingResponse:
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    async def event_stream():
        last_signature: tuple[Any, ...] | None = None
        while True:
            snapshot = _build_trace_snapshot(run_id)
            if snapshot is None:
                payload = {"run_id": run_id, "status": "FAILED", "error": "Run disappeared from database."}
                yield f"event: snapshot\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break

            run = snapshot.get("run") or {}
            events = snapshot.get("events") or []
            signature = (
                run.get("updated_at"),
                run.get("current_stage"),
                run.get("status"),
                len(events),
            )
            if signature != last_signature:
                yield f"event: snapshot\ndata: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                last_signature = signature

            if str(run.get("status") or "").upper() in {"COMPLETED", "FAILED"}:
                break
            await asyncio.sleep(0.4)

        yield "event: end\ndata: done\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/ocr/trace/{run_id}/tables")
async def ocr_trace_tables(run_id: str) -> dict[str, Any]:
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return {
        "run_id": run_id,
        "tables": [
            table_view_for_run(run_id),
            table_view_for_events(run_id),
            table_view_for_chunks(run_id),
            table_view_for_entity_mentions(run_id),
            table_view_for_entity_candidates(run_id),
            table_view_for_entity_decisions(run_id),
            table_view_for_entity_attempts(run_id),
        ],
    }
