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
    insert_entity_candidates,
    insert_entity_mentions,
    list_chunks,
    list_entity_mentions,
    list_events,
    log_event,
    table_view_for_chunks,
    table_view_for_entity_candidates,
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
from app.services.saia_client import SaiaConfigError

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


def _norm_surface(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _is_roman_numeral_token(token: str) -> bool:
    value = str(token or "").upper()
    return bool(value) and re.fullmatch(r"[IVXLCDM]+", value) is not None


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
        if line.strip():
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


def _extract_mentions_from_text(base_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    value = str(base_text or "")
    if not value.strip():
        return [], []

    candidates = [*_extract_person_mentions(value), *_extract_anchor_mentions(value)]
    mentions = _dedupe_mentions(value, candidates)

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
    return mentions, heuristic_candidates


def _run_trace_analysis(run_id: str, base_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    clear_analysis_for_run(run_id)
    chunks_input = _build_line_chunks(base_text)
    chunk_rows = insert_chunks(run_id, chunks_input)

    mentions_input, candidate_input = _extract_mentions_from_text(base_text)
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

    return chunk_rows, mention_rows, candidate_rows


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


@router.post("/ocr/page_with_trace")
async def ocr_page_with_trace(payload: SaiaFullPageExtractRequest) -> dict[str, Any]:
    """Run single-page OCR pipeline with trace logging. Reuses SaiaFullPageExtractRequest (image_b64 required)."""
    try:
        image_bytes = decode_image_bytes(payload.image_b64 or "")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image_b64 payload: {exc}") from exc

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
            log_event(
                run_id,
                "RECEIVED",
                "INFO",
                f"Segmentation location suggestions prepared: {len(location_suggestions)}",
            )
        except Exception as exc:
            base_warnings.append(f"SEGMENTATION_FAILED:{exc}")
            log_event(run_id, "RECEIVED", "ERROR", f"Segmentation failed: {exc}")
    else:
        log_event(
            run_id,
            "RECEIVED",
            "INFO",
            f"Using provided location suggestions: {len(location_suggestions)}",
        )

    try:
        log_event(run_id, "OCR_RUNNING", "START", "Calling SAIA OCR agent.")
        update_run_fields(run_id, status="RUNNING", current_stage="OCR_RUNNING")

        ocr_result = await asyncio.to_thread(
            _get_saia_ocr_agent().extract,
            SaiaOCRRequest(
                image_id=payload.image_id,
                page_id=payload.page_id,
                image_b64=payload.image_b64,
                script_hint_seed=payload.script_hint_seed,
                apply_proofread=False,
                location_suggestions=location_suggestions,
            ),
        )
        log_event(run_id, "OCR_DONE", "END", f"OCR complete with model: {ocr_result.model_used}")

        ocr_payload = {
            "lines": list(ocr_result.lines),
            "text": str(ocr_result.text or ""),
            "script_hint": str(ocr_result.script_hint or "unknown"),
            "detected_language": _normalize_detected_language(ocr_result.detected_language),
            "confidence": float(ocr_result.confidence) if ocr_result.confidence is not None else 0.0,
            "warnings": list(dict.fromkeys([*base_warnings, *ocr_result.warnings])),
        }

        update_run_fields(
            run_id,
            current_stage="OCR_DONE",
            script_hint=ocr_payload["script_hint"],
            detected_language=ocr_payload["detected_language"],
            confidence=ocr_payload["confidence"],
            warnings_json=json.dumps(ocr_payload["warnings"], ensure_ascii=False),
            ocr_lines_json=json.dumps(ocr_payload["lines"], ensure_ascii=False),
            ocr_text=ocr_payload["text"],
        )

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
                log_event(
                    run_id,
                    "PROOFREAD_RUNNING",
                    "INFO",
                    "Proofreader returned empty text; using OCR text as fallback.",
                )
        else:
            log_event(run_id, "PROOFREAD_RUNNING", "INFO", "Proofreader skipped by request or empty OCR.")

        log_event(run_id, "PROOFREAD_DONE", "END", "Proofreading complete.")
        update_run_fields(run_id, current_stage="PROOFREAD_DONE", proofread_text=proofread_text)

        log_event(run_id, "ANALYZE_RUNNING", "START", "Running chunk/entity analysis.")
        update_run_fields(run_id, current_stage="ANALYZE_RUNNING")

        base_text_source = "proofread" if str(proofread_text or "").strip() else "ocr"
        base_text = str(proofread_text or "").strip() if base_text_source == "proofread" else str(ocr_payload["text"] or "")
        chunks, mentions, _candidates = _run_trace_analysis(run_id, base_text)
        chunks_count = len(chunks)
        mentions_count = len(mentions)
        update_run_fields(
            run_id,
            current_stage="ANALYZE_DONE",
            base_text_source=base_text_source,
            chunks_count=chunks_count,
            mentions_count=mentions_count,
        )
        log_event(
            run_id,
            "ANALYZE_DONE",
            "END",
            f"analysis complete: chunks={chunks_count}, mentions={mentions_count}",
        )

        log_event(run_id, "STORED", "END", "Run results persisted in SQLite.")
        update_run_fields(run_id, current_stage="STORED", proofread_text=proofread_text)

        log_event(run_id, "DONE", "END", "Pipeline completed successfully.")
        update_run_fields(run_id, status="COMPLETED", current_stage="DONE", error=None)

        return {
            "run_id": run_id,
            "ocr_result": ocr_payload,
            "proofread_text": proofread_text,
            "detected_language": ocr_payload["detected_language"],
            "final_confidence": ocr_payload["confidence"],
            "chunks_count": chunks_count,
            "mentions_count": mentions_count,
            "top_mentions": mentions[:20],
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
        ],
    }
