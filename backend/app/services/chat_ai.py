"""Chat AI proxy utilities for GWDG OpenAI-compatible API."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
import logging
import os
import re
import time
from typing import Any

from app.config import settings
from app.services.model_router import (
    chat_stage_from_context,
    fallback_models_for_stage,
    get_task_model_assignments,
    model_for_chat_stage,
)

ARCHAI_SYSTEM_PROMPT = (
    "You are ArchAI, a manuscript research and extraction assistant. "
    "Be precise, cite uncertainty explicitly, and provide structured JSON when the user asks for structured outputs."
)
LEGACY_SAIA_MODEL_FALLBACKS = [
    "internvl3.5-30b-a3b",
    "mistral-large-3-675b-instruct-2512",
    "gemma-3-27b-it",
    "medgemma-27b-it",
]
_TRANSIENT_CHAT_RETRY_DELAYS = (0.75, 1.5)
log = logging.getLogger(__name__)


class ChatConfigError(RuntimeError):
    """Raised when Chat AI configuration is incomplete."""


def _require_api_key() -> str:
    key = str(
        settings.chat_ai_api_key
        or os.getenv("CHAT_AI_API_KEY", "")
        or settings.saia_api_key
        or settings.archai_chat_ai_api_key
        or settings.archai_saia_api_key
        or os.getenv("SAIA_API_KEY", "")
        or os.getenv("ARCHAI_CHAT_AI_API_KEY", "")
        or os.getenv("ARCHAI_SAIA_API_KEY", "")
        or ""
    ).strip()
    if not key:
        raise ChatConfigError(
            "Chat API key not configured. Set CHAT_AI_API_KEY (or SAIA_API_KEY) in backend/.env or environment."
        )
    return key


def _base_url() -> str:
    return str(
        settings.chat_ai_base_url
        or os.getenv("CHAT_AI_BASE_URL", "")
        or settings.saia_base_url
        or settings.archai_chat_ai_base_url
        or os.getenv("SAIA_BASE_URL", "")
        or "https://chat-ai.academiccloud.de/v1"
    ).strip().rstrip("/")


def _default_model(context: dict[str, Any] | None = None, *, prefer_vision: bool = False) -> str:
    return model_for_chat_stage(chat_stage_from_context(context), prefer_vision=prefer_vision)


def _create_client() -> Any:
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for chat proxy support.") from exc
    return OpenAI(api_key=_require_api_key(), base_url=_base_url())


def is_vision_model(model_id: str) -> bool:
    key = (model_id or "").lower()
    return any(token in key for token in ("vl", "internvl", "vision"))


def _raw_server_models(client: Any) -> list[str]:
    response = client.models.list()
    models: list[str] = []
    seen: set[str] = set()
    for item in getattr(response, "data", []) or []:
        model_id = str(getattr(item, "id", "") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append(model_id)
    return models


def _messages_include_image(messages: Sequence[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and str(block.get("type") or "").strip().lower() == "image_url":
                return True
    return False


def _is_model_not_found_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "model not found" in text
        or "unknown model" in text
        or "does not exist" in text
        or "invalid model" in text
    )


def _is_transient_upstream_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return any(
        token in text
        for token in (
            "502",
            "503",
            "504",
            "proxy error",
            "bad gateway",
            "gateway timeout",
            "error reading from remote server",
            "upstream server",
            "temporarily unavailable",
        )
    )


def _friendly_upstream_error(exc: Exception) -> str:
    if _is_transient_upstream_error(exc):
        return "Upstream chat provider temporarily failed (502/503 proxy error). Please retry."
    return str(exc)


def _choose_retry_model(
    requested_model: str,
    available_models: Sequence[str],
    prefer_vision: bool,
    stage: str | None = None,
) -> str | None:
    if not available_models:
        return None

    normalized = {model.lower(): model for model in available_models}
    for candidate in fallback_models_for_stage(stage, prefer_vision=prefer_vision):
        match = normalized.get(candidate.lower())
        if match and match != requested_model:
            return match

    configured_default = model_for_chat_stage(stage, prefer_vision=prefer_vision)
    default_match = normalized.get(configured_default.lower())
    if default_match and default_match != requested_model:
        return default_match

    for candidate in LEGACY_SAIA_MODEL_FALLBACKS:
        match = normalized.get(candidate.lower())
        if match and match != requested_model:
            return match

    if prefer_vision:
        for model in available_models:
            if is_vision_model(model) and model != requested_model:
                return model

    for model in available_models:
        if model != requested_model:
            return model
    return None


def _run_chat_call_with_retries(
    *,
    client: Any,
    selected_model: str,
    prefer_vision: bool,
    stage: str | None,
    runner: Any,
) -> tuple[Any, str]:
    transient_attempt = 0
    fallback_checked = False

    while True:
        try:
            return runner(selected_model), selected_model
        except Exception as exc:
            if _is_model_not_found_error(exc):
                available_models: list[str] = []
                try:
                    available_models = _raw_server_models(client)
                except Exception:
                    available_models = []
                retry_model = _choose_retry_model(
                    selected_model,
                    available_models,
                    prefer_vision,
                    stage=stage,
                )
                if not retry_model:
                    raise
                selected_model = retry_model
                transient_attempt = 0
                fallback_checked = True
                continue

            if _is_transient_upstream_error(exc):
                if transient_attempt < len(_TRANSIENT_CHAT_RETRY_DELAYS):
                    delay = _TRANSIENT_CHAT_RETRY_DELAYS[transient_attempt]
                    transient_attempt += 1
                    time.sleep(delay)
                    continue

                if not fallback_checked:
                    available_models = []
                    try:
                        available_models = _raw_server_models(client)
                    except Exception:
                        available_models = []
                    retry_model = _choose_retry_model(
                        selected_model,
                        available_models,
                        prefer_vision,
                        stage=stage,
                    )
                    if retry_model and retry_model != selected_model:
                        selected_model = retry_model
                        transient_attempt = 0
                        fallback_checked = True
                        continue

                raise RuntimeError(_friendly_upstream_error(exc)) from exc

            raise


def list_available_models() -> dict[str, Any]:
    assignments = get_task_model_assignments()
    models: list[str] = []
    listed_ok = False
    try:
        client = _create_client()
        models = _raw_server_models(client)
        listed_ok = True
    except ChatConfigError:
        raise
    except Exception:
        models = []

    seen: set[str] = set()
    unique_models: list[str] = []
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        unique_models.append(model)

    preferred_models = [
        assignments.chat_rag_model,
        assignments.translation_model,
        assignments.label_visual_model,
        assignments.label_visual_fallback_model,
        assignments.verifier_model,
    ]
    for preferred in reversed(preferred_models):
        if preferred and preferred not in seen:
            unique_models = [preferred, *unique_models]
            seen.add(preferred)
    if not listed_ok:
        for fallback in LEGACY_SAIA_MODEL_FALLBACKS:
            if fallback not in seen:
                unique_models.append(fallback)
                seen.add(fallback)

    return {
        "models": unique_models,
        "default_model": assignments.chat_rag_model,
        "vision_models": [model for model in unique_models if is_vision_model(model)],
        "task_models": assignments.as_payload(),
        "base_url": _base_url(),
    }


def _context_to_system_message(context: dict[str, Any] | None) -> str:
    if not context:
        return ARCHAI_SYSTEM_PROMPT

    lines = [ARCHAI_SYSTEM_PROMPT, "", "Document context:"]
    doc_id = context.get("document_id")
    filename = context.get("filename")
    page_index = context.get("current_page_index")
    page_total = context.get("page_count")
    transcript = context.get("transcript")
    authority_report = context.get("authority_report")
    ocr_run_id = context.get("ocr_run_id") or context.get("run_id")
    chat_stage = context.get("chat_stage")
    document_language = context.get("document_language")
    document_year = context.get("document_year")
    place_or_origin = context.get("place_or_origin")
    script_family = context.get("script_family")
    document_type = context.get("document_type")
    document_notes = context.get("document_notes")

    if doc_id is not None:
        lines.append(f"- document_id: {doc_id}")
    if ocr_run_id is not None:
        lines.append(f"- ocr_run_id: {ocr_run_id}")
    if chat_stage is not None:
        lines.append(f"- chat_stage: {chat_stage}")
    if filename is not None:
        lines.append(f"- filename: {filename}")
    if page_index is not None:
        lines.append(f"- current_page_index: {page_index}")
    if page_total is not None:
        lines.append(f"- page_count: {page_total}")
    if document_language:
        lines.append(f"- document_language: {document_language}")
    if document_year:
        lines.append(f"- document_year: {document_year}")
    if place_or_origin:
        lines.append(f"- place_or_origin: {place_or_origin}")
    if script_family:
        lines.append(f"- script_family: {script_family}")
    if document_type:
        lines.append(f"- document_type: {document_type}")
    if document_notes:
        lines.append("- document_notes:")
        lines.append(str(document_notes)[:1000])
    if transcript:
        lines.append("- transcript_snippet:")
        lines.append(str(transcript)[:1500])
    if authority_report:
        lines.append("- authority_report_snippet:")
        lines.append(str(authority_report)[:2000])
    if str(chat_stage or "").strip().lower() == "translation":
        source_language = str(document_language or "the source language").strip() or "the source language"
        lines.append("- translation_stage_instructions:")
        lines.append(
            f"Translate the authoritative transcript from {source_language} into English as a coherent passage. "
            "Use passage-level context to resolve likely orthographic, scribal, or OCR distortions when the intended sense is reasonably clear. "
            "Do not produce a token-by-token gloss, do not describe the manuscript page, do not inflate a damaged passage into a longer narrative than the source supports, and return only the English translation."
        )
        if "old french" in source_language.lower():
            lines.append(
                "- old_french_translation_note: Treat spelling variation and inflectional variation as expected features of Old French and prefer the best contextually supported modern English rendering."
            )
        lines.append(
            "- translation_scope_note: Keep the translation roughly proportional to the source passage. Do not turn unclear tokens into confident people, places, or plot details unless the passage clearly supports them."
        )
        lines.append(
            "- translation_uncertainty_note: If a clause remains too corrupt to interpret confidently, compress that local span with [unclear] instead of inventing narrative connective tissue, repeated moral commentary, or translator notes."
        )
    if str(chat_stage or "").strip().lower() == "entity_qa":
        lines.append("- entity_answer_style:")
        lines.append(
            "List linked entities first with source IDs and confidence, then unresolved mentions with reasons. "
            "Do not dump raw audit traces or internal reports."
        )

    return "\n".join(lines)


# ── RAG evidence injection ──────────────────────────────────────────────

_RAG_INSTRUCTION = (
    "\n\nYou have been given three evidence strata: OCR_CHUNK_EVIDENCE, "
    "LINKED_ENTITY_EVIDENCE, and UNRESOLVED_MENTION_EVIDENCE. "
    "When quoting or paraphrasing manuscript text you MUST cite like: "
    "(asset_ref=..., run_id=..., chunk_id=..., offsets=start-end).  "
    "Use LINKED_ENTITY_EVIDENCE for canonical identity, authority IDs, "
    "and concise entity summaries. Use UNRESOLVED_MENTION_EVIDENCE to "
    "state uncertainty plainly instead of inventing identifications. "
    "Do not invent quotes outside the evidence.  If none of the "
    "evidence is relevant, say so explicitly and answer from general "
    "knowledge."
)

_RAG_INSPECTION_RE = re.compile(
    r"\b(print|show|display|inspect)\b.*\b(rag|retrieval|evidence)\b|\bwhat did you retrieve\b",
    re.IGNORECASE,
)
_RAG_INSPECTION_DEBUG_RE = re.compile(r"\b(debug|detailed|verbose|json)\b", re.IGNORECASE)


def _build_stage_metadata(
    *,
    stage_name: str,
    model_used: str,
    mode_used: str | None = None,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage_name": str(stage_name or "").strip() or "rag_chat",
        "model_used": str(model_used or "").strip(),
    }
    if mode_used:
        payload["mode_used"] = str(mode_used).strip()
    if duration_ms is not None:
        payload["duration_ms"] = round(float(duration_ms), 1)
    return payload


def _evidence_tag_counts(evidence_text: str) -> dict[str, int]:
    text = str(evidence_text or "")
    return {
        "ocr_chunk_evidence": len(re.findall(r"\[(?:OCR_CHUNK_EVIDENCE|EVIDENCE)\]", text)),
        "linked_entity_evidence": len(re.findall(r"\[LINKED_ENTITY_EVIDENCE\]", text)),
        "unresolved_mention_evidence": len(re.findall(r"\[UNRESOLVED_MENTION_EVIDENCE\]", text)),
    }


def _translation_inspection_payload(
    *,
    context: dict[str, Any] | None,
    output_text: str,
    stage_metadata: dict[str, Any],
) -> dict[str, Any]:
    context = context or {}
    metadata_hints = {
        key: value
        for key, value in {
            "language": context.get("document_language"),
            "year": context.get("document_year"),
            "origin": context.get("place_or_origin"),
            "script": context.get("script_family"),
            "document_type": context.get("document_type"),
            "notes": context.get("document_notes"),
        }.items()
        if str(value or "").strip()
    }
    transcript = str(context.get("transcript") or "")
    return {
        "input_source_summary": {
            "source_type": "extracted_transcript",
            "ocr_run_id": str(context.get("ocr_run_id") or context.get("run_id") or "") or None,
            "transcript_chars": len(transcript),
            "source_language": str(context.get("document_language") or "") or None,
            "target_language": "English",
            "metadata_hints": metadata_hints,
        },
        "model_used": stage_metadata.get("model_used"),
        "evidence_used": {
            "ocr_transcript_used": bool(transcript),
            "page_image_used": False,
            "metadata_hint_keys": list(metadata_hints.keys()),
        },
        "final_output": output_text,
        "confidence_or_assessment": None,
        "stage_metadata": stage_metadata,
    }


def _normalize_translation_output(text: str, *, source_text: str = "") -> str:
    lines = [str(line or "").strip().strip('"') for line in str(text or "").replace("\r", "\n").split("\n")]
    filtered: list[str] = []
    for line in lines:
        if not line:
            continue
        line = re.sub(r"(?is)```(?:json)?|```", " ", line).strip()
        line = re.sub(
            r"(?i)^\s*(here(?: is|'s)? (?:the )?translation|translation|tentative translation|english translation|modern english|english rendering|fluent translation|old french to english(?: translation)?)\s*:\s*",
            "",
            line,
        ).strip()
        lower = line.lower()
        if lower.startswith(
            (
                "user request:",
                "ocr-extracted source text:",
                "source text:",
                "source language:",
                "target language:",
                "translation note:",
                "translator's note:",
                "metadata:",
                "context:",
                "explanation:",
                "commentary:",
            )
        ):
            continue
        if re.match(r"(?i)^(the|this)\s+(page|image|manuscript)\b", line) and len(lines) > 1:
            continue
        filtered.append(line)

    body = re.sub(r"\s+", " ", " ".join(filtered)).strip()
    body = re.sub(
        r"(?i)^\s*(translation|tentative translation|english translation|modern english|english rendering)\s*:\s*",
        "",
        body,
    ).strip()
    body = re.sub(r"(?is)\s+\bnote:\s+.*$", "", body).strip()
    body = re.sub(r"(?is)\s+\btranslator'?s note:\s+.*$", "", body).strip()
    if source_text:
        source_norm = re.sub(r"\s+", " ", str(source_text or "")).strip().lower()
        if source_norm and body.lower() == source_norm:
            body = "[unclear]"
    if not body:
        body = "[unclear]"
    return body


def _chat_inspection_payload(
    *,
    context: dict[str, Any] | None,
    question: str,
    output_text: str,
    evidence_text: str,
    stage_metadata: dict[str, Any],
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    context = context or {}
    evidence_counts = _evidence_tag_counts(evidence_text)
    return {
        "input_source_summary": {
            "chat_stage": chat_stage_from_context(context),
            "question": _short_text(question, limit=180),
            "ocr_run_id": str(context.get("ocr_run_id") or context.get("run_id") or "") or None,
            "document_id": str(context.get("document_id") or "") or None,
        },
        "model_used": stage_metadata.get("model_used"),
        "evidence_used": {
            **evidence_counts,
            "authority_report_present": bool(str(context.get("authority_report") or "").strip()),
        },
        "final_output": output_text,
        "confidence_or_assessment": verification.get("assessment") if verification else None,
        "stage_metadata": stage_metadata,
    }


def _rag_inspection_mode(query: str) -> str:
    return "debug" if _RAG_INSPECTION_DEBUG_RE.search(str(query or "")) else "presentation"


def _format_rag_presentation_markdown(payload: dict[str, Any]) -> str:
    lines = ["**RAG Evidence**", f"- Query: {payload.get('query', '')}", f"- Run ID: {payload.get('run_id', '')}"]
    document_metadata = dict(payload.get("document_metadata") or {})
    metadata_lines: list[str] = []
    for key in ("asset_ref", "detected_language"):
        value = str(document_metadata.get(key) or "").strip()
        if value:
            label = "Asset ref" if key == "asset_ref" else "Detected language"
            metadata_lines.append(f"- {label}: {value}")
    if metadata_lines:
        lines.extend(["", "**Document Metadata**", *metadata_lines])

    extraction_summary = dict(payload.get("extraction_summary") or {})
    extraction_lines = [
        f"- OCR chunk hits: {int(extraction_summary.get('ocr_chunk_hits', 0))}",
        f"- Linked entity hits: {int(extraction_summary.get('linked_entity_hits', 0))}",
        f"- Unresolved mention hits: {int(extraction_summary.get('unresolved_mention_hits', 0))}",
    ]
    lines.extend(["", "**Extraction Summary**", *extraction_lines])

    transcript_snippet = str(payload.get("transcript_snippet") or "").strip()
    lines.extend(["", "**Transcript Snippet**"])
    if transcript_snippet:
        lines.append(f"- {transcript_snippet}")
    else:
        lines.append("- None")

    lines.extend(["", "**Linked Entities**"])
    chunk_rows = list(payload.get("ocr_chunk_evidence", []) or [])
    linked_rows = list(payload.get("linked_entity_evidence", []) or [])
    if not linked_rows:
        lines.append("- None")
    for row in linked_rows[:4]:
        authority_bits = ", ".join(
            part
            for part in (
                f"Wikidata {row.get('wikidata_qid', '')}" if row.get("wikidata_qid") else "",
                f"VIAF {row.get('viaf_id', '')}" if row.get("viaf_id") else "",
                f"GeoNames {row.get('geonames_id', '')}" if row.get("geonames_id") else "",
            )
            if part
        )
        authority_suffix = f"; {authority_bits}" if authority_bits else ""
        lines.append(
            "- "
            f"{row.get('canonical_label', '') or row.get('mention_surface', '')} "
            f"({row.get('entity_type', '') or 'entity'}; confidence {_confidence_text(row.get('confidence', ''))}"
            f"{authority_suffix}) "
            f"[chunk {row.get('chunk_id', '')}, offsets {row.get('offsets', '')}]"
        )

    lines.extend(["", "**Unresolved Mentions**"])
    unresolved_rows = list(payload.get("unresolved_mention_evidence", []) or [])
    if not unresolved_rows:
        lines.append("- None")
    for row in unresolved_rows[:4]:
        lines.append(
            "- "
            f"{row.get('mention_surface', '')} "
            f"({row.get('probable_type', '') or 'unknown type'}; confidence {_confidence_text(row.get('confidence', ''))}) "
            f"unresolved because {_short_text(row.get('reason_unresolved', ''), limit=110)} "
            f"[chunk {row.get('chunk_id', '')}, offsets {row.get('offsets', '')}]"
        )

    uncertainty_notes = []
    if len(chunk_rows) > 4:
        uncertainty_notes.append(f"{len(chunk_rows) - 4} additional OCR chunk hit(s) omitted from presentation view.")
    if len(linked_rows) > 4:
        uncertainty_notes.append(f"{len(linked_rows) - 4} additional linked entity hit(s) omitted from presentation view.")
    if len(unresolved_rows) > 4:
        uncertainty_notes.append(f"{len(unresolved_rows) - 4} additional unresolved mention hit(s) omitted from presentation view.")
    if uncertainty_notes:
        lines.extend(["", "**Uncertainty Notes**"])
        for note in uncertainty_notes[:3]:
            lines.append(f"- {note}")
    return "\n".join(lines)


def _format_rag_debug_markdown(payload: dict[str, Any]) -> str:
    chunk_rows = list(payload.get("ocr_chunk_evidence", []) or [])
    linked_rows = list(payload.get("linked_entity_evidence", []) or [])
    unresolved_rows = list(payload.get("unresolved_mention_evidence", []) or [])
    debug_payload = {
        "query": payload.get("query", ""),
        "run_id": payload.get("run_id", ""),
        "document_metadata": payload.get("document_metadata", {}),
        "summary": {
            "ocr_chunk_hits": len(chunk_rows),
            "linked_entity_hits": len(linked_rows),
            "unresolved_mention_hits": len(unresolved_rows),
        },
        "authority_summary": payload.get("authority_summary", {}),
        "transcript_snippet": payload.get("transcript_snippet", ""),
        "ocr_chunk_evidence_sample": chunk_rows[:4],
        "linked_entity_evidence_sample": linked_rows[:4],
        "unresolved_mention_evidence_sample": unresolved_rows[:4],
        "omitted_counts": {
            "ocr_chunk_evidence": max(0, len(chunk_rows) - 4),
            "linked_entity_evidence": max(0, len(linked_rows) - 4),
            "unresolved_mention_evidence": max(0, len(unresolved_rows) - 4),
        },
    }
    lines = [
        "**RAG Debug View**",
        f"- Query: {payload.get('query', '')}",
        f"- Run ID: {payload.get('run_id', '')}",
        f"- OCR chunk hits: {len(chunk_rows)}",
        f"- Linked entity hits: {len(linked_rows)}",
        f"- Unresolved mention hits: {len(unresolved_rows)}",
        "",
        "```json",
        json.dumps(debug_payload, ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines)


def _query_terms(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(token) >= 2}


def _latest_user_query(messages: Sequence[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if str(msg.get("role") or "").lower() != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and str(block.get("type", "")).lower() == "text"
            ]
            return " ".join(parts).strip()
    return ""


def _latest_substantive_user_query(messages: Sequence[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if str(msg.get("role") or "").lower() != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            candidate = content.strip()
        elif isinstance(content, list):
            candidate = " ".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and str(block.get("type", "")).lower() == "text"
            ).strip()
        else:
            candidate = ""
        if candidate and not _is_rag_inspection_query(candidate):
            return candidate
    return _latest_user_query(messages)


def _short_text(value: Any, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _confidence_text(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return ""


def _display_unresolved_type(raw_type: Any, reason: Any) -> str:
    ent_type = str(raw_type or "").strip().lower() or "unknown"
    reason_text = str(reason or "")
    if ent_type == "place" and "low_evidence_place" in reason_text:
        if "lexical=True" in reason_text:
            return "lexical/unknown"
        if "context=False" in reason_text:
            return "possible_place_unresolved"
    return ent_type


def _entity_hit_lookup(entity_hits: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    entity_scores: dict[str, float] = {}
    mention_scores: dict[str, float] = {}
    for hit in entity_hits:
        score = round(1.0 - float(hit.get("distance", 1.0)), 4)
        entity_id = str(hit.get("entity_id") or "").strip()
        mention_id = str(hit.get("mention_id") or "").strip()
        if entity_id:
            entity_scores[entity_id] = max(score, entity_scores.get(entity_id, 0.0))
        if mention_id:
            mention_scores[mention_id] = max(score, mention_scores.get(mention_id, 0.0))
    return entity_scores, mention_scores


def _structured_entity_sections(query: str, run_id: str, entity_hits: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    try:
        from app.db import pipeline_db
    except Exception:  # noqa: BLE001
        rows = []
    else:
        rows = pipeline_db.list_mention_links_for_run(run_id)
    if not rows:
        linked_from_hits: list[dict[str, Any]] = []
        unresolved_from_hits: list[dict[str, Any]] = []
        for hit in entity_hits or []:
            score = round(1.0 - float(hit.get("distance", 1.0)), 4)
            if str(hit.get("record_type") or "") == "unresolved_mention" or str(hit.get("link_status") or "").lower() != "linked":
                unresolved_from_hits.append(
                    {
                        "run_id": hit.get("run_id", ""),
                        "asset_ref": hit.get("asset_ref", ""),
                        "mention_id": hit.get("mention_id", ""),
                        "mention_surface": hit.get("mention_surface", ""),
                        "probable_type": _display_unresolved_type(
                            hit.get("entity_type", ""),
                            hit.get("reason_unresolved", ""),
                        ),
                        "reason_unresolved": _short_text(hit.get("reason_unresolved") or "", limit=120),
                        "confidence": float(hit.get("confidence") or 0.0),
                        "retrieval_score": score,
                        "chunk_id": hit.get("chunk_id", ""),
                        "offsets": f"{hit.get('start_offset', 0)}-{hit.get('end_offset', 0)}",
                        "bbox": [],
                        "evidence_excerpt": _short_text(hit.get("text") or ""),
                    }
                )
            else:
                mention_surface = str(hit.get("mention_surface") or "").strip()
                linked_from_hits.append(
                    {
                        "run_id": hit.get("run_id", ""),
                        "asset_ref": hit.get("asset_ref", ""),
                        "entity_id": hit.get("entity_id", ""),
                        "canonical_label": hit.get("canonical_label", ""),
                        "mention_surfaces": [mention_surface] if mention_surface else [],
                        "entity_type": hit.get("entity_type", ""),
                        "authority_source": hit.get("authority_source", ""),
                        "authority_id": hit.get("authority_id", ""),
                        "wikidata_qid": hit.get("wikidata_qid", ""),
                        "viaf_id": hit.get("viaf_id", ""),
                        "geonames_id": hit.get("geonames_id", ""),
                        "confidence": float(hit.get("confidence") or 0.0),
                        "retrieval_score": score,
                        "chunk_id": hit.get("chunk_id", ""),
                        "offsets": f"{hit.get('start_offset', 0)}-{hit.get('end_offset', 0)}",
                        "bbox": [],
                        "evidence_excerpt": _short_text(hit.get("text") or ""),
                    }
                )
        return {"linked": linked_from_hits[:4], "unresolved": unresolved_from_hits[:4]}

    terms = _query_terms(query)
    entity_scores, mention_scores = _entity_hit_lookup(entity_hits or [])
    preferred_entity_ids = set(entity_scores)
    preferred_mention_ids = set(mention_scores)

    def _matches_query(row: dict[str, Any]) -> bool:
        if not terms:
            return True
        haystack = " ".join(
            [
                str(row.get("surface") or ""),
                str(row.get("canonical_label") or ""),
                str(row.get("entity_type") or ""),
                str(row.get("ent_type") or ""),
                str(row.get("reason") or ""),
                str(row.get("link_status") or ""),
            ]
        ).lower()
        return any(term in haystack for term in terms)

    linked_rows = [row for row in rows if str(row.get("link_status") or "").lower() == "linked"]
    unresolved_rows = [row for row in rows if str(row.get("link_status") or "").lower() != "linked"]

    if preferred_entity_ids or preferred_mention_ids:
        prioritized_linked = [
            row for row in linked_rows
            if str(row.get("entity_id") or "") in preferred_entity_ids
            or str(row.get("mention_id") or "") in preferred_mention_ids
        ]
        prioritized_unresolved = [
            row for row in unresolved_rows
            if str(row.get("mention_id") or "") in preferred_mention_ids
        ]
    else:
        prioritized_linked = [row for row in linked_rows if _matches_query(row)]
        prioritized_unresolved = [row for row in unresolved_rows if _matches_query(row)]

    if not prioritized_linked:
        prioritized_linked = sorted(
            linked_rows,
            key=lambda row: float(row.get("confidence") or 0.0),
            reverse=True,
        )[:4]
    if not prioritized_unresolved:
        prioritized_unresolved = sorted(
            unresolved_rows,
            key=lambda row: float(row.get("confidence") or 0.0),
            reverse=True,
        )[:4]

    grouped_linked: dict[str, dict[str, Any]] = {}
    for row in prioritized_linked:
        entity_id = str(row.get("entity_id") or "").strip() or f"mention:{row.get('mention_id', '')}"
        current = grouped_linked.get(entity_id)
        confidence = float(row.get("confidence") or 0.0)
        row_entity_score = entity_scores.get(entity_id, 0.0)
        mention_surface = str(row.get("surface") or "").strip()
        if not current:
            grouped_linked[entity_id] = {
                "run_id": row.get("run_id", ""),
                "asset_ref": row.get("asset_ref", ""),
                "entity_id": entity_id,
                "canonical_label": str(row.get("canonical_label") or mention_surface),
                "entity_type": str(row.get("entity_type") or row.get("ent_type") or ""),
                "authority_source": str(row.get("authority_source") or ""),
                "authority_id": str(row.get("authority_id") or ""),
                "wikidata_qid": str(row.get("wikidata_qid") or ""),
                "viaf_id": str(row.get("viaf_id") or ""),
                "geonames_id": str(row.get("geonames_id") or ""),
                "confidence": confidence,
                "retrieval_score": row_entity_score,
                "chunk_id": str(row.get("chunk_id") or ""),
                "offsets": f"{row.get('evidence_start_offset', row.get('start_offset', 0))}-{row.get('evidence_end_offset', row.get('end_offset', 0))}",
                "bbox": row.get("bbox_json") if isinstance(row.get("bbox_json"), list) else [],
                "evidence_excerpt": _short_text(row.get("evidence_raw_text") or ""),
                "mention_surfaces": [mention_surface] if mention_surface else [],
            }
        else:
            current["confidence"] = max(confidence, float(current.get("confidence") or 0.0))
            current["retrieval_score"] = max(row_entity_score, float(current.get("retrieval_score") or 0.0))
            if mention_surface and mention_surface not in current["mention_surfaces"]:
                current["mention_surfaces"].append(mention_surface)

    unresolved_payload: list[dict[str, Any]] = []
    for row in prioritized_unresolved[:4]:
        mention_id = str(row.get("mention_id") or "").strip()
        unresolved_payload.append(
            {
                "run_id": row.get("run_id", ""),
                "asset_ref": row.get("asset_ref", ""),
                "mention_id": mention_id,
                "mention_surface": str(row.get("surface") or ""),
                "probable_type": _display_unresolved_type(
                    row.get("ent_type") or "",
                    row.get("reason") or "",
                ),
                "reason_unresolved": _short_text(row.get("reason") or "", limit=120),
                "confidence": float(row.get("confidence") or 0.0),
                "retrieval_score": mention_scores.get(mention_id, 0.0),
                "chunk_id": str(row.get("chunk_id") or ""),
                "offsets": f"{row.get('evidence_start_offset', row.get('start_offset', 0))}-{row.get('evidence_end_offset', row.get('end_offset', 0))}",
                "bbox": row.get("bbox_json") if isinstance(row.get("bbox_json"), list) else [],
                "evidence_excerpt": _short_text(row.get("evidence_raw_text") or ""),
            }
        )

    linked_payload = sorted(
        grouped_linked.values(),
        key=lambda row: (float(row.get("retrieval_score") or 0.0), float(row.get("confidence") or 0.0)),
        reverse=True,
    )[:4]
    return {"linked": linked_payload, "unresolved": unresolved_payload}


def _format_linked_entity_evidence(section: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for row in section:
        authority_bits = [
            f"authority_source: {row.get('authority_source', '')}",
            f"authority_id: {row.get('authority_id', '')}",
            f"wikidata_qid: {row.get('wikidata_qid', '')}",
            f"viaf_id: {row.get('viaf_id', '')}",
            f"geonames_id: {row.get('geonames_id', '')}",
        ]
        bbox = row.get("bbox") or []
        blocks.append(
            "[LINKED_ENTITY_EVIDENCE]\n"
            f"run_id: {row.get('run_id', '')}\n"
            f"asset_ref: {row.get('asset_ref', '')}\n"
            f"entity_id: {row.get('entity_id', '')}\n"
            f"canonical_label: {row.get('canonical_label', '')}\n"
            f"mention_surfaces: {', '.join(row.get('mention_surfaces', [])[:4])}\n"
            f"entity_type: {row.get('entity_type', '')}\n"
            + "\n".join(authority_bits)
            + "\n"
            f"link_confidence: {_confidence_text(row.get('confidence', ''))}\n"
            f"retrieval_score: {_confidence_text(row.get('retrieval_score', ''))}\n"
            f"chunk_id: {row.get('chunk_id', '')}\n"
            f"offsets: {row.get('offsets', '')}\n"
            f"bbox: {','.join(str(item) for item in bbox) if bbox else ''}\n"
            f"evidence_excerpt: {row.get('evidence_excerpt', '')}\n"
            "[/LINKED_ENTITY_EVIDENCE]"
        )
    return "\n\n".join(blocks)


def _format_unresolved_mention_evidence(section: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for row in section:
        bbox = row.get("bbox") or []
        blocks.append(
            "[UNRESOLVED_MENTION_EVIDENCE]\n"
            f"run_id: {row.get('run_id', '')}\n"
            f"asset_ref: {row.get('asset_ref', '')}\n"
            f"mention_id: {row.get('mention_id', '')}\n"
            f"mention_surface: {row.get('mention_surface', '')}\n"
            f"probable_type: {row.get('probable_type', '')}\n"
            f"reason_unresolved: {row.get('reason_unresolved', '')}\n"
            f"link_confidence: {_confidence_text(row.get('confidence', ''))}\n"
            f"retrieval_score: {_confidence_text(row.get('retrieval_score', ''))}\n"
            f"chunk_id: {row.get('chunk_id', '')}\n"
            f"offsets: {row.get('offsets', '')}\n"
            f"bbox: {','.join(str(item) for item in bbox) if bbox else ''}\n"
            f"evidence_excerpt: {row.get('evidence_excerpt', '')}\n"
            "[/UNRESOLVED_MENTION_EVIDENCE]"
        )
    return "\n\n".join(blocks)


def _is_rag_inspection_query(query: str) -> bool:
    return bool(_RAG_INSPECTION_RE.search(str(query or "")))


def _format_rag_inspection_markdown(payload: dict[str, Any]) -> str:
    return _format_rag_presentation_markdown(payload)


def _rag_inspection_response(messages: Sequence[dict[str, Any]], context: dict[str, Any] | None) -> dict[str, Any] | None:
    started_at = time.perf_counter()
    question = _latest_user_query(messages)
    if not _is_rag_inspection_query(question):
        return None
    mode_used = _rag_inspection_mode(question)
    model_used = get_task_model_assignments().embedding_model
    run_id = str((context or {}).get("ocr_run_id") or (context or {}).get("run_id") or "").strip()
    if not run_id:
        duration_ms = (time.perf_counter() - started_at) * 1000
        stage_metadata = _build_stage_metadata(
            stage_name="rag_inspection",
            model_used=model_used,
            mode_used=mode_used,
            duration_ms=duration_ms,
        )
        return {
            "text": "No OCR run is available for RAG inspection on this page.",
            "model": "rag-inspection",
            "context_used": bool(context),
            "stage_metadata": stage_metadata,
            "inspection": {
                "input_source_summary": {
                    "query": question,
                    "run_id": None,
                },
                "model_used": model_used,
                "evidence_used": {
                    "ocr_chunk_evidence": 0,
                    "linked_entity_evidence": 0,
                    "unresolved_mention_evidence": 0,
                },
                "final_output": "No OCR run is available for RAG inspection on this page.",
                "confidence_or_assessment": None,
                "stage_metadata": stage_metadata,
            },
            "verification": None,
        }
    query = _latest_substantive_user_query(messages)
    payload = build_rag_evidence_for_debug(query, run_id)
    duration_ms = (time.perf_counter() - started_at) * 1000
    stage_metadata = _build_stage_metadata(
        stage_name="rag_inspection",
        model_used=model_used,
        mode_used=mode_used,
        duration_ms=duration_ms,
    )
    text_key = "debug_markdown" if mode_used == "debug" else "presentation_markdown"
    inspection_text = str(payload.get(text_key) or payload.get("inspection_markdown") or "")
    return {
        "text": inspection_text,
        "model": "rag-inspection",
        "context_used": bool(context),
        "stage_metadata": stage_metadata,
        "inspection": {
            "input_source_summary": {
                "query": payload.get("query", ""),
                "run_id": payload.get("run_id", ""),
            },
            "model_used": model_used,
            "evidence_used": {
                "ocr_chunk_evidence": len(payload.get("ocr_chunk_evidence", [])),
                "linked_entity_evidence": len(payload.get("linked_entity_evidence", [])),
                "unresolved_mention_evidence": len(payload.get("unresolved_mention_evidence", [])),
            },
            "final_output": inspection_text,
            "confidence_or_assessment": None,
            "stage_metadata": stage_metadata,
        },
        "verification": None,
    }


def _format_structured_entity_evidence(query: str, run_id: str) -> str:
    sections = _structured_entity_sections(query, run_id)
    parts = [
        _format_linked_entity_evidence(sections["linked"]),
        _format_unresolved_mention_evidence(sections["unresolved"]),
    ]
    return "\n\n".join(part for part in parts if part)


def _retrieve_evidence(
    messages: Sequence[dict[str, Any]],
    context: dict[str, Any] | None,
) -> str:
    """Run RAG retrieval and return formatted evidence blocks (or ``""``).

    Looks at the latest user message as the query.  Optionally scopes
    to a specific ``run_id`` if one is supplied in *context*.
    """
    try:
        from app.services.rag_store import retrieve_chunks, retrieve_entities, format_evidence_blocks
    except Exception:  # noqa: BLE001
        return ""

    if chat_stage_from_context(context) == "translation":
        return ""

    query = _latest_user_query(messages)
    if not query:
        return ""

    run_ids: list[str] | None = None
    if context:
        rid = context.get("ocr_run_id") or context.get("run_id") or context.get("document_id")
        if rid:
            run_ids = [str(rid)]

    hits: list[dict[str, Any]] = []
    entity_hits: list[dict[str, Any]] = []
    try:
        hits = retrieve_chunks(query, run_ids=run_ids)
    except Exception:  # noqa: BLE001
        hits = []
    try:
        entity_hits = retrieve_entities(query, run_ids=run_ids)
    except Exception:  # noqa: BLE001
        entity_hits = []

    evidence_parts: list[str] = []
    if hits:
        evidence_parts.append(format_evidence_blocks(hits))

    # Debug logging
    if hits:
        try:
            from app.services.rag_debug import log_chat_evidence
            evidence_ids = [
                {
                    "chunk_id": h.get("chunk_id", ""),
                    "chunk_idx": h.get("chunk_idx", 0),
                    "offsets": f"{h.get('start_offset', 0)}-{h.get('end_offset', 0)}",
                }
                for h in hits
            ]
            log_chat_evidence(
                run_id=run_ids[0] if run_ids else None,
                asset_ref=hits[0].get("asset_ref") if hits else None,
                k=len(hits),
                evidence_ids=evidence_ids,
                query=query,
                full_evidence_text=evidence_parts[0],
            )
        except Exception:  # noqa: BLE001
            pass

    if run_ids:
        sections = _structured_entity_sections(query, run_ids[0], entity_hits)
        entity_evidence = "\n\n".join(
            part
            for part in (
                _format_linked_entity_evidence(sections["linked"]),
                _format_unresolved_mention_evidence(sections["unresolved"]),
            )
            if part
        )
        if entity_evidence:
            evidence_parts.append(entity_evidence)
        try:
            from app.db import pipeline_db

            mention_links = pipeline_db.list_mention_links_for_run(run_ids[0])
            linked_count = sum(
                1 for row in mention_links if str(row.get("link_status") or "").lower() == "linked"
            )
            unresolved_count = max(0, len(mention_links) - linked_count)
            log.info(
                "rag retrieval stage=%s run_id=%s chunk_hits=%d entity_hits=%d linked_mentions=%d unresolved_mentions=%d",
                chat_stage_from_context(context),
                run_ids[0],
                len(hits),
                len(entity_hits),
                linked_count,
                unresolved_count,
            )
        except Exception:
            pass
    else:
        log.info(
            "rag retrieval stage=%s chunk_hits=%d entity_hits=%d",
            chat_stage_from_context(context),
            len(hits),
            len(entity_hits),
        )

    return "\n\n".join(part for part in evidence_parts if part)


def build_rag_evidence_for_debug(
    query: str,
    run_id: str,
    k: int = 8,
) -> dict[str, Any]:
    """Return a debug payload showing the three RAG evidence strata."""
    started_at = time.perf_counter()
    from app.services.rag_store import retrieve_chunks, retrieve_entities, format_evidence_blocks
    from app.db import pipeline_db

    hits = retrieve_chunks(query, top_k=k, run_ids=[run_id])
    entity_hits = retrieve_entities(query, top_k=min(k, settings.rag_entity_top_k), run_ids=[run_id])
    run = pipeline_db.get_run(run_id) or {}
    sections = _structured_entity_sections(query, run_id, entity_hits)
    evidence_text = "\n\n".join(
        part
        for part in (
            format_evidence_blocks(hits) if hits else "",
            _format_linked_entity_evidence(sections["linked"]),
            _format_unresolved_mention_evidence(sections["unresolved"]),
        )
        if part
    )

    evidence_ids = [
        {
            "chunk_id": h.get("chunk_id", ""),
            "chunk_idx": h.get("chunk_idx", 0),
            "offsets": f"{h.get('start_offset', 0)}-{h.get('end_offset', 0)}",
        }
        for h in hits
    ]

    citation_example = ""
    if hits:
        h = hits[0]
        citation_example = (
            f"(asset_ref={h.get('asset_ref', '')}, "
            f"run_id={h.get('run_id', '')}, "
            f"chunk_id={h.get('chunk_id', '')}, "
            f"offsets={h.get('start_offset', 0)}-{h.get('end_offset', 0)})"
        )

    # Build individual evidence blocks for structured preview
    evidence_blocks: list[dict[str, Any]] = []
    for h in hits:
        evidence_blocks.append({
            "run_id": h.get("run_id", ""),
            "asset_ref": h.get("asset_ref", ""),
            "chunk_id": h.get("chunk_id", ""),
            "chunk_idx": h.get("chunk_idx", 0),
            "offsets": f"{h.get('start_offset', 0)}-{h.get('end_offset', 0)}",
            "text": h.get("text", ""),
        })
    payload = {
        "query": query,
        "run_id": run_id,
        "k": k,
        "hits": len(hits),
        "entity_hits": len(entity_hits),
        "extraction_summary": {
            "ocr_chunk_hits": len(hits),
            "linked_entity_hits": len(sections["linked"]),
            "unresolved_mention_hits": len(sections["unresolved"]),
        },
        "authority_summary": {
            "linked_entities": [
                {
                    "entity_id": row.get("entity_id", ""),
                    "canonical_label": row.get("canonical_label", ""),
                    "authority_source": row.get("authority_source", ""),
                    "authority_id": row.get("authority_id", ""),
                    "wikidata_qid": row.get("wikidata_qid", ""),
                    "viaf_id": row.get("viaf_id", ""),
                    "geonames_id": row.get("geonames_id", ""),
                    "confidence": row.get("confidence", 0.0),
                }
                for row in sections["linked"][:4]
            ],
            "unresolved_mentions": [
                {
                    "mention_id": row.get("mention_id", ""),
                    "mention_surface": row.get("mention_surface", ""),
                    "probable_type": row.get("probable_type", ""),
                    "reason_unresolved": row.get("reason_unresolved", ""),
                    "confidence": row.get("confidence", 0.0),
                }
                for row in sections["unresolved"][:4]
            ],
        },
        "document_metadata": {
            "asset_ref": str(run.get("asset_ref") or ""),
            "detected_language": str(run.get("detected_language") or ""),
        },
        "transcript_snippet": _short_text(
            " || ".join(str(h.get("text") or "").strip() for h in hits[:2] if str(h.get("text") or "").strip())
            or str(run.get("proofread_text") or run.get("ocr_text") or ""),
            limit=280,
        ),
        "evidence_ids": evidence_ids,
        "ocr_chunk_evidence": evidence_blocks,
        "linked_entity_evidence": sections["linked"],
        "unresolved_mention_evidence": sections["unresolved"],
        "citation_example": citation_example,
        "evidence_text": evidence_text,
        "rag_instruction": _RAG_INSTRUCTION.strip(),
    }
    payload["presentation_markdown"] = _format_rag_presentation_markdown(payload)
    payload["debug_markdown"] = _format_rag_debug_markdown(payload)
    payload["inspection_markdown"] = payload["presentation_markdown"]
    payload["stage_metadata"] = _build_stage_metadata(
        stage_name="rag_inspection",
        model_used=get_task_model_assignments().embedding_model,
        duration_ms=(time.perf_counter() - started_at) * 1000,
    )
    return payload


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role") or "user").strip().lower()
    if role not in {"system", "user", "assistant"}:
        role = "user"

    content = message.get("content")
    if isinstance(content, str):
        return {"role": role, "content": content}
    if isinstance(content, list):
        return {"role": role, "content": content}
    return {"role": role, "content": str(content or "")}


def _prepare_messages(
    messages: Sequence[dict[str, Any]],
    context: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str]:
    if not messages:
        raise ValueError("At least one message is required.")

    prepared = [_normalize_message(msg) for msg in messages]
    if chat_stage_from_context(context) == "translation":
        translation_messages: list[dict[str, Any]] = []
        for message in prepared:
            content = message.get("content")
            if not isinstance(content, list):
                translation_messages.append(message)
                continue

            text_blocks: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if str(block.get("type") or "").strip().lower() != "text":
                    continue
                text_value = str(block.get("text") or "")
                if text_value:
                    text_blocks.append(text_value)

            translation_messages.append(
                {
                    "role": message.get("role", "user"),
                    "content": "\n".join(text_blocks),
                }
            )
        prepared = translation_messages
    system_prompt = _context_to_system_message(context)

    # ── RAG evidence injection ──────────────────────────────────────
    evidence = _retrieve_evidence(prepared, context)
    if evidence:
        system_prompt += _RAG_INSTRUCTION + "\n\n" + evidence

    return [{"role": "system", "content": system_prompt}, *prepared], evidence


def _extract_choice_text(choice: Any) -> str:
    delta = getattr(choice, "delta", None)
    if delta is None:
        message = getattr(choice, "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        return str(content or "")

    content = getattr(delta, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if txt:
                    out.append(str(txt))
            else:
                txt = getattr(item, "text", None)
                if txt:
                    out.append(str(txt))
        return "".join(out)
    return ""


def _normalize_stream_delta(raw_delta: str, current_text: str) -> tuple[str, str]:
    """Convert cumulative or overlapping stream chunks into append-only deltas."""
    if not raw_delta:
        return "", current_text

    if not current_text:
        return raw_delta, raw_delta

    if raw_delta.startswith(current_text):
        return raw_delta[len(current_text):], raw_delta

    if current_text.endswith(raw_delta):
        return "", current_text

    max_overlap = min(len(current_text), len(raw_delta))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if current_text.endswith(raw_delta[:size]):
            overlap = size
            break

    append_delta = raw_delta[overlap:]
    return append_delta, current_text + append_delta


def _should_run_verification(context: dict[str, Any] | None, evidence_text: str) -> bool:
    if not bool(settings.paleography_verification_enabled):
        return False
    if not context:
        return False
    if chat_stage_from_context(context) == "translation":
        return False
    if not str(context.get("ocr_run_id") or context.get("run_id") or "").strip():
        return False
    return bool(str(evidence_text or "").strip())


def _run_paleography_verification(
    *,
    question: str,
    draft_text: str,
    context: dict[str, Any] | None,
    evidence_text: str,
) -> dict[str, Any] | None:
    if not _should_run_verification(context, evidence_text):
        return None

    def _degraded_verification(model_used: str) -> dict[str, Any]:
        stage_metadata = _build_stage_metadata(
            stage_name="verification",
            model_used=model_used,
            duration_ms=0.0,
        )
        return {
            "assessment": "unavailable",
            "corrected_answer": draft_text,
            "notes": ["Verifier response could not be parsed cleanly."],
            "citations_checked": [],
            "model_used": model_used,
            "stage_metadata": stage_metadata,
            "inspection": {
                "input_source_summary": {
                    "question": question,
                    "ocr_run_id": str((context or {}).get("ocr_run_id") or (context or {}).get("run_id") or "") or None,
                },
                "model_used": model_used,
                "evidence_used": {
                    "ocr_chunk_evidence": evidence_text.count("[OCR_CHUNK_EVIDENCE]"),
                    "linked_entity_evidence": evidence_text.count("[LINKED_ENTITY_EVIDENCE]"),
                    "unresolved_mention_evidence": evidence_text.count("[UNRESOLVED_MENTION_EVIDENCE]"),
                },
                "final_output": {
                    "assessment": "unavailable",
                    "verified_answer": draft_text,
                    "notes": ["Verifier response could not be parsed cleanly."],
                    "citations_checked": [],
                },
                "confidence_or_assessment": "unavailable",
                "stage_metadata": stage_metadata,
            },
        }

    try:
        from types import SimpleNamespace

        from app.agents.paleography_verification_agent import PaleographyVerificationAgent
    except Exception:
        return _degraded_verification(str(get_task_model_assignments().verifier_model or ""))

    agent = PaleographyVerificationAgent()
    try:
        result = agent.run(
            SimpleNamespace(
                question=question,
                draft_answer=draft_text,
                transcript=str((context or {}).get("transcript") or ""),
                authority_report=str((context or {}).get("authority_report") or ""),
                evidence_text=evidence_text,
                ocr_run_id=str((context or {}).get("ocr_run_id") or (context or {}).get("run_id") or ""),
            )
        )
        return {
            "assessment": result.assessment,
            "corrected_answer": result.corrected_answer,
            "notes": result.notes,
            "citations_checked": result.citations_checked,
            "model_used": result.model_used,
            "stage_metadata": result.stage_metadata,
            "inspection": result.inspection,
        }
    except Exception:
        fallback_model = ""
        try:
            fallback_model = str(agent._target_model() or "")
        except Exception:
            fallback_model = str(get_task_model_assignments().verifier_model or "")
        return _degraded_verification(fallback_model)


def _clean_verification_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"(?is)<\s*(think|analysis|scratchpad)\b[^>]*>.*?<\s*/\s*\1\s*>", " ", text)
    text = re.sub(
        r"(?im)^\s*(reasoning|analysis|scratchpad|chain[\s-]*of[\s-]*thought|internal reasoning|thought process)\s*:\s*.*$",
        " ",
        text,
    )
    text = re.sub(r"(?is)```(?:json)?|```", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:\n\t")


def _format_verification_appendix(verification: dict[str, Any] | None) -> str:
    if not verification:
        return ""
    notes = [_clean_verification_value(item) for item in list(verification.get("notes") or [])]
    notes = [item for item in notes if item]
    citations = [
        _clean_verification_value(item) for item in list(verification.get("citations_checked") or [])
    ]
    citations = [item for item in citations if item]
    corrected_answer = _clean_verification_value(verification.get("corrected_answer", ""))
    lines = [
        "",
        "",
        "[Verification]",
        f"Assessment: {_clean_verification_value(verification.get('assessment', 'partially_supported')) or 'partially_supported'}",
        f"Verified answer: {corrected_answer}",
        f"Notes: {' | '.join(notes[:4]) if notes else 'none'}",
        f"Citations checked: {' | '.join(citations[:6]) if citations else 'none'}",
    ]
    lines.append(f"Verifier model: {_clean_verification_value(verification.get('model_used', ''))}")
    return "\n".join(lines)


def create_chat_completion(
    messages: Sequence[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.2,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    stage = chat_stage_from_context(context)
    inspection = _rag_inspection_response(messages, context)
    if inspection is not None:
        return inspection
    client = _create_client()
    prepared_messages, evidence_text = _prepare_messages(messages, context)
    verification_question = _latest_user_query(messages)
    prefer_vision = _messages_include_image(prepared_messages)
    if stage == "translation":
        selected_model = _default_model(context, prefer_vision=False).strip()
    else:
        selected_model = (model or "").strip()
        if not selected_model:
            selected_model = _default_model(context, prefer_vision=prefer_vision).strip()
    effective_temperature = 0.0 if stage == "translation" else float(temperature)

    def _run_completion(target_model: str) -> Any:
        return client.chat.completions.create(
            model=target_model,
            messages=prepared_messages,
            temperature=effective_temperature,
            stream=False,
        )

    response, selected_model = _run_chat_call_with_retries(
        client=client,
        selected_model=selected_model,
        prefer_vision=prefer_vision,
        stage=stage,
        runner=_run_completion,
    )

    choices = list(getattr(response, "choices", []) or [])
    draft_text = _extract_choice_text(choices[0]) if choices else ""
    if stage == "translation":
        draft_text = _normalize_translation_output(
            draft_text,
            source_text=str((context or {}).get("transcript") or ""),
        )
    verification = _run_paleography_verification(
        question=verification_question,
        draft_text=draft_text,
        context=context,
        evidence_text=evidence_text,
    )
    text = draft_text + _format_verification_appendix(verification)
    duration_ms = (time.perf_counter() - started_at) * 1000
    stage_metadata = _build_stage_metadata(
        stage_name=stage,
        model_used=selected_model,
        duration_ms=duration_ms,
    )
    if stage == "translation":
        inspection_payload = _translation_inspection_payload(
            context=context,
            output_text=draft_text,
            stage_metadata=stage_metadata,
        )
    else:
        inspection_payload = _chat_inspection_payload(
            context=context,
            question=verification_question,
            output_text=draft_text,
            evidence_text=evidence_text,
            stage_metadata=stage_metadata,
            verification=verification,
        )
    log.info(
        "chat completion stage=%s model=%s prefer_vision=%s evidence_chars=%d verification=%s duration_ms=%.1f",
        stage,
        selected_model,
        prefer_vision,
        len(evidence_text),
        bool(verification),
        duration_ms,
    )
    return {
        "text": text,
        "model": selected_model,
        "context_used": bool(context),
        "stage_metadata": stage_metadata,
        "inspection": inspection_payload,
        "verification": verification,
    }


def stream_chat_completion(
    messages: Sequence[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.2,
    context: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    started_at = time.perf_counter()
    stage = chat_stage_from_context(context)
    inspection = _rag_inspection_response(messages, context)
    if inspection is not None:
        inspection_text = str(inspection.get("text") or "")
        if inspection_text:
            yield {"type": "delta", "delta": inspection_text}
        yield {
            "type": "done",
            "text": inspection_text,
            "model": str(inspection.get("model") or "rag-inspection"),
            "context_used": bool(context),
            "stage_metadata": inspection.get("stage_metadata"),
            "inspection": inspection.get("inspection"),
            "verification": None,
        }
        return
    client = _create_client()
    prepared_messages, evidence_text = _prepare_messages(messages, context)
    verification_question = _latest_user_query(messages)
    prefer_vision = _messages_include_image(prepared_messages)
    if stage == "translation":
        selected_model = _default_model(context, prefer_vision=False).strip()
    else:
        selected_model = (model or "").strip()
        if not selected_model:
            selected_model = _default_model(context, prefer_vision=prefer_vision).strip()
    effective_temperature = 0.0 if stage == "translation" else float(temperature)

    def _run_stream(target_model: str) -> Any:
        return client.chat.completions.create(
            model=target_model,
            messages=prepared_messages,
            temperature=effective_temperature,
            stream=True,
        )

    stream, selected_model = _run_chat_call_with_retries(
        client=client,
        selected_model=selected_model,
        prefer_vision=prefer_vision,
        stage=stage,
        runner=_run_stream,
    )

    full_text = ""
    if stage == "translation":
        for event in stream:
            choices = list(getattr(event, "choices", []) or [])
            if not choices:
                continue
            raw_delta = _extract_choice_text(choices[0])
            _delta, full_text = _normalize_stream_delta(raw_delta, full_text)
        full_text = _normalize_translation_output(
            full_text,
            source_text=str((context or {}).get("transcript") or ""),
        )
        if full_text:
            yield {"type": "delta", "delta": full_text}
    else:
        for event in stream:
            choices = list(getattr(event, "choices", []) or [])
            if not choices:
                continue
            raw_delta = _extract_choice_text(choices[0])
            delta, full_text = _normalize_stream_delta(raw_delta, full_text)
            if not delta:
                continue
            yield {"type": "delta", "delta": delta}

    verification = _run_paleography_verification(
        question=verification_question,
        draft_text=full_text,
        context=context,
        evidence_text=evidence_text,
    )
    appendix = _format_verification_appendix(verification)
    if appendix:
        full_text += appendix
        yield {"type": "delta", "delta": appendix}

    duration_ms = (time.perf_counter() - started_at) * 1000
    stage_metadata = _build_stage_metadata(
        stage_name=stage,
        model_used=selected_model,
        duration_ms=duration_ms,
    )
    if stage == "translation":
        inspection_payload = _translation_inspection_payload(
            context=context,
            output_text=full_text[: max(0, len(full_text) - len(appendix))] if appendix else full_text,
            stage_metadata=stage_metadata,
        )
    else:
        inspection_payload = _chat_inspection_payload(
            context=context,
            question=verification_question,
            output_text=full_text[: max(0, len(full_text) - len(appendix))] if appendix else full_text,
            evidence_text=evidence_text,
            stage_metadata=stage_metadata,
            verification=verification,
        )
    log.info(
        "chat stream stage=%s model=%s prefer_vision=%s evidence_chars=%d verification=%s duration_ms=%.1f",
        stage,
        selected_model,
        prefer_vision,
        len(evidence_text),
        bool(verification),
        duration_ms,
    )

    yield {
        "type": "done",
        "text": full_text,
        "model": selected_model,
        "context_used": bool(context),
        "stage_metadata": stage_metadata,
        "inspection": inspection_payload,
        "verification": verification,
    }
