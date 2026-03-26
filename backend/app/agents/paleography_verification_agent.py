from __future__ import annotations

import json
from dataclasses import dataclass
import re
import time
from typing import Any

from app.agents.base import BaseAgent
from app.config import settings
from app.services.model_router import get_task_model_assignments


PALEOGRAPHY_VERIFICATION_SYSTEM_PROMPT = "\n".join(
    [
        "You are ArchAI's paleography verification agent.",
        "You review a draft answer against transcript evidence, OCR evidence blocks, structured entity evidence, unresolved mention evidence, and authority data.",
        "Your job is to verify claims conservatively, with manuscript and paleographic caution.",
        "",
        "Verification rules:",
        "- Treat the transcript as the authoritative downstream text substrate, but still check whether the draft answer is actually supported by the supplied evidence.",
        "- Reject claims that are unsupported, contradicted, or over-interpreted.",
        "- Distinguish clearly between supported, partially supported, and unsupported claims.",
        "- Pay attention to paleographic uncertainty, abbreviation ambiguity, damaged text, OCR uncertainty markers, and unresolved entities.",
        "- Use authority IDs only when they are present in the evidence.",
        "- Prefer evidence-backed correction over confident speculation.",
        "- If entity or place identification is unresolved, say that explicitly rather than smoothing it away.",
        "- Do not mark an answer as supported merely because it echoes the transcript or repeats source-language wording.",
        "- A weak English interpretation or bad translation remains unsupported if the evidence does not justify the meaning claimed.",
        "- Never invent authority IDs, readings, dates, iconography, or identifications.",
        "",
        "Return strict JSON with keys:",
        '- "assessment": one of "supported", "partially_supported", "unsupported"',
        '- "corrected_answer": concise prose answer that stays within the evidence',
        '- "notes": array of short verification notes explaining support, uncertainty, or contradiction',
        '- "citations_checked": array of short citation strings or evidence labels',
    ]
)


@dataclass
class PaleographyVerificationResult:
    assessment: str
    corrected_answer: str
    notes: list[str]
    citations_checked: list[str]
    model_used: str
    duration_ms: float
    raw_text: str
    stage_metadata: dict[str, Any] | None = None
    inspection: dict[str, Any] | None = None


class PaleographyVerificationAgentError(RuntimeError):
    """Raised when the verification agent fails irrecoverably."""


_REASONING_BLOCK_RE = re.compile(
    r"(?is)<\s*(think|analysis|scratchpad)\b[^>]*>.*?<\s*/\s*\1\s*>"
)
_REASONING_LINE_RE = re.compile(
    r"(?im)^\s*(reasoning|analysis|scratchpad|chain[\s-]*of[\s-]*thought|internal reasoning|thought process)\s*:\s*.*$"
)
_LABELED_VALUE_RE = {
    "assessment": re.compile(r"(?im)^\s*assessment\s*:\s*(.+?)\s*$"),
    "verified_answer": re.compile(r"(?im)^\s*(?:verified answer|corrected answer|answer)\s*:\s*(.+?)\s*$"),
    "notes": re.compile(r"(?im)^\s*notes?\s*:\s*(.+?)\s*$"),
    "citations_checked": re.compile(r"(?im)^\s*citations?\s*checked\s*:\s*(.+?)\s*$"),
}


def _extract_json_blob(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    if "```" in text:
        for block in text.split("```"):
            candidate = block.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except Exception:
                continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _strip_reasoning_leaks(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = _REASONING_BLOCK_RE.sub(" ", text)
    text = _REASONING_LINE_RE.sub(" ", text)
    text = re.sub(r"(?is)```(?:json)?|```", " ", text)
    text = re.sub(r"(?i)\b(let'?s think step by step|hidden chain of thought|private reasoning)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:\n\t")


def _normalized_assessment(value: str) -> str:
    clean = str(value or "").strip().lower()
    if clean in {"supported", "partially_supported", "unsupported", "unavailable"}:
        return clean
    return "partially_supported"


def _clean_string_list(items: list[Any], *, limit: int) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = _strip_reasoning_leaks(str(item or ""))
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value[:300])
        if len(cleaned) >= limit:
            break
    return cleaned


def _split_labeled_list(raw: str) -> list[str]:
    text = _strip_reasoning_leaks(raw)
    if not text:
        return []
    parts = re.split(r"(?:\s*[;•]\s*|\s*,\s*|\s+\|\s+)", text)
    cleaned: list[str] = []
    for part in parts:
        value = _strip_reasoning_leaks(part)
        if value:
            cleaned.append(value)
    return cleaned


def _salvage_structured_fields(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    if not _strip_reasoning_leaks(text):
        return {}

    salvaged: dict[str, Any] = {}
    for key, pattern in _LABELED_VALUE_RE.items():
        match = pattern.search(text)
        if not match:
            continue
        value = _strip_reasoning_leaks(match.group(1))
        if not value:
            continue
        if key in {"notes", "citations_checked"}:
            salvaged[key] = _split_labeled_list(value)
        else:
            salvaged[key] = value

    return salvaged


def _normalized_verification_payload(raw: str, *, draft_answer: str) -> tuple[str, str, list[str], list[str]]:
    payload = _extract_json_blob(raw) or {}
    salvaged = _salvage_structured_fields(raw)

    assessment = _normalized_assessment(
        str(
            payload.get("assessment")
            or salvaged.get("assessment")
            or "unavailable"
        )
    )
    if assessment == "partially_supported" and not (payload or salvaged):
        assessment = "unavailable"

    corrected_answer = _strip_reasoning_leaks(
        str(
            payload.get("corrected_answer")
            or payload.get("verified_answer")
            or salvaged.get("verified_answer")
            or draft_answer
        )
    ) or draft_answer

    notes = _clean_string_list(
        list(payload.get("notes") or salvaged.get("notes") or []),
        limit=4,
    )
    citations_checked = _clean_string_list(
        list(payload.get("citations_checked") or salvaged.get("citations_checked") or []),
        limit=6,
    )

    if not (payload or salvaged):
        notes = ["Verifier response could not be parsed cleanly."]
        citations_checked = []
    elif not notes:
        notes = ["Verification completed without additional notes."]

    return assessment, corrected_answer, notes, citations_checked


def _stage_metadata(*, model_used: str, duration_ms: float) -> dict[str, Any]:
    return {
        "stage_name": "verification",
        "model_used": str(model_used or "").strip(),
        "duration_ms": round(float(duration_ms), 1),
    }


def _evidence_counts(evidence_text: str) -> dict[str, int]:
    text = str(evidence_text or "")
    return {
        "ocr_chunk_evidence": len(re.findall(r"\[OCR_CHUNK_EVIDENCE\]", text)),
        "linked_entity_evidence": len(re.findall(r"\[LINKED_ENTITY_EVIDENCE\]", text)),
        "unresolved_mention_evidence": len(re.findall(r"\[UNRESOLVED_MENTION_EVIDENCE\]", text)),
    }


def _inspection_payload(
    *,
    question: str,
    ocr_run_id: str,
    transcript: str,
    authority_report: str,
    evidence_text: str,
    assessment: str,
    corrected_answer: str,
    notes: list[str],
    citations_checked: list[str],
    stage_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "input_source_summary": {
            "question": question,
            "ocr_run_id": ocr_run_id or None,
            "transcript_chars": len(str(transcript or "")),
            "authority_report_chars": len(str(authority_report or "")),
        },
        "model_used": stage_metadata.get("model_used"),
        "evidence_used": {
            **_evidence_counts(evidence_text),
            "authority_report_present": bool(str(authority_report or "").strip()),
        },
        "final_output": {
            "assessment": assessment,
            "verified_answer": corrected_answer,
            "notes": notes[:4],
            "citations_checked": citations_checked[:6],
        },
        "confidence_or_assessment": assessment,
        "stage_metadata": stage_metadata,
    }


class PaleographyVerificationAgent(BaseAgent):
    name = "paleography-verification-agent"

    def _target_model(self) -> str:
        assignments = get_task_model_assignments()
        configured = str(settings.paleography_verification_model or "").strip()
        if configured:
            return configured
        return assignments.verifier_model

    def run(self, payload: Any) -> PaleographyVerificationResult:
        from app.services.chat_ai import (
            _create_client,
            _is_model_not_found_error,
        )

        started_at = time.perf_counter()

        question = str(getattr(payload, "question", "") or "").strip()
        draft_answer = str(getattr(payload, "draft_answer", "") or "").strip()
        transcript = str(getattr(payload, "transcript", "") or "").strip()
        authority_report = str(getattr(payload, "authority_report", "") or "").strip()
        evidence_text = str(getattr(payload, "evidence_text", "") or "").strip()
        ocr_run_id = str(getattr(payload, "ocr_run_id", "") or "").strip()

        if not question or not draft_answer:
            raise PaleographyVerificationAgentError("question and draft_answer are required.")

        user_prompt = "\n".join(
            [
                f"Question: {question}",
                f"Run ID: {ocr_run_id or '(none)'}",
                "",
                "Draft answer to verify:",
                draft_answer,
                "",
                "Transcript snippet:",
                transcript[:4000] or "(none)",
                "",
                "Authority report snippet:",
                authority_report[:4000] or "(none)",
                "",
                "Retrieved evidence blocks:",
                evidence_text[:8000] or "(none)",
            ]
        )

        messages = [
            {"role": "system", "content": PALEOGRAPHY_VERIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        client = _create_client()
        selected_model = self._target_model()

        def _run_completion(target_model: str) -> Any:
            return client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=float(settings.paleography_verification_temperature),
                max_tokens=int(settings.paleography_verification_max_tokens),
                stream=False,
            )

        try:
            response = _run_completion(selected_model)
        except Exception as exc:
            if _is_model_not_found_error(exc):
                raise PaleographyVerificationAgentError(
                    f"Verification model unavailable: {selected_model}"
                ) from exc
            raise PaleographyVerificationAgentError(f"Verification failed: {exc}") from exc

        choices = list(getattr(response, "choices", []) or [])
        raw_text = ""
        if choices:
            message = getattr(choices[0], "message", None)
            raw_text = str(getattr(message, "content", "") or "")

        assessment, corrected_answer, notes, citations_checked = _normalized_verification_payload(
            raw_text,
            draft_answer=draft_answer,
        )

        duration_ms = (time.perf_counter() - started_at) * 1000
        stage_metadata = _stage_metadata(model_used=selected_model, duration_ms=duration_ms)

        return PaleographyVerificationResult(
            assessment=assessment,
            corrected_answer=corrected_answer,
            notes=notes,
            citations_checked=citations_checked,
            model_used=selected_model,
            duration_ms=duration_ms,
            stage_metadata=stage_metadata,
            inspection=_inspection_payload(
                question=question,
                ocr_run_id=ocr_run_id,
                transcript=transcript,
                authority_report=authority_report,
                evidence_text=evidence_text,
                assessment=assessment,
                corrected_answer=corrected_answer,
                notes=notes,
                citations_checked=citations_checked,
                stage_metadata=stage_metadata,
            ),
            raw_text=raw_text,
        )
