from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.config import settings


@dataclass(frozen=True)
class TaskModelAssignments:
    ocr_model: str
    chat_rag_model: str
    translation_model: str
    label_visual_model: str
    label_visual_fallback_model: str
    verifier_model: str
    embedding_model: str

    def as_payload(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def get_task_model_assignments() -> TaskModelAssignments:
    chat_rag_model = str(
        settings.chat_rag_model
        or settings.archai_chat_ai_model
        or "qwen3-30b-a3b-instruct-2507"
    ).strip() or "qwen3-30b-a3b-instruct-2507"
    translation_model = str(
        settings.translation_model
        or chat_rag_model
        or "llama-3.3-70b-instruct"
    ).strip() or chat_rag_model
    label_visual_model = str(
        settings.label_visual_model
        or settings.saia_label_analysis_model
        or "qwen3-vl-30b-a3b-instruct"
    ).strip() or "qwen3-vl-30b-a3b-instruct"
    label_visual_fallback_model = str(
        settings.label_visual_fallback_model
        or "internvl3.5-30b-a3b"
    ).strip() or "internvl3.5-30b-a3b"
    verifier_model = str(
        settings.paleography_verification_model
        or chat_rag_model
        or "qwen3-235b-a22b"
    ).strip() or chat_rag_model
    embedding_model = str(
        settings.rag_embedding_model
        or "multilingual-e5-large-instruct"
    ).strip() or "multilingual-e5-large-instruct"
    ocr_model = str(
        settings.glmocr_ollama_model
        or "glm-ocr:latest"
    ).strip() or "glm-ocr:latest"

    return TaskModelAssignments(
        ocr_model=ocr_model,
        chat_rag_model=chat_rag_model,
        translation_model=translation_model,
        label_visual_model=label_visual_model,
        label_visual_fallback_model=label_visual_fallback_model,
        verifier_model=verifier_model,
        embedding_model=embedding_model,
    )


def chat_stage_from_context(context: dict[str, Any] | None) -> str:
    if not context:
        return "rag_chat"
    stage = str(
        context.get("chat_stage")
        or context.get("task")
        or context.get("mode")
        or "rag_chat"
    ).strip().lower()
    return stage or "rag_chat"


def model_for_chat_stage(stage: str | None, *, prefer_vision: bool = False) -> str:
    assignments = get_task_model_assignments()
    normalized = str(stage or "rag_chat").strip().lower() or "rag_chat"
    if normalized == "translation":
        return assignments.translation_model
    if normalized in {"label_visual", "visual_label_analysis", "visual_chat"}:
        return assignments.label_visual_model
    if normalized in {"verification", "verifier", "paleography_verification"}:
        return assignments.verifier_model
    if prefer_vision:
        return assignments.label_visual_model
    return assignments.chat_rag_model


def fallback_models_for_stage(stage: str | None, *, prefer_vision: bool = False) -> list[str]:
    assignments = get_task_model_assignments()
    normalized = str(stage or "rag_chat").strip().lower() or "rag_chat"

    ordered: list[str] = []
    if normalized == "translation":
        ordered.extend([assignments.translation_model, assignments.chat_rag_model])
    elif normalized in {"label_visual", "visual_label_analysis", "visual_chat"}:
        ordered.extend([assignments.label_visual_model, assignments.label_visual_fallback_model, assignments.chat_rag_model])
    elif normalized in {"verification", "verifier", "paleography_verification"}:
        ordered.extend([assignments.verifier_model, assignments.chat_rag_model])
    else:
        ordered.append(assignments.chat_rag_model)
        if prefer_vision:
            ordered.extend([assignments.label_visual_model, assignments.label_visual_fallback_model])

    deduped: list[str] = []
    seen: set[str] = set()
    for model_id in ordered:
        clean = str(model_id or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped
