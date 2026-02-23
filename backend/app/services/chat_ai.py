"""Chat AI proxy utilities for GWDG OpenAI-compatible API."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import os
from typing import Any

from app.config import settings

ARCHAI_SYSTEM_PROMPT = (
    "You are ArchAI, a manuscript research and extraction assistant. "
    "Be precise, cite uncertainty explicitly, and provide structured JSON when the user asks for structured outputs."
)
LEGACY_SAIA_MODEL_FALLBACKS = [
    "qwen3-vl-30b-a3b-instruct",
    "internvl3.5-30b-a3b",
    "mistral-large-3-675b-instruct-2512",
    "gemma-3-27b-it",
    "medgemma-27b-it",
]


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


def _default_model() -> str:
    configured = str(settings.archai_chat_ai_model or "").strip()
    if configured:
        return configured
    return LEGACY_SAIA_MODEL_FALLBACKS[0]


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


def _choose_retry_model(
    requested_model: str,
    available_models: Sequence[str],
    prefer_vision: bool,
) -> str | None:
    if not available_models:
        return None

    normalized = {model.lower(): model for model in available_models}
    for candidate in LEGACY_SAIA_MODEL_FALLBACKS:
        match = normalized.get(candidate.lower())
        if match and match != requested_model:
            return match

    configured_default = _default_model()
    default_match = normalized.get(configured_default.lower())
    if default_match and default_match != requested_model:
        return default_match

    if prefer_vision:
        for model in available_models:
            if is_vision_model(model) and model != requested_model:
                return model

    for model in available_models:
        if model != requested_model:
            return model
    return None


def list_available_models() -> dict[str, Any]:
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

    configured_default = _default_model()
    if configured_default and configured_default not in seen:
        unique_models = [configured_default, *unique_models]
        seen.add(configured_default)
    if not listed_ok:
        for fallback in LEGACY_SAIA_MODEL_FALLBACKS:
            if fallback not in seen:
                unique_models.append(fallback)
                seen.add(fallback)

    return {
        "models": unique_models,
        "default_model": configured_default,
        "vision_models": [model for model in unique_models if is_vision_model(model)],
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

    if doc_id is not None:
        lines.append(f"- document_id: {doc_id}")
    if filename is not None:
        lines.append(f"- filename: {filename}")
    if page_index is not None:
        lines.append(f"- current_page_index: {page_index}")
    if page_total is not None:
        lines.append(f"- page_count: {page_total}")
    if transcript:
        lines.append("- transcript_snippet:")
        lines.append(str(transcript)[:1500])

    return "\n".join(lines)


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


def _prepare_messages(messages: Sequence[dict[str, Any]], context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not messages:
        raise ValueError("At least one message is required.")

    prepared = [_normalize_message(msg) for msg in messages]
    system_prompt = _context_to_system_message(context)
    return [{"role": "system", "content": system_prompt}, *prepared]


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


def create_chat_completion(
    messages: Sequence[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.2,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = _create_client()
    selected_model = (model or _default_model()).strip()
    prepared_messages = _prepare_messages(messages, context)
    prefer_vision = _messages_include_image(prepared_messages)

    def _run_completion(target_model: str) -> Any:
        return client.chat.completions.create(
            model=target_model,
            messages=prepared_messages,
            temperature=float(temperature),
            stream=False,
        )

    try:
        response = _run_completion(selected_model)
    except Exception as exc:
        if not _is_model_not_found_error(exc):
            raise
        available_models: list[str] = []
        try:
            available_models = _raw_server_models(client)
        except Exception:
            available_models = []
        retry_model = _choose_retry_model(selected_model, available_models, prefer_vision)
        if not retry_model:
            raise
        selected_model = retry_model
        response = _run_completion(selected_model)

    choices = list(getattr(response, "choices", []) or [])
    text = _extract_choice_text(choices[0]) if choices else ""
    return {
        "text": text,
        "model": selected_model,
        "context_used": bool(context),
    }


def stream_chat_completion(
    messages: Sequence[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.2,
    context: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    client = _create_client()
    selected_model = (model or _default_model()).strip()
    prepared_messages = _prepare_messages(messages, context)
    prefer_vision = _messages_include_image(prepared_messages)

    def _run_stream(target_model: str) -> Any:
        return client.chat.completions.create(
            model=target_model,
            messages=prepared_messages,
            temperature=float(temperature),
            stream=True,
        )

    try:
        stream = _run_stream(selected_model)
    except Exception as exc:
        if not _is_model_not_found_error(exc):
            raise
        available_models: list[str] = []
        try:
            available_models = _raw_server_models(client)
        except Exception:
            available_models = []
        retry_model = _choose_retry_model(selected_model, available_models, prefer_vision)
        if not retry_model:
            raise
        selected_model = retry_model
        stream = _run_stream(selected_model)

    chunks: list[str] = []
    for event in stream:
        choices = list(getattr(event, "choices", []) or [])
        if not choices:
            continue
        delta = _extract_choice_text(choices[0])
        if not delta:
            continue
        chunks.append(delta)
        yield {"type": "delta", "delta": delta}

    yield {
        "type": "done",
        "text": "".join(chunks),
        "model": selected_model,
        "context_used": bool(context),
    }
