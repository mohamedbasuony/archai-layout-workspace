from __future__ import annotations

import os
import time
from typing import Any, Sequence

from app.config import settings


class SaiaConfigError(RuntimeError):
    """Raised when SAIA credentials/config are missing."""


class SaiaClientError(RuntimeError):
    """Raised when SAIA request/response handling fails."""


class SaiaClient:
    _shared_models_cache: tuple[float, list[str]] | None = None

    def __init__(self) -> None:
        self._api_key = self._resolve_api_key()
        self._base_url = self._resolve_base_url()
        self._timeout_seconds = float(
            settings.saia_timeout_seconds
            or os.getenv("SAIA_TIMEOUT_SECONDS", "120")
            or 120
        )
        self._cache_ttl_seconds = int(
            settings.saia_models_cache_ttl_seconds
            or os.getenv("SAIA_MODELS_CACHE_TTL_SECONDS", "300")
            or 300
        )
        self._client = self._create_client()

    @staticmethod
    def _resolve_api_key() -> str:
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
            raise SaiaConfigError(
                "Chat API key not configured. Set CHAT_AI_API_KEY (or SAIA_API_KEY) in backend/.env or environment."
            )
        return key

    @staticmethod
    def _resolve_base_url() -> str:
        base_url = str(
            settings.chat_ai_base_url
            or os.getenv("CHAT_AI_BASE_URL", "")
            or settings.saia_base_url
            or settings.archai_chat_ai_base_url
            or os.getenv("SAIA_BASE_URL", "")
            or os.getenv("ARCHAI_CHAT_AI_BASE_URL", "")
            or "https://chat-ai.academiccloud.de/v1"
        ).strip()
        if not base_url:
            raise SaiaConfigError("SAIA_BASE_URL is empty.")
        return base_url.rstrip("/")

    def _create_client(self) -> Any:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise SaiaClientError("openai package is required for SAIA client support.") from exc
        return OpenAI(api_key=self._api_key, base_url=self._base_url, timeout=self._timeout_seconds)

    @property
    def base_url(self) -> str:
        return self._base_url

    def list_models(self, force_refresh: bool = False) -> list[str]:
        now = time.monotonic()
        if not force_refresh and SaiaClient._shared_models_cache is not None:
            expires_at, cached_models = SaiaClient._shared_models_cache
            if now < expires_at:
                return list(cached_models)

        response = self._client.models.list()
        models: list[str] = []
        seen: set[str] = set()
        for item in getattr(response, "data", []) or []:
            model_id = str(getattr(item, "id", "") or "").strip()
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            models.append(model_id)

        SaiaClient._shared_models_cache = (now + max(1, self._cache_ttl_seconds), models)
        return list(models)

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        temperature: float = 0.0,
        top_p: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        if deterministic:
            temperature = 0.0
            top_p = 1.0 if top_p is None else top_p

        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": float(temperature),
            "stream": False,
        }
        if deterministic:
            payload["n"] = 1
            payload["frequency_penalty"] = 0.0
            payload["presence_penalty"] = 0.0
            payload["extra_body"] = {"do_sample": False}
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            response = self._client.chat.completions.create(**payload)
        except Exception as exc:
            lowered = str(exc).lower()
            if deterministic and (
                "do_sample" in lowered
                or "extra_body" in lowered
                or "unknown field" in lowered
                or "unexpected keyword" in lowered
            ):
                payload.pop("extra_body", None)
                response = self._client.chat.completions.create(**payload)
            else:
                raise
        choices = list(getattr(response, "choices", []) or [])
        if not choices:
            return {"text": "", "raw": response}

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        text: str
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    maybe_text = item.get("text")
                    if isinstance(maybe_text, str):
                        parts.append(maybe_text)
                else:
                    maybe_text = getattr(item, "text", None)
                    if isinstance(maybe_text, str):
                        parts.append(maybe_text)
            text = "".join(parts)
        else:
            text = str(content or "")

        return {"text": text, "raw": response}


def is_model_not_found_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "model not found" in text
        or "unknown model" in text
        or "does not exist" in text
        or "invalid model" in text
        or "404" in text
        or "invalid_request_error" in text
    )
