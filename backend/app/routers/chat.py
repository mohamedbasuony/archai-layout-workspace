"""Chat endpoints for ArchAI workspace."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.chat_ai import (
    ChatConfigError,
    create_chat_completion,
    list_available_models,
    stream_chat_completion,
)

router = APIRouter(tags=["chat"])


class ChatMessage(BaseModel):
    role: str = Field(default="user")
    content: Any


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.2
    stream: bool = True
    context: dict[str, Any] | None = None


@router.get("/chat/models")
async def chat_models() -> dict[str, Any]:
    try:
        return list_available_models()
    except ChatConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Failed to list chat models: {exc}") from exc


def _sse_events(payload: ChatCompletionRequest) -> Iterator[str]:
    try:
        for event in stream_chat_completion(
            messages=[msg.model_dump() for msg in payload.messages],
            model=payload.model,
            temperature=payload.temperature,
            context=payload.context,
        ):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except Exception as exc:
        error_event = {"type": "error", "error": str(exc)}
        yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest) -> Any:
    try:
        if payload.stream:
            return StreamingResponse(
                _sse_events(payload),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return create_chat_completion(
            messages=[msg.model_dump() for msg in payload.messages],
            model=payload.model,
            temperature=payload.temperature,
            context=payload.context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ChatConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {exc}") from exc
