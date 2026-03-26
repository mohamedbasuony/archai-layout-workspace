"""Chat endpoints for ArchAI workspace."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.label_analysis_agent import LabelAnalysisAgent, LabelAnalysisAgentError
from app.services.chat_ai import (
    ChatConfigError,
    create_chat_completion,
    list_available_models,
    stream_chat_completion,
)
from app.services.saia_client import SaiaConfigError

router = APIRouter(tags=["chat"])
_label_analysis_agent_instance: LabelAnalysisAgent | None = None


def _get_label_analysis_agent() -> LabelAnalysisAgent:
    global _label_analysis_agent_instance
    if _label_analysis_agent_instance is None:
        _label_analysis_agent_instance = LabelAnalysisAgent()
    return _label_analysis_agent_instance


class ChatMessage(BaseModel):
    role: str = Field(default="user")
    content: Any


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.2
    stream: bool = True
    context: dict[str, Any] | None = None


class LabelRegionPayload(BaseModel):
    region_id: str | None = None
    bbox_xyxy: list[float] | None = None
    polygons: list[list[float]] = Field(default_factory=list)


class LabelAnalysisRequest(BaseModel):
    question: str
    label_name: str
    image_b64: str
    regions: list[LabelRegionPayload] = Field(default_factory=list)
    filename: str | None = None
    page_id: str | None = None
    document_id: str | None = None


class LabelAnalysisResponse(BaseModel):
    status: str
    text: str
    label_name: str
    analysis_mode: str | None = None
    model_used: str
    warnings: list[str] = Field(default_factory=list)
    region_count: int
    crop_image_b64: str
    crop_bounds_xyxy: list[int] = Field(default_factory=list)
    ocr_text: str | None = None
    stage_metadata: dict[str, Any] | None = None
    inspection: dict[str, Any] | None = None


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


@router.post("/chat/label-analysis", response_model=LabelAnalysisResponse)
async def chat_label_analysis(payload: LabelAnalysisRequest) -> LabelAnalysisResponse:
    try:
        return await asyncio.to_thread(_get_label_analysis_agent().run, payload)
    except LabelAnalysisAgentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ChatConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SaiaConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Label analysis failed: {exc}") from exc
