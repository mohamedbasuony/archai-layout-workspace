"""Debug-only RAG retrieval endpoint.

Active only when ``ARCHAI_DEBUG_RAG`` is set to a truthy value.
Returns 404 otherwise so it stays invisible in production.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.services.rag_debug import is_rag_debug_enabled
from app.services import rag_store

router = APIRouter(tags=["rag-debug"])


@router.get("/rag/debug/retrieve")
async def rag_debug_retrieve(
    query: str = Query(..., description="Free-text search query"),
    k: int = Query(5, ge=1, le=50, description="Top-k results"),
    run_id: str | None = Query(None, description="Filter by run_id"),
    asset_ref: str | None = Query(None, description="Filter by asset_ref"),
) -> dict[str, Any]:
    """Return structured top-k retrieval results for a query.

    **Only available when** ``ARCHAI_DEBUG_RAG=1`` (or ``true``/``yes``).
    """
    if not is_rag_debug_enabled():
        raise HTTPException(
            status_code=404,
            detail="RAG debug endpoint disabled.  Set ARCHAI_DEBUG_RAG=1 to enable.",
        )
    try:
        return rag_store.retrieve_debug(
            query, k=k, run_id=run_id, asset_ref=asset_ref,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Retrieve failed: {exc}") from exc


@router.get("/rag/debug/evidence-preview")
async def rag_debug_evidence_preview(
    query: str = Query(..., description="Free-text search query"),
    run_id: str = Query(..., description="Run ID to scope retrieval"),
    k: int = Query(8, ge=1, le=50, description="Top-k results"),
) -> dict[str, Any]:
    """Preview the exact [EVIDENCE] blocks and citation format injected into the LLM.

    **Only available when** ``ARCHAI_DEBUG_RAG=1`` (or ``true``/``yes``).
    """
    if not is_rag_debug_enabled():
        raise HTTPException(
            status_code=404,
            detail="RAG debug endpoint disabled.  Set ARCHAI_DEBUG_RAG=1 to enable.",
        )
    try:
        from app.services.chat_ai import build_rag_evidence_for_debug
        return build_rag_evidence_for_debug(query, run_id, k=k)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Evidence preview failed: {exc}") from exc
