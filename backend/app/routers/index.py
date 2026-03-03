"""Index router — trigger and inspect RAG indexing for pipeline runs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.services import rag_store

router = APIRouter(tags=["index"])


@router.post("/index/run/{run_id}")
async def index_run(run_id: str) -> dict[str, Any]:
    """Embed all chunks for *run_id* and upsert into the vector store."""
    try:
        return rag_store.index_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc


@router.get("/index/run/{run_id}/status")
async def index_status(run_id: str) -> dict[str, Any]:
    """Check whether *run_id* has been indexed."""
    try:
        return rag_store.is_run_indexed(run_id)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Status check failed: {exc}") from exc


@router.delete("/index/run/{run_id}")
async def delete_index(run_id: str) -> dict[str, Any]:
    """Remove all vectors for *run_id* from the store."""
    try:
        deleted = rag_store.delete_run(run_id)
        return {"run_id": run_id, "deleted": deleted}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Deletion failed: {exc}") from exc


@router.post("/index/search")
async def search_index(
    query: str,
    top_k: int | None = None,
    run_id: str | None = None,
    asset_ref: str | None = None,
) -> dict[str, Any]:
    """Retrieve the most relevant chunks for a free-text *query*.

    Optionally filter by *run_id* and/or *asset_ref*.
    """
    try:
        run_ids = [run_id] if run_id else None
        hits = rag_store.retrieve_chunks(
            query, top_k=top_k, run_ids=run_ids, asset_ref=asset_ref,
        )
        return {"query": query, "top_k": top_k or 5, "results": hits}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc
