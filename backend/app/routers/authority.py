"""Authority-linking router — trigger and inspect entity linking for pipeline runs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["authority-linking"])


@router.post("/authority/link/{run_id}")
async def link_run(run_id: str, force_refresh: bool = False) -> dict[str, Any]:
    """Run authority linking (Wikidata search + scoring) for all mentions in *run_id*."""
    from app.services.authority_linking import run_authority_linking

    try:
        return run_authority_linking(run_id, force_refresh=force_refresh)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Authority linking failed: {exc}") from exc


@router.get("/authority/report/{run_id}")
async def linking_report(run_id: str) -> dict[str, Any]:
    """Return a pre-built linking report for *run_id*.

    Reads existing candidates from the DB (does NOT re-run linking).
    """
    from app.services.authority_linking import build_report_from_db

    try:
        return build_report_from_db(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Report failed: {exc}") from exc
