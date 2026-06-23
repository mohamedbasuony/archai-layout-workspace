"""API router registration for the backend."""

from fastapi import FastAPI

from app.routers import (
    agents_ocr,
    analytics,
    authority,
    chat,
    classes,
    download,
    evidence,
    health,
    index,
    ocr,
    predict,
    rag_debug,
)


def register_api_routes(app: FastAPI) -> None:
    """Attach every public API router under the common API prefix."""
    for router in (
        health.router,
        classes.router,
        predict.router,
        chat.router,
        agents_ocr.router,
        ocr.router,
        evidence.router,
        index.router,
        rag_debug.router,
        authority.router,
        download.router,
        analytics.router,
    ):
        app.include_router(router, prefix="/api")
