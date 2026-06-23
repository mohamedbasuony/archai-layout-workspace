"""FastAPI application factory and HTTP configuration."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import register_api_routes
from app.config import settings
from app.core.lifecycle import model_pool_lifespan


def create_app() -> FastAPI:
    """Build the configured API application without starting model processes."""
    app = FastAPI(
        title="Manuscript Layout Analysis API",
        version="1.0.0",
        lifespan=model_pool_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )
    register_api_routes(app)

    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app
