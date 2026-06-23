"""Application lifecycle hooks for model-backed services."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.services.inference import init_model_pool, shutdown_model_pool


@asynccontextmanager
async def model_pool_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start and stop the shared model pool with the application."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    init_model_pool()
    try:
        yield
    finally:
        shutdown_model_pool()
