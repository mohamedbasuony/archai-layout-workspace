import multiprocessing as mp
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.services.inference import init_model_pool, shutdown_model_pool
from app.routers import health, classes, predict, download, analytics, chat, agents_ocr, ocr, evidence, index, rag_debug, authority


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model pool on startup, clean up on shutdown."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    init_model_pool()
    yield
    shutdown_model_pool()


app = FastAPI(
    title="Manuscript Layout Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

app.include_router(health.router, prefix="/api")
app.include_router(classes.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(agents_ocr.router, prefix="/api")
app.include_router(ocr.router, prefix="/api")
app.include_router(evidence.router, prefix="/api")
app.include_router(index.router, prefix="/api")
app.include_router(rag_debug.router, prefix="/api")
app.include_router(authority.router, prefix="/api")
app.include_router(download.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
