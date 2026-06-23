from __future__ import annotations

import sys
from pathlib import Path

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.core.application import create_app  # type: ignore[import-untyped]


def test_application_factory_registers_expected_routes() -> None:
    app = create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    for route in app.routes:
        original_router = getattr(route, "original_router", None)
        include_context = getattr(route, "include_context", None)
        if original_router is None or include_context is None:
            continue
        paths.update(f"{include_context.prefix}{child.path}" for child in original_router.routes)

    assert "/api/health" in paths
    assert "/api/chat/models" in paths
    assert "/api/ocr/extract_full_page" in paths
    assert "/static" in paths
