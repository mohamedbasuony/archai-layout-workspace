from __future__ import annotations

import sys
from pathlib import Path

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.routers import ocr as ocr_router  # type: ignore[import-untyped]


def test_mentions_include_place_and_title_for_middle_french_sample() -> None:
    sample = (
        "Tresreuerend pere en dieu euesque et prince de lausanne.\n"
        "Approchant l'an de grace mil cinq cens."
    )
    mentions, _candidates = ocr_router._extract_mentions_from_text(sample)
    assert any(str(item.get("ent_type")) == "place" and "lausanne" in str(item.get("surface", "")).lower() for item in mentions)
    assert any(str(item.get("ent_type")) == "title" and str(item.get("surface", "")).lower() in {"pere", "euesque", "prince"} for item in mentions)


def test_mention_offsets_map_to_exact_surface_substring() -> None:
    sample = "Tresreuerend pere en dieu euesque et prince de lausanne."
    mentions, _candidates = ocr_router._extract_mentions_from_text(sample)
    assert mentions
    for mention in mentions:
        start = int(mention["start_offset"])
        end = int(mention["end_offset"])
        assert sample[start:end] == str(mention["surface"])


def test_no_mentions_for_empty_text() -> None:
    mentions, candidates = ocr_router._extract_mentions_from_text("")
    chunks = ocr_router._build_line_chunks("")
    assert mentions == []
    assert candidates == []
    assert chunks == []
