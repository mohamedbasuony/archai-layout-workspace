from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents.ocr_proofreader_agent import (  # type: ignore[import-untyped]
    OcrProofreaderAgent,
    apply_archai_safe_normalizer,
    apply_decorated_initial_fix,
    apply_latin_micro_corrections,
)


class _FakeSaiaClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def list_models(self, force_refresh: bool = False) -> list[str]:
        _ = force_refresh
        return ["internvl3.5-30b-a3b"]

    def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {"text": kwargs["messages"][1]["content"].split("OCR text:\n", 1)[-1]}


def test_decorated_initial_whitelist_merge_variants() -> None:
    variants = [
        "cuero di",
        "cvero di",
        "euerendi",
        "euerendu",
        "euerendi i",
        "euerendu i",
    ]

    for variant in variants:
        text = f"R\n{variant}\npatru clemencia"
        merged = apply_decorated_initial_fix(text, "latin_medieval")
        assert merged.splitlines()[0] == "Reuerendi"
        assert merged.splitlines()[1] == "patru clemencia"


def test_decorated_initial_generic_merge_when_euer_prefix() -> None:
    text = "R\neuerendissime\nlinea secunda"
    merged = apply_decorated_initial_fix(text, "latin_medieval")

    lines = merged.splitlines()
    assert lines[0] == "Reuerendissime"
    assert lines[1] == "linea secunda"


def test_decorated_initial_no_merge_when_not_whitelisted_or_prefix() -> None:
    text = "R\nquodlibet\nlinea secunda"
    merged = apply_decorated_initial_fix(text, "latin_medieval")
    assert merged == text


def test_latin_micro_corrections_safe_tokens_only() -> None:
    text = "patru clemencia\npaTru clemencia\npatru? [... ]"
    text = text.replace("[... ]", "[…]")
    corrected = apply_latin_micro_corrections(text, "latin_medieval")

    lines = corrected.splitlines()
    assert lines[0] == "patri clementia"
    assert lines[1] == "paTru clementia"
    assert lines[2] == "patri? […]"


def test_archai_safe_normalizer_combines_merge_and_micro_corrections() -> None:
    text = "R\ncuero di\npatru clemencia"
    normalized = apply_archai_safe_normalizer(text, "latin_medieval")
    assert normalized == "Reuerendi\npatri clementia"


def test_archai_safe_normalizer_non_latin_no_changes() -> None:
    text = "R\ncuero di\npatru clemencia"
    normalized = apply_archai_safe_normalizer(text, "greek")
    assert normalized == text


def test_archai_safe_normalizer_preserves_insular_symbols() -> None:
    text = "þe ƿord 7 tacn"
    normalized = apply_archai_safe_normalizer(text, "insular_old_english")
    assert normalized == text


def test_proofreader_chat_completion_is_stateless_single_turn() -> None:
    fake = _FakeSaiaClient()
    agent = OcrProofreaderAgent(client=fake, model_override="internvl3.5-30b-a3b")
    _ = agent.proofread("patru clemencia", "latin_medieval")

    assert fake.calls
    messages = fake.calls[0]["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_proofreader_preserves_line_count_without_decorated_initial_merge() -> None:
    fake = _FakeSaiaClient()
    agent = OcrProofreaderAgent(client=fake, model_override="internvl3.5-30b-a3b")
    source = "prima linea\nsecunda linea\ntertia linea"
    result = agent.proofread(source, "latin")

    assert len(result.splitlines()) == len(source.splitlines())


def test_proofreader_allows_single_line_merge_for_decorated_initial_case() -> None:
    fake = _FakeSaiaClient()
    agent = OcrProofreaderAgent(client=fake, model_override="internvl3.5-30b-a3b")
    source = "R\neuerendi\nsecunda linea"
    result = agent.proofread(source, "latin")

    assert result.splitlines()[0] == "Reuerendi"
    assert len(result.splitlines()) == len(source.splitlines()) - 1
