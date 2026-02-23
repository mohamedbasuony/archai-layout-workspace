from __future__ import annotations

import sys
from pathlib import Path

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents import ocr_agent  # type: ignore[import-untyped]


def test_allowed_script_hint_values_are_new_set() -> None:
    assert ocr_agent.ALLOWED_SCRIPT_HINTS == {"insular_old_english", "latin_medieval", "unknown"}


def test_strict_json_validator_rejects_extra_keys_and_fixes_text_join() -> None:
    raw = (
        '{'
        '"lines":["linea prima","linea secunda"],'
        '"text":"linea prima",'
        '"script_hint":"latin_medieval",'
        '"confidence":0.7,'
        '"warnings":[],'
        '"extra":"not-allowed"'
        '}'
    )
    parsed = ocr_agent.parse_ocr_json(raw, latin_lock=True, strict_keys=True)

    assert parsed["text"] == "linea prima\nlinea secunda"
    assert "INVALID_SCHEMA_KEYS" in parsed["warnings"]
    assert "TEXT_JOIN_FIXED" in parsed["warnings"]


def test_script_drift_detector_flags_greek_cyrillic_and_cjk() -> None:
    sample = "latin Α Β Г Д 漢字"
    assert ocr_agent._contains_script_drift(sample)


def test_validator_keeps_insular_characters_when_script_hint_is_insular() -> None:
    raw = (
        '{'
        '"lines":["þe ðæt ƿe 7"],'
        '"text":"þe ðæt ƿe 7",'
        '"script_hint":"insular_old_english",'
        '"confidence":0.8,'
        '"warnings":[]'
        '}'
    )
    parsed = ocr_agent.parse_ocr_json(raw, latin_lock=True, strict_keys=True)

    assert parsed["script_hint"] == "insular_old_english"
    assert parsed["text"] == "þe ðæt ƿe 7"
    assert "SCRIPT_DRIFT" not in parsed["warnings"]
