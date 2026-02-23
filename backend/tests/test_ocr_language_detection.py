from __future__ import annotations

import sys
from pathlib import Path

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents import saia_ocr_agent  # type: ignore[import-untyped]
from app.routers import ocr as ocr_router  # type: ignore[import-untyped]


def test_noisy_medieval_french_is_not_latin_or_unknown() -> None:
    sample = (
        "La satyre megerc.\n"
        "Tresfauuand par en dieu Amc\n"
        "& monfalcon eusque & pena\n"
        "& laufanne. chapellain et desirs de la grace approchant"
    )

    language, confidence = ocr_router._detect_language_metadata(sample)

    assert language in {"middle_french", "old_french"}
    assert language not in {"latin", "unknown"}
    assert confidence is None or (0.0 <= confidence <= 1.0)


def test_clearly_latin_text_stays_latin() -> None:
    sample = (
        "In nomine domini amen. Hoc instrumentum publicum et autenticum omnibus pateat evidenter, "
        "quod nos canonici et capitulum consensu concordi statuimus et confirmamus perpetuo."
    )

    language, _confidence = ocr_router._detect_language_metadata(sample)

    assert language == "latin"


def test_arabic_and_hebrew_detection_still_works() -> None:
    arabic_text = "هذا نص عربي قديم للاختبار في المخطوط"
    hebrew_text = "שלום עולם טקסט עברי"

    lang_ar, conf_ar = ocr_router._detect_language_metadata(arabic_text)
    assert lang_ar == "arabic"
    assert conf_ar is not None and conf_ar > 0.9

    assert ocr_router._fallback_detected_language("latin", hebrew_text) == "hebrew"
    assert saia_ocr_agent._fallback_detected_language("latin", hebrew_text) == "hebrew"


def test_empty_text_returns_unknown() -> None:
    language, confidence = ocr_router._detect_language_metadata("")
    assert language == "unknown"
    assert confidence is None

    assert ocr_router._fallback_detected_language("latin", "") == "unknown"
    assert saia_ocr_agent._fallback_detected_language("latin", "") == "unknown"


def test_latin_script_fallback_prefers_middle_french_when_anchor_present() -> None:
    sample = "Tresreuerend pere et desirs de la grace"

    assert ocr_router._fallback_detected_language("latin", sample) == "middle_french"
    assert saia_ocr_agent._fallback_detected_language("latin", sample) == "middle_french"
