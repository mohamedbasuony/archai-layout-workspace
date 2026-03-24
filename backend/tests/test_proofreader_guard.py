"""Regression tests for the proofreader guard (check_proofread_delta)."""
from __future__ import annotations

import sys
from pathlib import Path

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.agents.ocr_proofreader_agent import check_proofread_delta  # type: ignore[import-untyped]


def test_identical_text_accepted() -> None:
    raw = "Reuerendi patris et domini"
    verdict = check_proofread_delta(raw, raw)
    assert verdict.accepted
    assert verdict.char_edit_ratio == 0.0


def test_minor_correction_accepted() -> None:
    raw = "Reuerendi patru et domini"
    proofread = "Reuerendi patri et domini"
    verdict = check_proofread_delta(raw, proofread)
    assert verdict.accepted
    assert verdict.char_edit_ratio < 0.10


def test_massive_rewrite_rejected() -> None:
    raw = "qjxvvbbx cccnnn zzppp"
    proofread = "In the name of the Lord and His holy kingdom we proclaim"
    verdict = check_proofread_delta(raw, proofread)
    assert not verdict.accepted
    assert "char_edit_ratio" in verdict.reason or "token_churn" in verdict.reason


def test_line_count_drift_rejected() -> None:
    # Content is similar (low char edit) but line count changes drastically
    raw = "ab cd\nef gh\nij kl\nmn op\nqr st\nuv wx\nyz ab\ncd ef\ngh ij\nkl mn"
    proofread = "ab cd ef gh ij kl mn op qr st uv wx yz ab cd ef gh ij kl mn"
    verdict = check_proofread_delta(raw, proofread)
    assert not verdict.accepted
    assert "line_drift" in verdict.reason


def test_uncertainty_stripping_rejected() -> None:
    raw = "Reuerendi [?] et [?] domini [?] nostri [?]"
    proofread = "Reuerendi patris et sancti domini nostri regis"
    verdict = check_proofread_delta(raw, proofread)
    assert not verdict.accepted


def test_moderate_correction_accepted() -> None:
    raw = "furent les noces et le festin\nsi grant joie ne fu onques veue"
    proofread = "furent les noces et le festin\nsi grant joie ne fu onques veue"
    verdict = check_proofread_delta(raw, proofread)
    assert verdict.accepted


def test_ocr_confusion_fix_accepted() -> None:
    # Typical minim confusion fix: rn → m
    raw = "dorninus noster"
    proofread = "dominus noster"
    verdict = check_proofread_delta(raw, proofread)
    assert verdict.accepted
    assert verdict.char_edit_ratio < 0.15
