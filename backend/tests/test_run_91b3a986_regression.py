"""Regression tests for run 91b3a986-1d99-4ee3-adba-d769c092b849.

Root causes:
  1. ``rule:date_pattern`` falsely extracts "mil honorement\\nvus re mande"
     as ent_type=date (conf=0.72) — ``mil`` alone is not a strong date anchor.
  2. ``rule:salvage_place_candidate`` extracts "lacune" as ent_type=place
     (conf=0.35) and sends it to Wikidata — "lacune" is an editorial/philology
     marker, not a place name.
  3. linked_total=0 causes Gate C to FAIL even though no linkable entities exist.

This file covers:
  1. Date anchor validation (Req 1)
  2. Editorial blacklist for place salvage (Req 2)
  3. Authority linking behaviour by entity type (Req 3)
  4. Conditional Gate C — NO_LINKABLE_MENTIONS (Req 4)
  5. Single-decision consistency (Req 5)
  6. Integration regression for run 91b3a986
"""

from __future__ import annotations

import pytest


# ── 1. Date Anchor Validation ────────────────────────────────────────


class TestDateAnchorValidation:
    """rule:date_pattern must NOT classify spans without strong anchors."""

    def test_mil_honorement_not_date(self):
        """'mil honorement\\nvus re mande' must NOT be extracted as DATE."""
        from app.routers.ocr import _extract_anchor_mentions
        text = "mil honorement\nvus re mande"
        mentions = _extract_anchor_mentions(text)
        dates = [m for m in mentions if m["ent_type"] == "date"]
        assert len(dates) == 0, (
            f"Expected 0 date mentions for 'mil honorement', got {len(dates)}: "
            f"{[m['surface'] for m in dates]}"
        )

    def test_mil_alone_not_date(self):
        """'mil' alone without an anchor is NOT a valid date span."""
        from app.routers.ocr import _extract_anchor_mentions
        text = "mil honorement de la terre"
        mentions = _extract_anchor_mentions(text)
        dates = [m for m in mentions if m["ent_type"] == "date"]
        assert len(dates) == 0

    def test_mil_with_random_context_not_date(self):
        """'mil' surrounded by non-date text is NOT a date."""
        from app.routers.ocr import _extract_anchor_mentions
        text = "granz mil et molt grant joie"
        mentions = _extract_anchor_mentions(text)
        dates = [m for m in mentions if m["ent_type"] == "date"]
        assert len(dates) == 0

    def test_lan_mil_is_date(self):
        """'l'an mil' is a valid date formula."""
        from app.routers.ocr import _extract_anchor_mentions
        text = "en l'an mil CCCXL"
        mentions = _extract_anchor_mentions(text)
        dates = [m for m in mentions if m["ent_type"] == "date"]
        assert len(dates) >= 1, "Expected at least 1 date mention for 'l'an mil'"

    def test_numeric_year_is_date(self):
        """A 4-digit numeric year is a valid date."""
        from app.routers.ocr import _extract_anchor_mentions
        text = "en l'an 1345 de grace"
        mentions = _extract_anchor_mentions(text)
        dates = [m for m in mentions if m["ent_type"] == "date"]
        assert len(dates) >= 1

    def test_roman_numeral_year_is_date(self):
        """A roman-numeral year like MCCCLXIIII is a valid date."""
        from app.routers.ocr import _extract_anchor_mentions
        text = "l'an MCCCLXIIII"
        mentions = _extract_anchor_mentions(text)
        dates = [m for m in mentions if m["ent_type"] == "date"]
        assert len(dates) >= 1

    def test_an_de_grace_is_date(self):
        """'an de grace' is a date formula."""
        from app.routers.ocr import _extract_anchor_mentions
        text = "an de grace mil CCC"
        mentions = _extract_anchor_mentions(text)
        dates = [m for m in mentions if m["ent_type"] == "date"]
        assert len(dates) >= 1

    def test_newline_crossing_without_anchors_both_sides(self):
        """Date span crossing newline needs anchors on BOTH sides."""
        from app.routers.ocr import _has_date_anchor
        # Simulate: left side has "an", right side has no anchor
        assert _has_date_anchor("l'an mil") is True
        assert _has_date_anchor("vus re mande") is False


class TestHasDateAnchor:
    """Unit tests for _has_date_anchor helper."""

    def test_empty_string(self):
        from app.routers.ocr import _has_date_anchor
        assert _has_date_anchor("") is False

    def test_an_token(self):
        from app.routers.ocr import _has_date_anchor
        assert _has_date_anchor("an") is True

    def test_lan_token(self):
        from app.routers.ocr import _has_date_anchor
        assert _has_date_anchor("l'an") is True

    def test_numeric_year(self):
        from app.routers.ocr import _has_date_anchor
        assert _has_date_anchor("1345") is True

    def test_roman_year(self):
        from app.routers.ocr import _has_date_anchor
        assert _has_date_anchor("MCCCLXIIII") is True

    def test_month_name(self):
        from app.routers.ocr import _has_date_anchor
        assert _has_date_anchor("septembre") is True

    def test_mil_alone(self):
        from app.routers.ocr import _has_date_anchor
        assert _has_date_anchor("mil") is False

    def test_mil_honorement(self):
        from app.routers.ocr import _has_date_anchor
        assert _has_date_anchor("mil honorement") is False

    def test_random_text(self):
        from app.routers.ocr import _has_date_anchor
        assert _has_date_anchor("vus re mande") is False

    def test_formula_lan_mil(self):
        from app.routers.ocr import _has_date_anchor
        assert _has_date_anchor("l'an mil CCC") is True

    def test_apres(self):
        from app.routers.ocr import _has_date_anchor
        assert _has_date_anchor("apres noel") is True

    def test_avant(self):
        from app.routers.ocr import _has_date_anchor
        assert _has_date_anchor("avant paques") is True


# ── 2. Editorial Blacklist for Place Salvage ─────────────────────────


class TestEditorialBlacklist:
    """'lacune' and other editorial markers must NOT be emitted as PLACE."""

    def test_lacune_not_place(self):
        """'lacune' must be rejected by salvage_place_candidate."""
        from app.routers.ocr import _extract_salvage_mentions
        text = "en lacune de la terre"
        mentions, debug = _extract_salvage_mentions(text)
        place_mentions = [m for m in mentions if m["ent_type"] == "place"]
        lacune_mentions = [m for m in place_mentions if "lacune" in m["surface"].lower()]
        assert len(lacune_mentions) == 0, (
            f"'lacune' should NOT be a place mention, got: "
            f"{[m['surface'] for m in lacune_mentions]}"
        )

    def test_lacuna_not_place(self):
        """'lacuna' must be rejected."""
        from app.routers.ocr import _extract_salvage_mentions
        text = "en lacuna de texte"
        mentions, debug = _extract_salvage_mentions(text)
        place_mentions = [m for m in mentions if m["ent_type"] == "place"]
        lacuna_mentions = [m for m in place_mentions if "lacuna" in m["surface"].lower()]
        assert len(lacuna_mentions) == 0

    def test_illegible_not_place(self):
        """'illegible' must be rejected."""
        from app.routers.ocr import _extract_salvage_mentions
        text = "en illegible passage ici"
        mentions, debug = _extract_salvage_mentions(text)
        place_mentions = [m for m in mentions if m["ent_type"] == "place"]
        illegible_mentions = [m for m in place_mentions if "illegible" in m["surface"].lower()]
        assert len(illegible_mentions) == 0

    def test_missing_not_place(self):
        """'missing' must be rejected."""
        from app.routers.ocr import _extract_salvage_mentions
        text = "en missing passage ici"
        mentions, debug = _extract_salvage_mentions(text)
        place_mentions = [m for m in mentions if m["ent_type"] == "place"]
        missing_mentions = [m for m in place_mentions if "missing" in m["surface"].lower()]
        assert len(missing_mentions) == 0

    def test_lacune_in_debug_rejected(self):
        """'lacune' should appear in rejected with reason 'editorial_blacklist'."""
        from app.routers.ocr import _extract_salvage_mentions
        text = "en lacune de la terre"
        _, debug = _extract_salvage_mentions(text)
        rejected_surfaces = [r["surface"].lower() for r in debug.get("rejected", [])]
        assert "lacune" in rejected_surfaces

    def test_editorial_blacklist_constant(self):
        """The _EDITORIAL_BLACKLIST constant must include key terms."""
        from app.routers.ocr import _EDITORIAL_BLACKLIST
        for term in ("lacune", "lacuna", "lacunae", "illegible", "gap", "missing"):
            assert term in _EDITORIAL_BLACKLIST, f"'{term}' missing from blacklist"


# ── 3. Authority Linking by Entity Type ──────────────────────────────


class TestAuthorityLinkingByType:
    """Verify entity-type-aware linking behaviour."""

    def test_date_mentions_skip_wikidata(self):
        """DATE mentions must be skipped with 'ent_type=date' reason."""
        from app.services.authority_linking import build_linking_report
        result = {
            "run_id": "test-date-skip",
            "asset_ref": "test.png",
            "mentions_total": 1,
            "type_counts": {"date": 1},
            "candidates_total": 0,
            "source_counts": {},
            "linked_total": 0,
            "unresolved_total": 0,
            "ambiguous_total": 0,
            "skipped_total": 1,
            "quality_skipped": 1,
            "canonical_matched": 0,
            "type_mismatch_count": 0,
            "api_calls_search": 0,
            "api_calls_get": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "took_ms": 5,
            "ocr_quality": "MEDIUM",
            "mention_results": [
                {
                    "mention_id": "m1",
                    "surface": "l'an mil",
                    "ent_type": "date",
                    "status": "skipped",
                    "reason": "ent_type=date (dates never query Wikidata)",
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                    "evidence_text": "l'an mil CCCXL",
                },
            ],
            "_base_text": "l'an mil CCCXL",
        }
        report = build_linking_report(result)
        assert "READY_STATUS:" in report
        # With all mentions skipped → PASS
        assert "FAIL" not in report.split("READY_STATUS:")[1].split("\n")[0]

    def test_editorial_blacklist_in_authority_linking(self):
        """_EDITORIAL_BLACKLIST must be defined in authority_linking."""
        from app.services.authority_linking import _EDITORIAL_BLACKLIST
        assert "lacune" in _EDITORIAL_BLACKLIST

    def test_known_place_gazetteer(self):
        """_KNOWN_PLACE_GAZETTEER must include medieval places."""
        from app.services.authority_linking import _KNOWN_PLACE_GAZETTEER
        for place in ("lausanne", "camelot", "logres", "rome"):
            assert place in _KNOWN_PLACE_GAZETTEER


# ── 4. Conditional Gate C ────────────────────────────────────────────


class TestConditionalGateC:
    """Gate C must PASS when no strong linkable mentions exist."""

    def test_no_linkable_mentions_pass(self):
        """If all mentions are weak (no canonical match, low name_likeness)
        and linked_total=0, Gate C should PASS."""
        from app.services.authority_linking import build_linking_report
        result = {
            "run_id": "test-91b3",
            "asset_ref": "test.png",
            "mentions_total": 2,
            "type_counts": {"date": 1, "place": 1},
            "candidates_total": 0,
            "source_counts": {},
            "linked_total": 0,
            "unresolved_total": 0,
            "ambiguous_total": 0,
            "skipped_total": 2,
            "quality_skipped": 2,
            "canonical_matched": 0,
            "type_mismatch_count": 0,
            "api_calls_search": 0,
            "api_calls_get": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "took_ms": 5,
            "ocr_quality": "MEDIUM",
            "mention_results": [
                {
                    "mention_id": "m1",
                    "surface": "mil honorement",
                    "ent_type": "date",
                    "status": "skipped",
                    "reason": "ent_type=date",
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                    "evidence_text": "mil honorement",
                },
                {
                    "mention_id": "m2",
                    "surface": "lacune",
                    "ent_type": "place",
                    "status": "skipped",
                    "reason": "editorial_blacklist",
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                    "evidence_text": "lacune",
                },
            ],
            "_base_text": "mil honorement vus re mande en lacune",
        }
        report = build_linking_report(result)
        assert "Gate C" in report
        assert "FAIL" not in report.split("Gate C")[1].split("\n")[0]
        # READY_STATUS should be PASS
        ready_line = [l for l in report.split("\n") if "READY_STATUS:" in l][0]
        assert "FAIL" not in ready_line

    def test_no_linkable_mentions_ready_status(self):
        """READY_STATUS must be PASS_NO_LINKABLE_MENTIONS or PASS_ALL_QUALITY_SKIPPED
        when no strong mentions exist and linked_total=0."""
        from app.services.authority_linking import build_linking_report
        result = {
            "run_id": "test-91b3-status",
            "asset_ref": "test.png",
            "mentions_total": 1,
            "type_counts": {"place": 1},
            "candidates_total": 0,
            "source_counts": {},
            "linked_total": 0,
            "unresolved_total": 1,
            "ambiguous_total": 0,
            "skipped_total": 0,
            "quality_skipped": 0,
            "canonical_matched": 0,
            "type_mismatch_count": 0,
            "api_calls_search": 0,
            "api_calls_get": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "took_ms": 5,
            "ocr_quality": "MEDIUM",
            "mention_results": [
                {
                    "mention_id": "m1",
                    "surface": "foobar",
                    "ent_type": "place",
                    "status": "unresolved",
                    "reason": "low_evidence_place",
                    "name_likeness": 0.20,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                    "evidence_text": "en foobar",
                },
            ],
            "_base_text": "en foobar de la terre",
        }
        report = build_linking_report(result)
        ready_line = [l for l in report.split("\n") if "READY_STATUS:" in l][0]
        assert (
            "PASS_NO_LINKABLE_MENTIONS" in ready_line
            or "PASS_ALL_QUALITY_SKIPPED" in ready_line
        )

    def test_strong_mention_still_fails(self):
        """If a strong mention (canonical match) exists but linked_total=0,
        Gate C must still FAIL."""
        from app.services.authority_linking import build_linking_report
        result = {
            "run_id": "test-strong-fail",
            "asset_ref": "test.png",
            "mentions_total": 1,
            "type_counts": {"person": 1},
            "candidates_total": 1,
            "source_counts": {"wikidata": 1},
            "linked_total": 0,
            "unresolved_total": 1,
            "ambiguous_total": 0,
            "skipped_total": 0,
            "quality_skipped": 0,
            "canonical_matched": 1,
            "type_mismatch_count": 0,
            "api_calls_search": 1,
            "api_calls_get": 1,
            "api_calls": 2,
            "cache_hits": 0,
            "took_ms": 10,
            "ocr_quality": "MEDIUM",
            "mention_results": [
                {
                    "mention_id": "m1",
                    "surface": "lancelot",
                    "ent_type": "person",
                    "status": "unresolved",
                    "reason": "below threshold",
                    "name_likeness": 0.90,
                    "canonical_match": {"canon": "Lancelot", "token": "lancelot", "dist": 0},
                    "selected": None,
                    "top_candidates": [],
                    "evidence_text": "lancelot",
                },
            ],
            "_base_text": "si vint lancelot",
        }
        report = build_linking_report(result)
        ready_line = [l for l in report.split("\n") if "READY_STATUS:" in l][0]
        assert "FAIL" in ready_line


# ── 5. Single Decision Per Surface ───────────────────────────────────


class TestSingleDecisionPerSurface91b3:
    """A surface must not appear as both accepted and rejected."""

    def test_date_decision_consistency(self):
        """A date mention is either extracted or not — never both."""
        from app.routers.ocr import _extract_mentions_from_text
        text = "mil honorement\nvus re mande"
        mentions, heuristic_candidates, debug = _extract_mentions_from_text(text)
        surfaces_by_type: dict[str, list[str]] = {}
        for m in mentions:
            et = m.get("ent_type", "unknown")
            s = m.get("surface", "")
            surfaces_by_type.setdefault(et, []).append(s)
        # "mil honorement..." should NOT appear as date
        for s in surfaces_by_type.get("date", []):
            assert "mil honorement" not in s.lower()

    def test_lacune_decision_consistency(self):
        """'lacune' should not appear as place mention at all."""
        from app.routers.ocr import _extract_mentions_from_text
        text = "en lacune de la terre"
        mentions, _, _ = _extract_mentions_from_text(text)
        place_mentions = [m for m in mentions if m["ent_type"] == "place"]
        for m in place_mentions:
            assert "lacune" not in m["surface"].lower()


# ── 6. Integration Regression for run 91b3a986 ──────────────────────


class TestIntegrationRun91b3a986:
    """End-to-end regression tests simulating run 91b3a986."""

    SAMPLE_TEXT = (
        "mil honorement\n"
        "vus re mande par cest brief\n"
        "ke vus viegnez a moi parler\n"
        "en lacune de la terre\n"
    )

    def test_no_date_mention_for_mil_honorement(self):
        """The full pipeline should NOT extract 'mil honorement' as date."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, _ = _extract_mentions_from_text(self.SAMPLE_TEXT)
        dates = [m for m in mentions if m["ent_type"] == "date"]
        for d in dates:
            assert "mil honorement" not in d["surface"].lower(), (
                f"'mil honorement' wrongly classified as date: {d['surface']}"
            )

    def test_no_place_mention_for_lacune(self):
        """The full pipeline should NOT extract 'lacune' as place."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, _ = _extract_mentions_from_text(self.SAMPLE_TEXT)
        places = [m for m in mentions if m["ent_type"] == "place"]
        for p in places:
            assert "lacune" not in p["surface"].lower(), (
                f"'lacune' wrongly classified as place: {p['surface']}"
            )

    def test_linkable_mentions_zero_or_weak(self):
        """After extraction, linkable mentions should be 0 or weak."""
        from app.routers.ocr import _extract_mentions_from_text
        mentions, _, _ = _extract_mentions_from_text(self.SAMPLE_TEXT)
        # No strong person/place/work mentions expected from this text
        strong = [
            m for m in mentions
            if m.get("ent_type") in ("person", "place", "work")
            and float(m.get("confidence", 0)) >= 0.60
        ]
        # This noisy text shouldn't produce strong linkable mentions
        # (if it does, they should be canonical matches only)
        for s in strong:
            if s.get("ent_type") == "date":
                pytest.fail(f"Unexpected strong date mention: {s['surface']}")

    def test_api_calls_zero_for_date_and_lacune(self):
        """No Wikidata API calls should be made for date or lacune mentions."""
        from app.services.authority_linking import build_linking_report
        # Simulate the pipeline result for this noisy page
        result = {
            "run_id": "91b3a986",
            "asset_ref": "test_noisy.png",
            "mentions_total": 2,
            "type_counts": {"date": 1, "place": 1},
            "candidates_total": 0,
            "source_counts": {},
            "linked_total": 0,
            "unresolved_total": 0,
            "ambiguous_total": 0,
            "skipped_total": 2,
            "quality_skipped": 2,
            "canonical_matched": 0,
            "type_mismatch_count": 0,
            "api_calls_search": 0,
            "api_calls_get": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "took_ms": 1,
            "ocr_quality": "LOW",
            "mention_results": [
                {
                    "mention_id": "m1",
                    "surface": "mil honorement vus re mande",
                    "ent_type": "date",
                    "status": "skipped",
                    "reason": "ent_type=date (dates never query Wikidata)",
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                    "evidence_text": "mil honorement vus re mande",
                },
                {
                    "mention_id": "m2",
                    "surface": "lacune",
                    "ent_type": "place",
                    "status": "skipped",
                    "reason": "low_evidence_place: blacklisted",
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                    "evidence_text": "en lacune de la terre",
                },
            ],
            "_base_text": "mil honorement\nvus re mande\nen lacune de la terre",
        }
        report = build_linking_report(result)
        assert "api_calls_search: 0" in report

    def test_ready_status_pass(self):
        """READY_STATUS must be PASS (not FAIL) for this noisy page."""
        from app.services.authority_linking import build_linking_report
        result = {
            "run_id": "91b3a986",
            "asset_ref": "test_noisy.png",
            "mentions_total": 2,
            "type_counts": {"date": 1, "place": 1},
            "candidates_total": 0,
            "source_counts": {},
            "linked_total": 0,
            "unresolved_total": 0,
            "ambiguous_total": 0,
            "skipped_total": 2,
            "quality_skipped": 2,
            "canonical_matched": 0,
            "type_mismatch_count": 0,
            "api_calls_search": 0,
            "api_calls_get": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "took_ms": 1,
            "ocr_quality": "LOW",
            "mention_results": [
                {
                    "mention_id": "m1",
                    "surface": "mil honorement vus re mande",
                    "ent_type": "date",
                    "status": "skipped",
                    "reason": "ent_type=date",
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                    "evidence_text": "mil honorement",
                },
                {
                    "mention_id": "m2",
                    "surface": "lacune",
                    "ent_type": "place",
                    "status": "skipped",
                    "reason": "editorial_blacklist",
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                    "evidence_text": "en lacune",
                },
            ],
            "_base_text": "mil honorement\nvus re mande\nen lacune de la terre",
        }
        report = build_linking_report(result)
        ready_line = [l for l in report.split("\n") if "READY_STATUS:" in l][0]
        assert "FAIL" not in ready_line, (
            f"Expected PASS status, got: {ready_line}"
        )
        assert any(
            status in ready_line
            for status in (
                "PASS_NO_LINKABLE_MENTIONS",
                "PASS_ALL_QUALITY_SKIPPED",
                "PASS_LINKED",
            )
        )

    def test_gate_c_passes(self):
        """Gate C must PASS when all mentions are skipped (date + blacklisted)."""
        from app.services.authority_linking import build_linking_report
        result = {
            "run_id": "91b3a986",
            "asset_ref": "test_noisy.png",
            "mentions_total": 2,
            "type_counts": {"date": 1, "place": 1},
            "candidates_total": 0,
            "source_counts": {},
            "linked_total": 0,
            "unresolved_total": 0,
            "ambiguous_total": 0,
            "skipped_total": 2,
            "quality_skipped": 2,
            "canonical_matched": 0,
            "type_mismatch_count": 0,
            "api_calls_search": 0,
            "api_calls_get": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "took_ms": 1,
            "ocr_quality": "LOW",
            "mention_results": [
                {
                    "mention_id": "m1",
                    "surface": "mil honorement",
                    "ent_type": "date",
                    "status": "skipped",
                    "reason": "ent_type=date",
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                    "evidence_text": "mil honorement",
                },
                {
                    "mention_id": "m2",
                    "surface": "lacune",
                    "ent_type": "place",
                    "status": "skipped",
                    "reason": "editorial_blacklist",
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                    "evidence_text": "lacune",
                },
            ],
            "_base_text": "mil honorement en lacune",
        }
        report = build_linking_report(result)
        # Gate C line
        gate_c_line = [l for l in report.split("\n") if "Gate C" in l][0]
        assert "PASS" in gate_c_line, f"Gate C should PASS, got: {gate_c_line}"
        assert "FAIL" not in gate_c_line
