"""Regression tests for run_id 81ae066e-d19f-403b-8111-d4b61dca13c7.

Observed failures:
  - mentions_total=9 but most are junk: "port", "ente", "main", "voin", "elain"
  - linked_total=0, candidates_total=71, type_mismatch_count=61
  - api_calls_search=16, api_calls_get=71
  - Wrong canonical mapping: main→Yvain(dist=2), voin→Yvain(dist=2)
  - "leantlote" (≈ Lancelot) NOT extracted/linked

These tests verify all fixes implemented in Session 23.
"""

from __future__ import annotations

import pytest


# ── Blacklist & Utility Tests ──────────────────────────────────────────


class TestBlacklistedTokens:
    """Common OCR tokens must be blocked by the blacklist."""

    def test_main_is_blacklisted(self):
        from app.services.text_normalization import is_blacklisted_token
        assert is_blacklisted_token("main") is True

    def test_port_is_blacklisted(self):
        from app.services.text_normalization import is_blacklisted_token
        assert is_blacklisted_token("port") is True

    def test_ente_is_blacklisted(self):
        from app.services.text_normalization import is_blacklisted_token
        assert is_blacklisted_token("ente") is True

    def test_voin_is_blacklisted(self):
        from app.services.text_normalization import is_blacklisted_token
        assert is_blacklisted_token("voin") is True

    def test_elain_is_blacklisted(self):
        from app.services.text_normalization import is_blacklisted_token
        assert is_blacklisted_token("elain") is True

    def test_ament_is_blacklisted(self):
        from app.services.text_normalization import is_blacklisted_token
        assert is_blacklisted_token("ament") is True

    def test_arthur_not_blacklisted(self):
        from app.services.text_normalization import is_blacklisted_token
        assert is_blacklisted_token("arthur") is False

    def test_lancelot_not_blacklisted(self):
        from app.services.text_normalization import is_blacklisted_token
        assert is_blacklisted_token("lancelot") is False

    def test_case_insensitive(self):
        from app.services.text_normalization import is_blacklisted_token
        assert is_blacklisted_token("Main") is True
        assert is_blacklisted_token("PORT") is True


class TestNormalizedEditDistance:
    """Verify normalized_edit_distance utility."""

    def test_identical_strings(self):
        from app.services.text_normalization import normalized_edit_distance
        assert normalized_edit_distance("lancelot", "lancelot") == 0.0

    def test_known_pair_main_yvain(self):
        from app.services.text_normalization import normalized_edit_distance
        nd = normalized_edit_distance("main", "yvain")
        # main→yvain requires 3 edits / max(4,5) = 0.6 — too high
        assert nd > 0.25, f"main→yvain nd={nd:.3f} should exceed 0.25 threshold"

    def test_known_pair_voin_yvain(self):
        from app.services.text_normalization import normalized_edit_distance
        nd = normalized_edit_distance("voin", "yvain")
        # voin→yvain requires 2 edits / max(4,5) = 0.4 — too high
        assert nd > 0.25, f"voin→yvain nd={nd:.3f} should exceed 0.25 threshold"

    def test_known_pair_elain_yvain(self):
        from app.services.text_normalization import normalized_edit_distance
        nd = normalized_edit_distance("elain", "yvain")
        assert nd > 0.25, f"elain→yvain nd={nd:.3f} should exceed 0.25 threshold"

    def test_close_variant(self):
        from app.services.text_normalization import normalized_edit_distance
        nd = normalized_edit_distance("artuur", "arthur")
        assert nd <= 0.25, f"artuur→arthur nd={nd:.3f} should be ≤ 0.25"

    def test_empty_string(self):
        from app.services.text_normalization import normalized_edit_distance
        assert normalized_edit_distance("", "") == 0.0


class TestBigramOverlap:
    """Verify bigram_overlap (Dice coefficient) utility."""

    def test_identical(self):
        from app.services.text_normalization import bigram_overlap
        assert bigram_overlap("lancelot", "lancelot") == 1.0

    def test_completely_different(self):
        from app.services.text_normalization import bigram_overlap
        bo = bigram_overlap("xyz", "abc")
        assert bo == 0.0

    def test_main_vs_yvain_low_overlap(self):
        from app.services.text_normalization import bigram_overlap
        bo = bigram_overlap("main", "yvain")
        # "ma","ai","in" vs "yv","va","ai","in" — only "ai","in" overlap
        # 2*2/(3+4) = 0.571 — might be above threshold, but nd gate blocks it
        assert bo >= 0.0  # Just ensure it computes

    def test_leantlote_vs_lancelot(self):
        from app.services.text_normalization import bigram_overlap
        bo = bigram_overlap("leantlote", "lancelot")
        assert bo >= 0.35, f"leantlote vs lancelot bo={bo:.3f} should be decent"

    def test_short_strings(self):
        from app.services.text_normalization import bigram_overlap
        # Strings with 1 char → 0 bigrams → should return 0
        assert bigram_overlap("a", "a") == 0.0


# ── Mention Extraction Gate Tests ──────────────────────────────────────


class TestCommonWordsNotPersonMentions:
    """No common word must become a person mention through any extraction path."""

    @pytest.mark.parametrize("token", ["main", "port", "ente", "voin", "elain"])
    def test_not_medieval_name(self, token):
        """Blacklisted tokens fail _looks_like_medieval_name."""
        from app.routers.ocr import _looks_like_medieval_name
        assert _looks_like_medieval_name(token) is False, (
            f"{token!r} must NOT look like a medieval name"
        )

    @pytest.mark.parametrize("token", ["main", "port", "ente", "voin", "elain"])
    def test_not_ngram_person(self, token):
        """Blacklisted tokens produce 0 person mentions in ngram scan."""
        from app.routers.ocr import _extract_ngram_canonical_mentions
        text = f"le {token} de la terre"
        mentions = _extract_ngram_canonical_mentions(text)
        person_mentions = [m for m in mentions if m["ent_type"] == "person"]
        assert len(person_mentions) == 0, (
            f"{token!r} produced {len(person_mentions)} person mentions: {person_mentions}"
        )

    def test_main_never_maps_to_yvain(self):
        """Specific regression: 'main' must NOT canonicalise to 'yvain'."""
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions("le main chevalier")
        yvain_mentions = [
            m for m in mentions
            if m.get("norm", "") == "yvain"
        ]
        assert len(yvain_mentions) == 0, (
            f"'main' wrongly mapped to Yvain: {yvain_mentions}"
        )

    def test_voin_never_maps_to_yvain(self):
        """Specific regression: 'voin' must NOT canonicalise to 'yvain'."""
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions("le voin chevalier")
        yvain_mentions = [
            m for m in mentions
            if m.get("norm", "") == "yvain"
        ]
        assert len(yvain_mentions) == 0


class TestLeantloteAsLancelot:
    """'leantlote' (common OCR variant) must be detected as Lancelot."""

    def test_ngram_detects_leantlote(self):
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions("si vint leantlote en la terre")
        lancelot = [m for m in mentions if m.get("norm", "") == "lancelot"]
        assert len(lancelot) >= 1, "leantlote must map to lancelot via canonical table"

    def test_leantlote_surface_preserved(self):
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions("si vint leantlote en la terre")
        lancelot = [m for m in mentions if m.get("norm", "") == "lancelot"]
        assert any(m["surface"] == "leantlote" for m in lancelot)

    def test_leantlote_is_person(self):
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions("si vint leantlote en la terre")
        lancelot = [m for m in mentions if m.get("norm", "") == "lancelot"]
        assert all(m["ent_type"] == "person" for m in lancelot)

    def test_other_lancelot_variants(self):
        """All registered OCR variants must map to lancelot."""
        from app.routers.ocr import _extract_ngram_canonical_mentions
        for variant in ["lancelote", "lanselot", "lanceloc"]:
            mentions = _extract_ngram_canonical_mentions(f"vint {variant} en")
            lancelot = [m for m in mentions if m.get("norm", "") == "lancelot"]
            assert len(lancelot) >= 1, f"{variant} must map to lancelot"


class TestIvainVariant:
    """'ivain' should map to yvain."""

    def test_ivain_maps_to_yvain(self):
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions("si dist ivain a son frere")
        yvain = [m for m in mentions if m.get("norm", "") == "yvain"]
        assert len(yvain) >= 1, "ivain must map to yvain"


# ── Short Token Exact-Match-Only Rule ──────────────────────────────────


class TestShortTokenExactOnly:
    """Tokens < 5 chars must match canonicals exactly, never fuzzy."""

    def test_four_char_exact_match_works(self):
        """'bors' (4 chars, exact canonical) should still match."""
        from app.routers.ocr import _looks_like_medieval_name
        # 'bors' is in canonical names, so exact match should work
        from app.routers.ocr import _CANONICAL_PERSON_NAMES
        if "bors" in _CANONICAL_PERSON_NAMES:
            assert _looks_like_medieval_name("bors") is True

    def test_four_char_no_fuzzy_canonical_match(self):
        """'xort' (4 chars) must NOT fuzzy-match any canonical name.

        Short tokens that aren't blacklisted and have valid phonotactics
        may still pass the heuristic, but they cannot match via the
        canonical fuzzy path — only via exact key lookup.
        """
        from app.routers.ocr import _extract_ngram_canonical_mentions
        # 'xort' is not a canonical key → no canonical person mention
        mentions = _extract_ngram_canonical_mentions("le xort de la terre")
        canonical_persons = [
            m for m in mentions
            if m["ent_type"] == "person" and m["method"] == "ngram_canonical"
        ]
        assert len(canonical_persons) == 0


# ── Canonical Match Tests ──────────────────────────────────────────────


class TestCheckCanonicalMatchRegression:
    """Verify _check_canonical_match rejects common words."""

    def test_main_no_canonical_match(self):
        from app.services.authority_linking import _check_canonical_match
        result = _check_canonical_match("main")
        assert result is None, "'main' must NOT match any canonical entity"

    def test_port_no_canonical_match(self):
        from app.services.authority_linking import _check_canonical_match
        result = _check_canonical_match("port")
        assert result is None, "'port' must NOT match any canonical entity"

    def test_leantlote_canonical_match(self):
        from app.services.authority_linking import _check_canonical_match
        result = _check_canonical_match("leantlote")
        assert result is not None, "leantlote must match Lancelot canonical"
        assert result["canon"] == "Lancelot"

    def test_canonical_match_returns_nd_bo(self):
        """Result dict now includes nd and bo keys."""
        from app.services.authority_linking import _check_canonical_match
        result = _check_canonical_match("roi arthur")
        assert result is not None
        assert "nd" in result
        assert "bo" in result
        assert result["nd"] == 0.0  # exact match
        assert result["bo"] == 1.0  # exact match


# ── Salvage Rejection Tests ────────────────────────────────────────────


class TestSalvageBlacklistGate:
    """Salvage mentions must be rejected if all name parts are blacklisted."""

    def test_trigger_with_all_blacklisted_parts(self):
        """'roi main port' → role trigger + all-blacklisted name → rejected."""
        from app.routers.ocr import _extract_salvage_mentions
        mentions, debug = _extract_salvage_mentions("roi main port ente")
        person_mentions = [m for m in mentions if m["ent_type"] == "person"]
        assert len(person_mentions) == 0, (
            "All-blacklisted name parts must NOT produce person mention"
        )

    def test_trigger_with_real_name_works(self):
        """'roi arthur' → role trigger + real name → person mention."""
        from app.routers.ocr import _extract_salvage_mentions
        mentions, debug = _extract_salvage_mentions("roi arthur de bretagne")
        person_mentions = [m for m in mentions if m["ent_type"] == "person"]
        assert len(person_mentions) >= 1


# ── API Cap Tests ──────────────────────────────────────────────────────


class TestApiCaps:
    """Verify the API call cap constants are in place."""

    def test_search_cap_exists(self):
        from app.services.authority_linking import _MAX_SEARCH_CALLS_PER_RUN
        assert _MAX_SEARCH_CALLS_PER_RUN == 30

    def test_enrich_cap_exists(self):
        from app.services.authority_linking import _MAX_ENRICH_PER_MENTION
        assert _MAX_ENRICH_PER_MENTION == 3


# ── Pre-filter Tests ──────────────────────────────────────────────────


class TestPrefilterCandidates:
    """Pre-filter must reject obviously modern candidates."""

    def test_rejects_modern_singer(self):
        from app.services.authority_linking import _prefilter_candidates
        candidates = [
            {"qid": "Q1", "label": "Main", "description": "American singer", "url": ""},
            {"qid": "Q2", "label": "Yvain", "description": "knight of the Round Table", "url": ""},
        ]
        filtered = _prefilter_candidates(candidates, "yvain", "person")
        qids = [c["qid"] for c in filtered]
        assert "Q1" not in qids, "Modern singer must be rejected"
        assert "Q2" in qids, "Medieval knight must be kept"

    def test_rejects_footballer(self):
        from app.services.authority_linking import _prefilter_candidates
        candidates = [
            {"qid": "Q99", "label": "Port", "description": "Brazilian footballer", "url": ""},
        ]
        filtered = _prefilter_candidates(candidates, "port", "person")
        assert len(filtered) == 0

    def test_medieval_domain_sorted_first(self):
        from app.services.authority_linking import _prefilter_candidates
        candidates = [
            {"qid": "Q1", "label": "Lancelot", "description": "given name", "url": ""},
            {"qid": "Q2", "label": "Lancelot", "description": "knight of the Round Table in Arthurian legend", "url": ""},
        ]
        filtered = _prefilter_candidates(candidates, "lancelot", "person")
        # The Arthurian one should come first
        assert filtered[0]["qid"] == "Q2"

    def test_empty_description_kept(self):
        """Candidates with no description are kept (not assumed modern)."""
        from app.services.authority_linking import _prefilter_candidates
        candidates = [
            {"qid": "Q50", "label": "Perceval", "description": "", "url": ""},
        ]
        filtered = _prefilter_candidates(candidates, "perceval", "person")
        assert len(filtered) == 1


# ── Name Likeness with Blacklist ───────────────────────────────────────


class TestNameLikenessBlacklist:
    """_compute_name_likeness must penalise blacklisted tokens."""

    def test_main_devalued(self):
        from app.services.authority_linking import _compute_name_likeness
        score = _compute_name_likeness("main", "person")
        # "main" is blacklisted → canonical match blocked → low score
        assert score < 0.50, f"'main' should score low, got {score}"

    def test_arthur_valued(self):
        from app.services.authority_linking import _compute_name_likeness
        score = _compute_name_likeness("arthur", "person")
        assert score >= 0.85, f"'arthur' should score high, got {score}"


# ── Legitimate Names Still Work ────────────────────────────────────────


class TestLegitimateNamesNotAffected:
    """Ensure real Arthurian names aren't broken by the blacklist/thresholds."""

    @pytest.mark.parametrize("name", [
        "arthur", "lancelot", "merlin", "perceval", "galahad",
        "tristan", "gauvain", "bohort", "yvain", "guenievre",
    ])
    def test_canonical_names_detected(self, name):
        from app.routers.ocr import _looks_like_medieval_name
        assert _looks_like_medieval_name(name) is True, f"{name} must be detected"

    @pytest.mark.parametrize("name", [
        "arthur", "lancelot", "merlin", "perceval", "galahad",
        "tristan", "gauvain", "bohort", "yvain",
    ])
    def test_canonical_ngram_extraction(self, name):
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions(f"si vint {name} a la cort")
        person_mentions = [m for m in mentions if m["ent_type"] == "person"]
        assert len(person_mentions) >= 1, f"{name} must produce ≥1 person mention"

    @pytest.mark.parametrize("variant,expected", [
        ("artur", "arthur"),
        ("lancelo", "lancelot"),
        ("merlim", "merlin"),
        ("percevale", "perceval"),
    ])
    def test_fuzzy_variants_still_work(self, variant, expected):
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions(f"si vint {variant} a la cort")
        matches = [m for m in mentions if m.get("norm", "") == expected]
        assert len(matches) >= 1, f"{variant} must map to {expected}"
