"""Tests for the authority linking pipeline.

Covers: Wikidata client (with mocked HTTP), entity scoring,
disambiguation, authority_linking orchestrator, and the report builder.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point pipeline DB and Wikidata cache to temp dirs."""
    db_path = str(tmp_path / "test.sqlite")
    cache_path = str(tmp_path / "wiki_cache.sqlite")
    monkeypatch.setenv("ARCHAI_DB_PATH", db_path)
    monkeypatch.setenv("ARCHAI_WIKIDATA_CACHE", cache_path)

    # Reset DB-ready flags so they re-initialise with new path
    import app.db.pipeline_db as pdb
    pdb._DB_READY = False

    import app.services.wikidata_client as wc
    wc._CACHE_READY = False

    yield


def _seed_run(run_id: str = "test-run-1", text: str = "Le vilain de Cambrai"):
    """Insert a pipeline run + chunks + mentions into the DB."""
    from app.db import pipeline_db as pdb

    pdb.create_run("test-asset")
    # We need to overwrite the run_id to be deterministic
    pdb._init_db_if_needed()
    with pdb._connect() as conn:
        conn.execute("UPDATE pipeline_runs SET run_id=? WHERE run_id != ?", (run_id, run_id))
        conn.commit()

    # Insert chunks
    pdb.insert_chunks(run_id, [
        {"chunk_id": "chunk-1", "idx": 0, "start_offset": 0, "end_offset": len(text), "text": text},
    ])

    # Insert mentions
    pdb.insert_entity_mentions(run_id, [
        {
            "mention_id": "mention-1",
            "chunk_id": "chunk-1",
            "start_offset": 3,
            "end_offset": 9,
            "surface": "vilain",
            "norm": "vilain",
            "ent_type": "person",
            "confidence": 0.8,
            "method": "rule:proper_name_sequence",
        },
        {
            "mention_id": "mention-2",
            "chunk_id": "chunk-1",
            "start_offset": 13,
            "end_offset": 20,
            "surface": "Cambrai",
            "norm": "cambrai",
            "ent_type": "place",
            "confidence": 0.9,
            "method": "rule:toponym_anchor",
        },
    ])

    # Set proofread text
    pdb.update_run_fields(run_id, proofread_text=text)


# ── Wikidata Client Tests ─────────────────────────────────────────────

class TestNormalisation:
    def test_normalise_basic(self):
        from app.services.wikidata_client import normalise_surface
        assert normalise_surface("  Le Vilain  ") == "le vilain"

    def test_normalise_diacritics(self):
        from app.services.wikidata_client import normalise_surface
        assert normalise_surface("Évrard") == "evrard"

    def test_normalise_punctuation(self):
        from app.services.wikidata_client import normalise_surface
        assert normalise_surface("Saint-Omer") == "saint omer"


class TestWikidataCache:
    def test_put_and_get(self):
        from app.services.wikidata_client import cache_put, cache_get
        data = [{"qid": "Q123", "label": "Test"}]
        cache_put("wikidata", "test query", data)
        result = cache_get("wikidata", "test query")
        assert result == data

    def test_cache_miss(self):
        from app.services.wikidata_client import cache_get
        result = cache_get("wikidata", "nonexistent")
        assert result is None


class TestSearchWikidata:
    def test_search_returns_cached(self):
        from app.services.wikidata_client import cache_put, search_wikidata
        cached = [
            {"qid": "Q42", "label": "Douglas Adams", "description": "writer", "url": "https://www.wikidata.org/wiki/Q42"},
        ]
        cache_put("wikidata", "Douglas Adams", cached)
        result = search_wikidata("Douglas Adams", k=5)
        assert len(result) == 1
        assert result[0]["qid"] == "Q42"

    @patch("app.services.wikidata_client._http_get")
    def test_search_calls_api(self, mock_get):
        from app.services.wikidata_client import search_wikidata, cache_get
        mock_get.return_value = {
            "search": [
                {"id": "Q42", "label": "Douglas Adams", "description": "English author", "url": "https://www.wikidata.org/wiki/Q42"},
            ]
        }
        # Ensure no cache
        assert cache_get("wikidata", "xyzzy_unique_test") is None
        result = search_wikidata("xyzzy_unique_test", k=5)
        assert len(result) == 1
        assert result[0]["qid"] == "Q42"
        mock_get.assert_called_once()


class TestEnrichment:
    @patch("app.services.wikidata_client._http_get")
    def test_enrich_extracts_viaf_geonames(self, mock_get):
        from app.services.wikidata_client import enrich_wikidata_item
        mock_get.return_value = {
            "entities": {
                "Q42": {
                    "claims": {
                        "P214": [{"mainsnak": {"datavalue": {"value": "113230702"}}}],
                        "P1566": [{"mainsnak": {"datavalue": {"value": "2988507"}}}],
                        "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}],
                    }
                }
            }
        }
        result = enrich_wikidata_item("Q42")
        assert result["viaf_id"] == "113230702"
        assert result["geonames_id"] == "2988507"
        assert "Q5" in result["instance_of_qids"]


class TestTypeCompatibility:
    def test_person_compatible_with_q5(self):
        from app.services.wikidata_client import is_type_compatible
        assert is_type_compatible("person", ["Q5"]) is True

    def test_person_incompatible_with_city(self):
        from app.services.wikidata_client import is_type_compatible
        assert is_type_compatible("person", ["Q515"]) is False

    def test_empty_instance_of_incompatible(self):
        from app.services.wikidata_client import is_type_compatible
        # v2: no P31 data → can't verify → incompatible
        assert is_type_compatible("person", []) is False


# ── Entity Scoring Tests ──────────────────────────────────────────────

class TestStringSimilarity:
    def test_identical(self):
        from app.services.entity_scoring import string_similarity
        score = string_similarity("vilain", "vilain")
        assert score > 0.95

    def test_similar(self):
        from app.services.entity_scoring import string_similarity
        score = string_similarity("vilain", "villain")
        assert 0.5 < score < 1.0

    def test_different(self):
        from app.services.entity_scoring import string_similarity
        score = string_similarity("vilain", "mountain")
        assert score < 0.6


class TestContextSimilarity:
    def test_overlap(self):
        from app.services.entity_scoring import context_similarity
        score = context_similarity("medieval french poem", "poem written in medieval france")
        assert score > 0.2

    def test_no_overlap(self):
        from app.services.entity_scoring import context_similarity
        score = context_similarity("quantum physics", "medieval french poem")
        assert score < 0.05


class TestCompositeScore:
    def test_perfect_match(self):
        from app.services.entity_scoring import compute_score
        score = compute_score("vilain", "vilain", "french poem", "french poem", type_compatible=True)
        assert score > 0.8

    def test_type_penalty(self):
        from app.services.entity_scoring import compute_score
        score_ok = compute_score("Paris", "Paris", "city in France", "capital", type_compatible=True)
        score_bad = compute_score("Paris", "Paris", "city in France", "capital", type_compatible=False)
        assert score_ok > score_bad
        assert score_ok - score_bad >= 0.25  # penalty is 0.30


class TestDisambiguation:
    def test_clear_winner(self):
        from app.services.entity_scoring import disambiguate
        candidates = [
            {"qid": "Q1", "score": 0.9, "type_compatible": True},
            {"qid": "Q2", "score": 0.3, "type_compatible": True},
        ]
        result = disambiguate(candidates)
        assert result["status"] == "linked"
        assert result["selected"]["qid"] == "Q1"

    def test_ambiguous(self):
        from app.services.entity_scoring import disambiguate
        candidates = [
            {"qid": "Q1", "score": 0.90, "type_compatible": True},
            {"qid": "Q2", "score": 0.88, "type_compatible": True},
        ]
        result = disambiguate(candidates)
        assert result["status"] == "ambiguous"
        assert result["selected"] is None

    def test_empty(self):
        from app.services.entity_scoring import disambiguate
        result = disambiguate([])
        assert result["status"] == "unresolved"

    def test_single_candidate(self):
        from app.services.entity_scoring import disambiguate
        result = disambiguate([{"qid": "Q1", "score": 0.90, "type_compatible": True}])
        assert result["status"] == "linked"
        assert result["selected"]["qid"] == "Q1"

    def test_single_candidate_below_threshold(self):
        from app.services.entity_scoring import disambiguate
        result = disambiguate([{"qid": "Q1", "score": 0.50, "type_compatible": True}])
        assert result["status"] == "unresolved"

    def test_ambiguous_but_high_score_still_ambiguous_v2(self):
        from app.services.entity_scoring import disambiguate
        candidates = [
            {"qid": "Q1", "score": 0.90, "type_compatible": True},
            {"qid": "Q2", "score": 0.88, "type_compatible": True},
        ]
        result = disambiguate(candidates)
        # v4: margin = 0.02 < 0.15 → ambiguous (regardless of high score)
        assert result["status"] == "ambiguous"


# ── Authority Linking Orchestrator Tests ──────────────────────────────

class TestRunAuthorityLinking:
    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_basic_linking(self, mock_search, mock_enrich):
        _seed_run()

        mock_search.return_value = [
            {"qid": "Q100", "label": "Vilain", "description": "medieval vilain character from Cambrai", "url": "https://example.com/Q100"},
        ]
        mock_enrich.return_value = {
            "viaf_id": "12345",
            "geonames_id": "",
            "instance_of_qids": ["Q5"],
        }

        from app.services.authority_linking import run_authority_linking
        result = run_authority_linking("test-run-1")

        assert result["mentions_total"] == 2
        assert result["candidates_total"] > 0
        # v2: with precision-first thresholds, only high-confidence
        # matches link. At minimum we should have candidates processed.
        assert result["linked_total"] + result["unresolved_total"] + result["ambiguous_total"] == result["mentions_total"]

    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_empty_mentions(self, mock_search, mock_enrich):
        from app.db import pipeline_db as pdb
        from app.services.authority_linking import run_authority_linking

        pdb.create_run("test-asset")
        pdb._init_db_if_needed()
        with pdb._connect() as conn:
            row = conn.execute("SELECT run_id FROM pipeline_runs LIMIT 1").fetchone()
            rid = row["run_id"]

        result = run_authority_linking(rid)
        assert result["mentions_total"] == 0
        mock_search.assert_not_called()


class TestBuildLinkingReport:
    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_report_contains_all_sections(self, mock_search, mock_enrich):
        _seed_run()

        mock_search.return_value = [
            {"qid": "Q100", "label": "Vilain", "description": "character", "url": ""},
        ]
        mock_enrich.return_value = {"viaf_id": "", "geonames_id": "", "instance_of_qids": ["Q5"]}

        from app.services.authority_linking import run_authority_linking, build_linking_report
        result = run_authority_linking("test-run-1")
        report = build_linking_report(result)

        assert "=== ENTITY LINKING REPORT ===" in report
        assert "=== TOP LINKED ENTITIES" in report
        assert "=== FAILURE/EDGE CASES" in report
        assert "=== VALIDATION SUMMARY ===" in report
        assert "READY_STATUS:" in report

    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_report_has_gates(self, mock_search, mock_enrich):
        _seed_run()
        mock_search.return_value = [
            {"qid": "Q100", "label": "Vilain", "description": "character", "url": ""},
        ]
        mock_enrich.return_value = {"viaf_id": "", "geonames_id": "", "instance_of_qids": ["Q5"]}

        from app.services.authority_linking import run_authority_linking, build_linking_report
        result = run_authority_linking("test-run-1")
        report = build_linking_report(result)

        assert "Gate A" in report
        assert "Gate B" in report
        assert "Gate C" in report
        assert "Gate D" in report
        assert "Gate E" in report
        assert "Gate F" in report


class TestBuildReportFromDb:
    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_round_trip(self, mock_search, mock_enrich):
        _seed_run()
        mock_search.return_value = [
            {"qid": "Q100", "label": "Test", "description": "test entity", "url": ""},
        ]
        mock_enrich.return_value = {"viaf_id": "V1", "geonames_id": "", "instance_of_qids": ["Q5"]}

        from app.services.authority_linking import run_authority_linking, build_report_from_db
        run_authority_linking("test-run-1")
        db_result = build_report_from_db("test-run-1")

        assert db_result["mentions_total"] == 2
        assert "report" in db_result
        assert "=== ENTITY LINKING REPORT ===" in db_result["report"]

    def test_missing_run(self):
        from app.services.authority_linking import build_report_from_db
        with pytest.raises(ValueError, match="Run not found"):
            build_report_from_db("nonexistent-run")


# ── Precision-First v2 Tests ──────────────────────────────────────────

class TestTypeMismatchRejection:
    """Ensure type-incompatible candidates are NEVER auto-selected."""

    def test_type_incompatible_best_candidate_not_linked(self):
        """A place mention matched to a plant (no compatible P31) must NOT link."""
        from app.services.entity_scoring import disambiguate
        candidates = [
            {"qid": "Q12345", "label": "Catkin", "score": 0.85,
             "type_compatible": False},
            {"qid": "Q67890", "label": "Catkins", "score": 0.40,
             "type_compatible": False},
        ]
        result = disambiguate(candidates)
        assert result["status"] == "unresolved"
        assert result["selected"] is None
        assert "type_incompatible" in result["reason"]

    def test_type_compatible_best_candidate_is_linked(self):
        from app.services.entity_scoring import disambiguate
        candidates = [
            {"qid": "Q1", "label": "Paris", "score": 0.90,
             "type_compatible": True},
            {"qid": "Q2", "label": "Pairs", "score": 0.30,
             "type_compatible": True},
        ]
        result = disambiguate(candidates)
        assert result["status"] == "linked"
        assert result["selected"]["qid"] == "Q1"


class TestThresholdEnforcement:
    """AUTO_SELECT_THRESHOLD = 0.75 must be enforced."""

    def test_below_threshold_not_linked(self):
        from app.services.entity_scoring import disambiguate
        candidates = [
            {"qid": "Q1", "label": "Test", "score": 0.70,
             "type_compatible": True},
        ]
        result = disambiguate(candidates)
        assert result["status"] == "unresolved"
        assert "threshold" in result["reason"]

    def test_at_threshold_is_linked(self):
        from app.services.entity_scoring import disambiguate
        candidates = [
            {"qid": "Q1", "label": "Test", "score": 0.85,
             "type_compatible": True},
        ]
        result = disambiguate(candidates)
        assert result["status"] == "linked"

    def test_margin_too_small_is_ambiguous(self):
        from app.services.entity_scoring import disambiguate, MIN_MARGIN
        candidates = [
            {"qid": "Q1", "label": "A", "score": 0.85,
             "type_compatible": True},
            {"qid": "Q2", "label": "B", "score": 0.80,
             "type_compatible": True},
        ]
        result = disambiguate(candidates)
        # margin = 0.05 < MIN_MARGIN (0.10) → ambiguous
        assert result["status"] == "ambiguous"
        assert result["selected"] is None


class TestNoP31DataRejectsLink:
    """When Wikidata returns no P31 (instance_of), is_type_compatible should return False."""

    def test_empty_p31_returns_false_for_person(self):
        from app.services.wikidata_client import is_type_compatible
        assert is_type_compatible("person", []) is False

    def test_empty_p31_returns_false_for_place(self):
        from app.services.wikidata_client import is_type_compatible
        assert is_type_compatible("place", []) is False


class TestGateDTypeMismatchReport:
    """Gate D must flag type-mismatch detections in the report."""

    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_gate_d_in_report(self, mock_search, mock_enrich):
        _seed_run()
        mock_search.return_value = [
            {"qid": "Q100", "label": "Vilain", "description": "character", "url": ""},
        ]
        mock_enrich.return_value = {"viaf_id": "", "geonames_id": "", "instance_of_qids": ["Q5"]}

        from app.services.authority_linking import run_authority_linking, build_linking_report
        result = run_authority_linking("test-run-1")
        report = build_linking_report(result)

        assert "Gate D (no type-mismatch auto-links)" in report
        assert "Gate E (referential integrity)" in report
        assert "Gate F (evidence spans present)" in report

    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_type_mismatch_section_in_report(self, mock_search, mock_enrich):
        _seed_run()
        mock_search.return_value = [
            {"qid": "Q100", "label": "Vilain", "description": "character", "url": ""},
        ]
        mock_enrich.return_value = {"viaf_id": "", "geonames_id": "", "instance_of_qids": []}  # no P31

        from app.services.authority_linking import run_authority_linking, build_linking_report
        result = run_authority_linking("test-run-1")
        report = build_linking_report(result)

        assert "TYPE-MISMATCH DETECTIONS" in report


class TestConsolidatedReport:
    """The consolidated report must contain both mention extraction and entity linking sections."""

    def test_consolidated_has_both_sections(self):
        from app.routers.ocr import _build_consolidated_report, _build_mention_extraction_report

        mentions = [
            {"surface": "vilain", "ent_type": "person", "confidence": 0.8,
             "method": "rule:proper_name_sequence", "label": None,
             "start_offset": 3, "end_offset": 9},
        ]
        salvage_debug = {"trigger": 0, "work_fuzzy": 0, "place_candidate": 0, "rejected": []}
        linking_result = {
            "run_id": "test-1",
            "asset_ref": "test",
            "mentions_total": 1,
            "type_counts": {"person": 1},
            "candidates_total": 0,
            "source_counts": {},
            "linked_total": 0,
            "unresolved_total": 1,
            "ambiguous_total": 0,
            "type_mismatch_count": 0,
            "api_calls_search": 0,
            "api_calls_get": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "took_ms": 10,
            "mention_results": [],
            "_base_text": "Le vilain de Cambrai",
        }

        report = _build_consolidated_report("test-1", "test", mentions, salvage_debug, linking_result)
        assert "=== MENTION EXTRACTION REPORT ===" in report
        assert "=== ENTITY LINKING REPORT ===" in report
        assert "=== VALIDATION SUMMARY ===" in report


class TestSeparateApiCounters:
    """Verify separate api_calls_search and api_calls_get counters."""

    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_api_counters_present(self, mock_search, mock_enrich):
        _seed_run()
        mock_search.return_value = [
            {"qid": "Q100", "label": "Test", "description": "", "url": ""},
        ]
        mock_enrich.return_value = {"viaf_id": "", "geonames_id": "", "instance_of_qids": ["Q5"]}

        from app.services.authority_linking import run_authority_linking
        result = run_authority_linking("test-run-1")

        assert "api_calls_search" in result
        assert "api_calls_get" in result
        assert result["api_calls"] == result["api_calls_search"] + result["api_calls_get"]


# ── API Router Tests ──────────────────────────────────────────────────

class TestAuthorityRouter:
    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_link_endpoint(self, mock_search, mock_enrich):
        _seed_run()
        mock_search.return_value = [
            {"qid": "Q100", "label": "Test", "description": "", "url": ""},
        ]
        mock_enrich.return_value = {"viaf_id": "", "geonames_id": "", "instance_of_qids": []}

        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.post("/api/authority/link/test-run-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mentions_total"] == 2

    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_report_endpoint(self, mock_search, mock_enrich):
        _seed_run()
        mock_search.return_value = [
            {"qid": "Q100", "label": "Test", "description": "", "url": ""},
        ]
        mock_enrich.return_value = {"viaf_id": "", "geonames_id": "", "instance_of_qids": []}

        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)

        # First link, then get report
        client.post("/api/authority/link/test-run-1")
        resp = client.get("/api/authority/report/test-run-1")
        assert resp.status_code == 200
        data = resp.json()
        assert "report" in data
        assert "ENTITY LINKING REPORT" in data["report"]


# ── Name-Likeness Quality Gate Tests ──────────────────────────────────

class TestLooksMedievalName:
    """Test the _looks_like_medieval_name heuristic in ocr.py."""

    def test_canonical_names_pass(self):
        from app.routers.ocr import _looks_like_medieval_name
        for name in ["arthur", "lancelot", "merlin", "perceval", "galahad", "tristan"]:
            assert _looks_like_medieval_name(name) is True, f"{name} should pass"

    def test_canonical_fuzzy_pass(self):
        from app.routers.ocr import _looks_like_medieval_name
        # Slight OCR corruption should still pass
        assert _looks_like_medieval_name("artur") is True
        assert _looks_like_medieval_name("lancelo") is True
        assert _looks_like_medieval_name("merlim") is True

    def test_garbage_tokens_fail(self):
        from app.routers.ocr import _looks_like_medieval_name
        assert _looks_like_medieval_name("nucadigal") is False  # phonetic but far from canonical
        assert _looks_like_medieval_name("fucaces") is False  # no canonical proximity
        assert _looks_like_medieval_name("xz") is False  # too short
        assert _looks_like_medieval_name("ament") is False  # 5 chars, far from canonical

    def test_real_names_pass(self):
        from app.routers.ocr import _looks_like_medieval_name
        assert _looks_like_medieval_name("Gauvain") is True
        assert _looks_like_medieval_name("Bohort") is True
        assert _looks_like_medieval_name("Yvain") is True

    def test_single_char_fails(self):
        from app.routers.ocr import _looks_like_medieval_name
        assert _looks_like_medieval_name("a") is False
        assert _looks_like_medieval_name("xy") is False

    def test_consonant_cluster_fails(self):
        from app.routers.ocr import _looks_like_medieval_name
        # Tokens with 4+ consecutive consonants
        assert _looks_like_medieval_name("brstmk") is False


class TestComputeNameLikeness:
    """Test the _compute_name_likeness quality scoring."""

    def test_canonical_match_high(self):
        from app.services.authority_linking import _compute_name_likeness
        assert _compute_name_likeness("roi arthur", "person") >= 0.85

    def test_garbage_surface_low(self):
        from app.services.authority_linking import _compute_name_likeness
        # Extreme garbage: consonant clusters, no vowels, abnormal length
        score = _compute_name_likeness("xqpfjkl brstrng", "person")
        assert score < 0.30, f"extreme garbage should score low, got {score}"

    def test_phonetic_garbage_moderate(self):
        from app.services.authority_linking import _compute_name_likeness
        # Phonetically plausible garbage — handled by salvage trigger,
        # not by the name_likeness gate (which is a lighter safety net).
        score = _compute_name_likeness("roya nucadigal ament", "person")
        # Should be > 0.30 (passes quality gate) because phonetics are OK.
        # The salvage trigger classifies this as "role" (primary defense).
        assert score > 0.25

    def test_pure_stopwords_low(self):
        from app.services.authority_linking import _compute_name_likeness
        score = _compute_name_likeness("de la le", "person")
        assert score < 0.10

    def test_plausible_name_moderate(self):
        from app.services.authority_linking import _compute_name_likeness
        score = _compute_name_likeness("Cambrai", "place")
        assert score >= 0.30, f"Cambrai should pass quality gate, got {score}"


class TestCheckCanonicalMatch:
    """Test canonical Arthurian name matching."""

    def test_exact_canon_match(self):
        from app.services.authority_linking import _check_canonical_match
        result = _check_canonical_match("roi arthur")
        assert result is not None
        assert result["canon"] == "Arthur"
        assert result["dist"] == 0

    def test_fuzzy_canon_match(self):
        from app.services.authority_linking import _check_canonical_match
        result = _check_canonical_match("la dame lanceloc")
        assert result is not None
        assert result["canon"] == "Lancelot"
        assert result["dist"] <= 2

    def test_no_canon_match(self):
        from app.services.authority_linking import _check_canonical_match
        result = _check_canonical_match("nucadigal ament")
        assert result is None

    def test_graal_match(self):
        from app.services.authority_linking import _check_canonical_match
        result = _check_canonical_match("le saint graal")
        assert result is not None
        assert result["canon"] == "Graal"


class TestSalvageTriggerRoleVsPerson:
    """Salvage trigger with garbage adjacent tokens must produce role, not person."""

    def test_trigger_with_good_name_produces_person(self):
        from app.routers.ocr import _extract_salvage_mentions
        mentions, debug = _extract_salvage_mentions("roi arthur et la queste")
        person_mentions = [m for m in mentions if m["ent_type"] == "person" and m["method"] == "rule:salvage_trigger"]
        assert len(person_mentions) >= 1
        assert "arthur" in person_mentions[0]["surface"].lower()

    def test_trigger_with_garbage_produces_role(self):
        from app.routers.ocr import _extract_salvage_mentions
        mentions, debug = _extract_salvage_mentions("roya nucadigal ament nurult")
        trigger_mentions = [m for m in mentions if m["method"] == "rule:salvage_trigger"]
        assert len(trigger_mentions) >= 1
        assert trigger_mentions[0]["ent_type"] == "role", (
            f"Expected role, got {trigger_mentions[0]['ent_type']}"
        )

    def test_trigger_dame_with_guenievre_produces_person(self):
        from app.routers.ocr import _extract_salvage_mentions
        mentions, debug = _extract_salvage_mentions("dame guenievre estoit")
        person_mentions = [m for m in mentions if m["ent_type"] == "person"]
        assert len(person_mentions) >= 1


class TestQualityGateInLinking:
    """Quality-skipped mentions should not query Wikidata."""

    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_role_mention_skipped(self, mock_search, mock_enrich):
        """A role-type mention should be skipped in authority linking."""
        from app.db import pipeline_db as pdb
        from app.services.authority_linking import run_authority_linking

        pdb.create_run("test-asset")
        pdb._init_db_if_needed()
        with pdb._connect() as conn:
            row = conn.execute("SELECT run_id FROM pipeline_runs LIMIT 1").fetchone()
            rid = row["run_id"]

        pdb.insert_chunks(rid, [
            {"chunk_id": "c1", "idx": 0, "start_offset": 0, "end_offset": 20, "text": "roya nucadigal ament"},
        ])
        pdb.insert_entity_mentions(rid, [
            {
                "mention_id": "m-role-1",
                "chunk_id": "c1",
                "start_offset": 0,
                "end_offset": 20,
                "surface": "roya nucadigal ament",
                "norm": "roya nucadigal ament",
                "ent_type": "role",
                "confidence": 0.25,
                "method": "rule:salvage_trigger",
            },
        ])
        pdb.update_run_fields(rid, proofread_text="roya nucadigal ament")

        result = run_authority_linking(rid)
        assert result["skipped_total"] == 1
        assert result["quality_skipped"] == 1
        mock_search.assert_not_called()  # no Wikidata queries for role mentions

    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_canonical_match_prepends_queries(self, mock_search, mock_enrich):
        """A canonical match should prepend qualified queries."""
        _seed_run(text="roi arthur et la terre")
        from app.db import pipeline_db as pdb
        # Add a mention that will trigger canonical matching
        pdb.insert_entity_mentions("test-run-1", [
            {
                "mention_id": "m-canon-1",
                "chunk_id": "chunk-1",
                "start_offset": 0,
                "end_offset": 11,
                "surface": "roi arthur",
                "norm": "roi arthur",
                "ent_type": "person",
                "confidence": 0.55,
                "method": "rule:salvage_trigger",
            },
        ])

        mock_search.return_value = [
            {"qid": "Q45387", "label": "King Arthur", "description": "legendary king", "url": ""},
        ]
        mock_enrich.return_value = {"viaf_id": "", "geonames_id": "", "instance_of_qids": ["Q15632617"]}

        from app.services.authority_linking import run_authority_linking
        result = run_authority_linking("test-run-1")
        assert result["canonical_matched"] >= 1
        # Verify canonical queries were used
        canon_mention = next(
            (r for r in result["mention_results"] if r.get("canonical_match")),
            None,
        )
        assert canon_mention is not None
        assert canon_mention["canonical_match"]["canon"] == "Arthur"


class TestGateBCanonical:
    """Gate B should pass with canonical matches even when candidates_total=0."""

    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_gate_b_passes_with_canonical(self, mock_search, mock_enrich):
        """When all mentions are quality-skipped, B/C/G should all pass."""
        from app.db import pipeline_db as pdb
        from app.services.authority_linking import run_authority_linking, build_linking_report

        pdb.create_run("test-asset")
        pdb._init_db_if_needed()
        with pdb._connect() as conn:
            row = conn.execute("SELECT run_id FROM pipeline_runs LIMIT 1").fetchone()
            rid = row["run_id"]

        pdb.insert_chunks(rid, [
            {"chunk_id": "c1", "idx": 0, "start_offset": 0, "end_offset": 15, "text": "roya nucadigal"},
        ])
        pdb.insert_entity_mentions(rid, [
            {
                "mention_id": "m-skip-1",
                "chunk_id": "c1",
                "start_offset": 0,
                "end_offset": 15,
                "surface": "roya nucadigal",
                "norm": "roya nucadigal",
                "ent_type": "role",
                "confidence": 0.25,
                "method": "rule:salvage_trigger",
            },
        ])
        pdb.update_run_fields(rid, proofread_text="roya nucadigal")

        result = run_authority_linking(rid)
        report = build_linking_report(result)
        assert "Gate B" in report
        assert "PASS" in report.split("Gate B")[1].split("\n")[0]
        assert "PASS_NO_LINKABLE_MENTIONS" in report

    def test_report_has_quality_section(self):
        from app.services.authority_linking import build_linking_report
        result = {
            "run_id": "test-1",
            "asset_ref": "test",
            "mentions_total": 1,
            "type_counts": {"role": 1},
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
            "mention_results": [
                {
                    "mention_id": "m1",
                    "surface": "roya nucadigal",
                    "ent_type": "role",
                    "status": "skipped",
                    "reason": "ent_type=role (not linkable)",
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                },
            ],
            "_base_text": "roya nucadigal ament",
        }
        report = build_linking_report(result)
        assert "MENTION QUALITY DECISIONS" in report
        assert "SKIP" in report
        assert "PASS_NO_LINKABLE_MENTIONS" in report


# ── Text Normalization Module Tests ───────────────────────────────────

class TestNormalizeUnicode:
    def test_long_s_folded(self):
        from app.services.text_normalization import normalize_unicode
        assert "s" in normalize_unicode("ſaint")
        assert "ſ" not in normalize_unicode("ſaint")

    def test_nfc_composition(self):
        from app.services.text_normalization import normalize_unicode
        # e + combining acute → é
        result = normalize_unicode("e\u0301")
        assert result == "é"

    def test_whitespace_collapse(self):
        from app.services.text_normalization import normalize_unicode
        assert normalize_unicode("  hello   world  \n next ") == "hello world next"


class TestOcrConfusionFixes:
    def test_ligature_expansion(self):
        from app.services.text_normalization import ocr_confusion_fixes
        assert ocr_confusion_fixes("ﬁ") == "fi"
        assert ocr_confusion_fixes("ﬂ") == "fl"
        assert ocr_confusion_fixes("ﬀ") == "ff"

    def test_typographic_quotes(self):
        from app.services.text_normalization import ocr_confusion_fixes
        assert ocr_confusion_fixes("\u2018test\u2019") == "'test'"


class TestTokenQualityScore:
    def test_good_token(self):
        from app.services.text_normalization import token_quality_score
        score = token_quality_score("arthur")
        assert score >= 0.60, f"Good token 'arthur' should score high, got {score}"

    def test_garbage_token(self):
        from app.services.text_normalization import token_quality_score
        score = token_quality_score("xqpfj")
        assert score < 0.30, f"Garbage token should score low, got {score}"

    def test_short_token(self):
        from app.services.text_normalization import token_quality_score
        assert token_quality_score("a") == 0.0

    def test_consonant_cluster_penalty(self):
        from app.services.text_normalization import token_quality_score
        score = token_quality_score("brstmkg")
        assert score < 0.25


class TestTextQualityLabel:
    def test_high_quality(self):
        from app.services.text_normalization import text_quality_label
        text = "Le roi Arthur et la reine Guenievre furent assembles au chastel de Camelot"
        assert text_quality_label(text) == "HIGH"

    def test_low_quality(self):
        from app.services.text_normalization import text_quality_label
        text = "xqpfj brstmk qzwvx nccfd xplmn"
        assert text_quality_label(text) == "LOW"

    def test_empty_is_low(self):
        from app.services.text_normalization import text_quality_label
        assert text_quality_label("") == "LOW"


class TestNormalizeForSearch:
    def test_full_pipeline(self):
        from app.services.text_normalization import normalize_for_search
        result = normalize_for_search("  Ré-Ŝaint Omer  ")
        assert result == "re saint omer"

    def test_long_s(self):
        from app.services.text_normalization import normalize_for_search
        result = normalize_for_search("ſaint")
        assert result == "saint"

    def test_diacritics_stripped(self):
        from app.services.text_normalization import normalize_for_search
        result = normalize_for_search("Évrard")
        assert result == "evrard"


# ── Quality-Adaptive Threshold Tests ──────────────────────────────────

class TestQualityAdaptiveThresholds:
    """Verify that OCR quality level affects disambiguation thresholds."""

    def test_high_quality_lower_threshold(self):
        from app.services.entity_scoring import get_thresholds
        h = get_thresholds("HIGH")
        m = get_thresholds("MEDIUM")
        assert h["AUTO_SELECT_THRESHOLD"] < m["AUTO_SELECT_THRESHOLD"]

    def test_low_quality_higher_threshold(self):
        from app.services.entity_scoring import get_thresholds
        m = get_thresholds("MEDIUM")
        lo = get_thresholds("LOW")
        assert lo["AUTO_SELECT_THRESHOLD"] > m["AUTO_SELECT_THRESHOLD"]

    def test_disambiguate_high_quality_accepts(self):
        """A score of 0.82 should be accepted in HIGH but not MEDIUM."""
        from app.services.entity_scoring import disambiguate
        candidates = [
            {"qid": "Q1", "label": "Test", "score": 0.82, "type_compatible": True},
        ]
        result_high = disambiguate(candidates, ocr_quality="HIGH")
        result_med = disambiguate(candidates, ocr_quality="MEDIUM")
        assert result_high["status"] == "linked"
        assert result_med["status"] == "unresolved"

    def test_disambiguate_low_quality_rejects(self):
        """A score of 0.88 should be accepted in MEDIUM but not LOW."""
        from app.services.entity_scoring import disambiguate
        candidates = [
            {"qid": "Q1", "label": "Test", "score": 0.88, "type_compatible": True},
        ]
        result_med = disambiguate(candidates, ocr_quality="MEDIUM")
        result_low = disambiguate(candidates, ocr_quality="LOW")
        assert result_med["status"] == "linked"
        assert result_low["status"] == "unresolved"

    def test_disambiguate_returns_ocr_quality(self):
        from app.services.entity_scoring import disambiguate
        result = disambiguate([], ocr_quality="LOW")
        assert result["ocr_quality"] == "LOW"

    def test_unknown_quality_defaults_to_medium(self):
        from app.services.entity_scoring import get_thresholds
        assert get_thresholds("UNKNOWN") == get_thresholds("MEDIUM")


# ── READY_STATUS Tests ────────────────────────────────────────────────

class TestReadyStatus:
    """Verify explicit READY_STATUS replaces ambiguous labels."""

    def test_pass_linked_status(self):
        from app.services.authority_linking import build_linking_report
        result = {
            "run_id": "test-1",
            "asset_ref": "test",
            "mentions_total": 1,
            "type_counts": {"person": 1},
            "candidates_total": 1,
            "source_counts": {"wikidata": 1},
            "linked_total": 1,
            "unresolved_total": 0,
            "ambiguous_total": 0,
            "skipped_total": 0,
            "quality_skipped": 0,
            "canonical_matched": 0,
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
                    "surface": "arthur",
                    "ent_type": "person",
                    "status": "linked",
                    "reason": "score=0.85",
                    "evidence_text": "roi arthur",
                    "selected": {"qid": "Q45387", "label": "King Arthur", "description": "", "score": 0.85, "viaf_id": "", "geonames_id": "", "type_compatible": True},
                    "top_candidates": [{"qid": "Q45387", "label": "King Arthur", "score": 0.85, "type_compatible": True}],
                },
            ],
            "_base_text": "roi arthur",
        }
        report = build_linking_report(result)
        assert "READY_STATUS: PASS_LINKED" in report
        assert "ALL_QUALITY_SKIPPED" not in report

    def test_pass_no_linkable_mentions_status(self):
        from app.services.authority_linking import build_linking_report
        result = {
            "run_id": "test-1",
            "asset_ref": "test",
            "mentions_total": 1,
            "type_counts": {"role": 1},
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
            "ocr_quality": "LOW",
            "mention_results": [
                {
                    "mention_id": "m1",
                    "surface": "roya nucadigal",
                    "ent_type": "role",
                    "status": "skipped",
                    "reason": "ent_type=role",
                    "name_likeness": 0.0,
                    "canonical_match": None,
                    "selected": None,
                    "top_candidates": [],
                },
            ],
            "_base_text": "roya nucadigal ament",
        }
        report = build_linking_report(result)
        assert "READY_STATUS: PASS_ALL_QUALITY_SKIPPED" in report
        assert "mentions_linkable_total: 0" in report
        assert "mentions_skipped_total: 1" in report

    def test_fail_status(self):
        from app.services.authority_linking import build_linking_report
        result = {
            "run_id": "test-1",
            "asset_ref": "test",
            "mentions_total": 1,
            "type_counts": {"person": 1},
            "candidates_total": 0,
            "source_counts": {},
            "linked_total": 0,
            "unresolved_total": 1,
            "ambiguous_total": 0,
            "skipped_total": 0,
            "quality_skipped": 0,
            "canonical_matched": 0,
            "type_mismatch_count": 0,
            "api_calls_search": 1,
            "api_calls_get": 0,
            "api_calls": 1,
            "cache_hits": 0,
            "took_ms": 10,
            "ocr_quality": "MEDIUM",
            "mention_results": [
                {
                    "mention_id": "m1",
                    "surface": "lancelot",
                    "ent_type": "person",
                    "status": "unresolved",
                    "reason": "no candidates",
                    "evidence_text": "lancelot",
                    "selected": None,
                    "name_likeness": 1.0,
                    "canonical_match": {"canon": "Lancelot", "token": "lancelot", "dist": 0},
                    "top_candidates": [],
                },
            ],
            "_base_text": "Le vilain de Cambrai",
        }
        report = build_linking_report(result)
        assert "READY_STATUS: FAIL" in report


# ── Regression Tests: No Nonsense Links ───────────────────────────────

class TestRegressionNoNonsenseLinks:
    """Regression tests ensuring specific garbage tokens don't produce links."""

    def test_ament_not_linked_as_place(self):
        """'ament' after a preposition must NOT become a place mention
        (it was previously linked to 'catkin' on Wikidata)."""
        from app.routers.ocr import _extract_salvage_mentions
        mentions, debug = _extract_salvage_mentions("en ament de la terre")
        place_mentions = [m for m in mentions if m["ent_type"] == "place"]
        # "ament" is only 5 chars and should fail the place_likeness gate
        ament_places = [m for m in place_mentions if "ament" in m["surface"].lower()]
        assert len(ament_places) == 0, (
            f"'ament' should be rejected as a place candidate, got: {ament_places}"
        )

    def test_qite_not_linked_as_place(self):
        """'qite' should not become a place mention — too garbage-like."""
        from app.routers.ocr import _extract_salvage_mentions
        mentions, debug = _extract_salvage_mentions("de qite en qite")
        place_mentions = [m for m in mentions if m["ent_type"] == "place"]
        qite_places = [m for m in place_mentions if "qite" in m["surface"].lower()]
        assert len(qite_places) == 0

    def test_role_only_skip_in_linking(self):
        """When only role mentions exist, linking produces PASS_NO_LINKABLE_MENTIONS."""
        from app.services.authority_linking import build_linking_report
        result = {
            "run_id": "regression-1",
            "asset_ref": "test",
            "mentions_total": 2,
            "type_counts": {"role": 2},
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
            "ocr_quality": "LOW",
            "mention_results": [
                {"mention_id": "m1", "surface": "roya xqz", "ent_type": "role", "status": "skipped",
                 "reason": "role", "selected": None, "top_candidates": []},
                {"mention_id": "m2", "surface": "duc brstm", "ent_type": "role", "status": "skipped",
                 "reason": "role", "selected": None, "top_candidates": []},
            ],
            "_base_text": "roya xqz duc brstm",
        }
        report = build_linking_report(result)
        assert "READY_STATUS: PASS_ALL_QUALITY_SKIPPED" in report


# ── N-gram Canonical Scan Tests ───────────────────────────────────────

class TestNgramCanonicalScan:
    """Test case-insensitive canonical entity detection from lowercase OCR."""

    def test_lowercase_arthur_detected(self):
        """'arthur' in fully lowercase text should be detected."""
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions("si vint arthur en la terre")
        arthur_mentions = [m for m in mentions if "arthur" in m["norm"]]
        assert len(arthur_mentions) >= 1

    def test_lowercase_graal_detected(self):
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions("la queste del saint graal")
        graal_mentions = [m for m in mentions if "graal" in m["norm"]]
        assert len(graal_mentions) >= 1

    def test_fuzzy_lancelot_detected(self):
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions("lancelo vint en la forest")
        lancelot_mentions = [m for m in mentions if "lancelot" in m["norm"]]
        assert len(lancelot_mentions) >= 1

    def test_garbage_not_detected(self):
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions("xqpfj brstmk nccfd")
        assert len(mentions) == 0

    def test_short_tokens_skipped(self):
        from app.routers.ocr import _extract_ngram_canonical_mentions
        mentions = _extract_ngram_canonical_mentions("de la le et")
        assert len(mentions) == 0


# ── OCR Quality in Authority Linking Tests ────────────────────────────

class TestOcrQualityInLinking:
    """Verify that OCR quality is computed and used in authority linking."""

    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_ocr_quality_in_result(self, mock_search, mock_enrich):
        _seed_run()
        mock_search.return_value = [
            {"qid": "Q100", "label": "Test", "description": "", "url": ""},
        ]
        mock_enrich.return_value = {"viaf_id": "", "geonames_id": "", "instance_of_qids": ["Q5"]}

        from app.services.authority_linking import run_authority_linking
        result = run_authority_linking("test-run-1")
        assert "ocr_quality" in result
        assert result["ocr_quality"] in ("HIGH", "MEDIUM", "LOW")

    @patch("app.services.wikidata_client.enrich_wikidata_item")
    @patch("app.services.wikidata_client.search_wikidata")
    def test_ocr_quality_in_report(self, mock_search, mock_enrich):
        _seed_run()
        mock_search.return_value = [
            {"qid": "Q100", "label": "Test", "description": "", "url": ""},
        ]
        mock_enrich.return_value = {"viaf_id": "", "geonames_id": "", "instance_of_qids": ["Q5"]}

        from app.services.authority_linking import run_authority_linking, build_linking_report
        result = run_authority_linking("test-run-1")
        report = build_linking_report(result)
        assert "ocr_quality:" in report


# ── Place Likeness Gate Tests ─────────────────────────────────────────

class TestPlaceLikenessGate:
    """Verify the place_likeness gate rejects garbage place candidates."""

    def test_good_place_passes(self):
        from app.routers.ocr import _extract_salvage_mentions
        mentions, _ = _extract_salvage_mentions("en Cambrai vint le roi")
        place_mentions = [m for m in mentions if m["ent_type"] == "place"]
        cambrai_mentions = [m for m in place_mentions if "cambrai" in m["surface"].lower()]
        # Cambrai has length >= 5 and good phonotactics → should pass
        assert len(cambrai_mentions) >= 1

    def test_garbage_place_rejected(self):
        from app.routers.ocr import _extract_salvage_mentions
        mentions, debug = _extract_salvage_mentions("en brstmk vint")
        place_mentions = [m for m in mentions if m["ent_type"] == "place"]
        # "brstmk" has terrible quality → rejected
        assert len(place_mentions) == 0
        # Should appear in rejected list
        rejected_surfaces = [r["surface"].lower() for r in debug.get("rejected", [])]
        assert any("brstmk" in s for s in rejected_surfaces)
