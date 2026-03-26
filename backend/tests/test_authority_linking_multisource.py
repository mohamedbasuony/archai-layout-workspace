from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.db import pipeline_db  # type: ignore[import-untyped]
from app.services import authority_linking  # type: ignore[import-untyped]


def _seed_run(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    surface: str,
    ent_type: str,
    base_text: str,
) -> str:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    run_id = pipeline_db.create_run(asset_ref="page-1", asset_sha256="sha")
    with pipeline_db._connect() as conn:
        conn.execute(
            "UPDATE pipeline_runs SET ocr_text=?, proofread_text=?, warnings_json=? WHERE run_id=?",
            (base_text, base_text, json.dumps({"quality_label": "HIGH"}), run_id),
        )
        conn.commit()
    pipeline_db.insert_chunks(
        run_id,
        [
            {
                "chunk_id": "chunk-1",
                "idx": 0,
                "start_offset": 0,
                "end_offset": len(base_text),
                "text": base_text,
            }
        ],
    )
    pipeline_db.insert_entity_mentions(
        run_id,
        [
            {
                "mention_id": "mention-1",
                "chunk_id": "chunk-1",
                "start_offset": 0,
                "end_offset": len(surface),
                "surface": surface,
                "norm": surface.lower(),
                "label": ent_type.upper(),
                "ent_type": ent_type,
                "confidence": 0.92,
                "method": "test",
            }
        ],
    )
    return run_id


def test_run_authority_linking_queries_wikidata_and_viaf(monkeypatch: Any, tmp_path: Path) -> None:
    run_id = _seed_run(
        tmp_path,
        monkeypatch,
        surface="Merlin",
        ent_type="person",
        base_text="Merlin medieval legendary wizard arthurian Merlinus",
    )

    monkeypatch.setattr(authority_linking, "cache_check", lambda *args, **kwargs: (None, "miss"))
    monkeypatch.setattr(
        authority_linking,
        "search_wikidata",
        lambda *args, **kwargs: [
            {
                "qid": "Q111",
                "label": "Merlin",
                "description": "medieval legendary wizard arthurian",
                "url": "https://www.wikidata.org/wiki/Q111",
            }
        ],
    )
    monkeypatch.setattr(
        authority_linking,
        "search_viaf",
        lambda *args, **kwargs: [
            {
                "source": "viaf",
                "authority_id": "12345",
                "qid": "",
                "viaf_id": "12345",
                "geonames_id": "",
                "label": "Myrddin Wyllt",
                "description": "legendary wizard authority record",
                "url": "https://viaf.org/viaf/12345",
                "aliases": [{"value": "Merlinus", "lang": "la"}],
                "source_confidence": 0.92,
            }
        ],
    )
    monkeypatch.setattr(authority_linking, "search_geonames", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        authority_linking,
        "enrich_wikidata_item",
        lambda qid: {
            "instance_of_qids": ["Q5"],
            "viaf_id": "12345",
            "canonical_label": "Merlin",
            "description": "medieval legendary wizard arthurian",
            "aliases": [{"value": "Merlinus", "lang": "la"}],
            "country_qids": [],
            "admin_qids": [],
        },
    )
    monkeypatch.setattr(authority_linking, "is_type_compatible", lambda *args, **kwargs: True)
    monkeypatch.setattr(authority_linking, "_fetch_viaf_profile", lambda viaf_id: {"aliases": [], "titles": []})

    result = authority_linking.run_authority_linking(run_id, top_k=3)

    assert result["linked_total"] == 1
    assert result["source_counts"]["wikidata"] >= 1
    assert result["source_counts"]["viaf"] >= 1
    assert result["api_calls_viaf"] >= 1

    rows = pipeline_db.list_mention_links_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["entity_id"] == "wikidata:Q111"
    assert rows[0]["wikidata_qid"] == "Q111"
    assert rows[0]["viaf_id"] == "12345"


def test_run_authority_linking_can_select_geonames_candidate(monkeypatch: Any, tmp_path: Path) -> None:
    run_id = _seed_run(
        tmp_path,
        monkeypatch,
        surface="Paris",
        ent_type="place",
        base_text="Paris medieval abbey monastery diocese city France Paris",
    )

    monkeypatch.setattr(authority_linking, "cache_check", lambda *args, **kwargs: (None, "miss"))
    monkeypatch.setattr(authority_linking, "search_wikidata", lambda *args, **kwargs: [])
    monkeypatch.setattr(authority_linking, "search_viaf", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        authority_linking,
        "search_geonames",
        lambda *args, **kwargs: [
            {
                "source": "geonames",
                "authority_id": "2988507",
                "qid": "",
                "viaf_id": "",
                "geonames_id": "2988507",
                "label": "Paris",
                "description": "medieval abbey monastery diocese city France",
                "url": "https://www.geonames.org/2988507",
                "aliases": [{"value": "Paris", "lang": ""}],
                "canonical_label": "Paris",
                "canonical_description": "medieval abbey monastery diocese city France",
                "country_name": "France",
                "admin1_name": "Ile-de-France",
                "parent_location": "Ile-de-France > France",
                "source_confidence": 0.95,
            }
        ],
    )
    monkeypatch.setattr(authority_linking, "is_type_compatible", lambda *args, **kwargs: True)

    result = authority_linking.run_authority_linking(run_id, top_k=3)

    assert result["linked_total"] == 1
    assert result["source_counts"]["geonames"] >= 1
    assert result["api_calls_geonames"] >= 1

    rows = pipeline_db.list_mention_links_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["entity_id"] == "geonames:2988507"
    assert rows[0]["geonames_id"] == "2988507"
    assert rows[0]["authority_source"] == "geonames"


def test_run_authority_linking_keeps_weak_lowercase_place_candidate_unresolved(monkeypatch: Any, tmp_path: Path) -> None:
    run_id = _seed_run(
        tmp_path,
        monkeypatch,
        surface="vilanie",
        ent_type="place",
        base_text="La vilanie fu mult grant et dure.",
    )

    wikidata_calls: list[str] = []
    monkeypatch.setattr(authority_linking, "cache_check", lambda *args, **kwargs: (None, "miss"))
    monkeypatch.setattr(authority_linking, "search_wikidata", lambda *args, **kwargs: wikidata_calls.append("wikidata") or [])
    monkeypatch.setattr(authority_linking, "search_viaf", lambda *args, **kwargs: [])
    monkeypatch.setattr(authority_linking, "search_geonames", lambda *args, **kwargs: [])

    result = authority_linking.run_authority_linking(run_id, top_k=3)

    assert result["linked_total"] == 0
    assert result["unresolved_total"] >= 1
    assert wikidata_calls == []
    rows = pipeline_db.list_mention_links_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["entity_id"] is None
    assert rows[0]["link_status"] == "unresolved_low_quality"
    assert "lexical=True" in str(rows[0]["reason"])
    report = authority_linking.build_linking_report_from_db(run_id)
    assert "insufficient evidence to treat this lexical item as a place" in report
    assert "low_evidence_place:" not in report


def test_run_authority_linking_keeps_enfant_out_of_place_linking(monkeypatch: Any, tmp_path: Path) -> None:
    run_id = _seed_run(
        tmp_path,
        monkeypatch,
        surface="enfant",
        ent_type="place",
        base_text="Li enfant parla au roi.",
    )

    wikidata_calls: list[str] = []
    monkeypatch.setattr(authority_linking, "cache_check", lambda *args, **kwargs: (None, "miss"))
    monkeypatch.setattr(authority_linking, "search_wikidata", lambda *args, **kwargs: wikidata_calls.append("wikidata") or [])
    monkeypatch.setattr(authority_linking, "search_viaf", lambda *args, **kwargs: [])
    monkeypatch.setattr(authority_linking, "search_geonames", lambda *args, **kwargs: [])

    result = authority_linking.run_authority_linking(run_id, top_k=3)

    assert result["linked_total"] == 0
    assert wikidata_calls == []
    rows = pipeline_db.list_mention_links_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["entity_id"] is None
    assert rows[0]["link_status"] == "unresolved_low_quality"
    assert "lexical=True" in str(rows[0]["reason"])


def test_run_authority_linking_keeps_capitalized_enfant_out_of_place_linking(monkeypatch: Any, tmp_path: Path) -> None:
    run_id = _seed_run(
        tmp_path,
        monkeypatch,
        surface="Enfant",
        ent_type="place",
        base_text="Enfant parla devant le roi.",
    )

    wikidata_calls: list[str] = []
    monkeypatch.setattr(authority_linking, "cache_check", lambda *args, **kwargs: (None, "miss"))
    monkeypatch.setattr(authority_linking, "search_wikidata", lambda *args, **kwargs: wikidata_calls.append("wikidata") or [])
    monkeypatch.setattr(authority_linking, "search_viaf", lambda *args, **kwargs: [])
    monkeypatch.setattr(authority_linking, "search_geonames", lambda *args, **kwargs: [])

    result = authority_linking.run_authority_linking(run_id, top_k=3)

    assert result["linked_total"] == 0
    assert wikidata_calls == []
    rows = pipeline_db.list_mention_links_for_run(run_id)
    assert len(rows) == 1
    assert rows[0]["entity_id"] is None
    assert rows[0]["link_status"] == "unresolved_low_quality"
    assert "lexical=True" in str(rows[0]["reason"])


def test_persist_unresolved_mentions_downgrades_weak_place_report(monkeypatch: Any, tmp_path: Path) -> None:
    run_id = _seed_run(
        tmp_path,
        monkeypatch,
        surface="vilanie",
        ent_type="place",
        base_text="La vilanie fu mult grant et dure.",
    )

    result = authority_linking.persist_unresolved_mentions(
        run_id,
        reason="token_search_allowed=False quality=RISKY",
    )

    report = authority_linking.build_linking_report(result)
    assert "vilanie [lexical/unknown]" in report
    assert "insufficient evidence to treat this lexical item as a place" in report
