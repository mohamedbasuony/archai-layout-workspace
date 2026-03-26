from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_backend_src = Path(__file__).resolve().parent.parent / "app"
if str(_backend_src.parent) not in sys.path:
    sys.path.insert(0, str(_backend_src.parent))

from app.db import pipeline_db  # type: ignore[import-untyped]
from app.services import rag_store  # type: ignore[import-untyped]


class _FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def upsert(self, *, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]], embeddings: Any = None) -> None:
        _ = embeddings
        for idx, doc_id in enumerate(ids):
            self.records[doc_id] = {"document": documents[idx], "metadata": metadatas[idx]}

    def count(self) -> int:
        return len(self.records)

    def get(self, where: dict[str, Any] | None = None, include: list[str] | None = None) -> dict[str, Any]:
        _ = include
        ids = []
        for doc_id, record in self.records.items():
            metadata = record["metadata"]
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            ids.append(doc_id)
        return {"ids": ids}

    def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self.records.pop(doc_id, None)

    def query(
        self,
        *,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int,
        where: dict[str, Any] | None = None,
        include: list[str],
    ) -> dict[str, Any]:
        _ = query_embeddings, include
        query = (query_texts or [""])[0].lower()
        hits = []
        for doc_id, record in self.records.items():
            metadata = record["metadata"]
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            text = str(record["document"] or "")
            score = 0.0 if query and query in text.lower() else 0.9
            hits.append((score, doc_id, text, metadata))
        hits.sort(key=lambda item: item[0])
        chosen = hits[:n_results]
        return {
            "ids": [[item[1] for item in chosen]],
            "documents": [[item[2] for item in chosen]],
            "metadatas": [[item[3] for item in chosen]],
            "distances": [[item[0] for item in chosen]],
        }


def _backend_collection_factory() -> tuple[dict[str, _FakeCollection], Any]:
    collections: dict[str, _FakeCollection] = {}

    def _get_collection(client: Any = None, *, backend_key: str | None = None) -> _FakeCollection:
        _ = client
        key = backend_key or rag_store._LOCAL_EMBED_BACKEND
        collections.setdefault(key, _FakeCollection())
        return collections[key]

    return collections, _get_collection


def test_index_run_builds_entity_native_index(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(rag_store, "_provider_embed", lambda texts: (None, rag_store._LOCAL_EMBED_BACKEND))

    chunk_collection = _FakeCollection()
    entity_collection = _FakeCollection()
    monkeypatch.setattr(rag_store, "_collection", lambda client=None, *, backend_key=None: chunk_collection)
    monkeypatch.setattr(rag_store, "_entity_collection", lambda client=None, *, backend_key=None: entity_collection)

    run_id = pipeline_db.create_run(asset_ref="page-1", asset_sha256="sha")
    pipeline_db.insert_chunks(
        run_id,
        [
            {
                "chunk_id": "chunk-1",
                "idx": 0,
                "start_offset": 0,
                "end_offset": 32,
                "text": "Arthur appears in the manuscript.",
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
                "end_offset": 6,
                "surface": "Arthur",
                "norm": "arthur",
                "label": "PERSON",
                "ent_type": "person",
                "confidence": 0.91,
                "method": "test",
            }
        ],
    )
    pipeline_db.upsert_authority_entity(
        {
            "entity_id": "wikidata:Q45720",
            "authority_source": "wikidata",
            "authority_id": "Q45720",
            "wikidata_qid": "Q45720",
            "viaf_id": "24604281",
            "canonical_label": "King Arthur",
            "normalized_label": "king arthur",
            "entity_type": "person",
            "description": "legendary king of Britain",
        }
    )
    pipeline_db.replace_authority_aliases(
        "wikidata:Q45720",
        [
            {"alias": "King Arthur", "alias_source": "wikidata"},
            {"alias": "Arthur", "alias_source": "mention_surface"},
        ],
    )
    pipeline_db.replace_authority_source_assertions(
        "wikidata:Q45720",
        [
            {"source_name": "wikidata", "property_name": "wikidata_qid", "property_value": "Q45720"},
            {"source_name": "viaf", "property_name": "viaf_id", "property_value": "24604281"},
        ],
    )
    span = pipeline_db.upsert_evidence_span(
        {
            "run_id": run_id,
            "asset_ref": "page-1",
            "page_id": "page-1",
            "chunk_id": "chunk-1",
            "mention_id": "mention-1",
            "raw_text": "Arthur",
            "normalized_text": "arthur",
            "start_offset": 0,
            "end_offset": 6,
        }
    )
    pipeline_db.upsert_mention_link(
        {
            "mention_id": "mention-1",
            "entity_id": "wikidata:Q45720",
            "confidence": 0.94,
            "link_status": "linked",
            "selected_by": "test",
            "type_compatible": True,
            "evidence_span_id": span["span_id"],
            "reason": "linked in test",
        }
    )

    result = rag_store.index_run(run_id)
    entity_hits = rag_store.retrieve_entities("Arthur", run_ids=[run_id])

    assert result["chunks_indexed"] == 1
    assert result["entities_indexed"] == 1
    assert any(doc_id.startswith(f"entity:{run_id}:wikidata:Q45720") for doc_id in entity_collection.records)
    assert entity_hits
    assert entity_hits[0]["entity_id"] == "wikidata:Q45720"


def test_retrieve_chunks_auto_indexes_run_into_provider_specific_collection(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "archai.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(
        rag_store,
        "_provider_embed",
        lambda texts: ([[0.1, 0.2, 0.3, 0.4] for _ in texts], "provider_1024"),
    )

    chunk_collections, chunk_factory = _backend_collection_factory()
    entity_collections, entity_factory = _backend_collection_factory()
    monkeypatch.setattr(rag_store, "_collection", chunk_factory)
    monkeypatch.setattr(rag_store, "_entity_collection", entity_factory)

    run_id = pipeline_db.create_run(asset_ref="page-2", asset_sha256="sha-2")
    pipeline_db.insert_chunks(
        run_id,
        [
            {
                "chunk_id": "chunk-2",
                "idx": 0,
                "start_offset": 0,
                "end_offset": 28,
                "text": "Merlin guides Arthur onward.",
            }
        ],
    )

    hits = rag_store.retrieve_chunks("Arthur", run_ids=[run_id])

    assert hits
    assert "provider_1024" in chunk_collections
    assert chunk_collections["provider_1024"].count() == 1
    assert rag_store._LOCAL_EMBED_BACKEND not in chunk_collections or chunk_collections[rag_store._LOCAL_EMBED_BACKEND].count() == 0
    assert entity_collections == {}
