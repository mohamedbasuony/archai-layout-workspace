"""Retrieval-Augmented Generation store backed by ChromaDB.

Responsibilities
----------------
* Manage a persistent ChromaDB collection of OCR text chunks.
* Embed texts via the GWDG provider endpoint (OpenAI-compatible
  ``/v1/embeddings``) or fall back to ChromaDB's built-in default
  embedding function.
* Index a completed pipeline run (read chunks from SQLite → upsert
  into Chroma with full provenance metadata).
* Retrieve top-k chunks for a user query and format them as citation
  evidence blocks for the chat LLM.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.db import pipeline_db
from app.services.rag_debug import log_index_done, log_index_status, log_retrieve_debug

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CLIENT: chromadb.ClientAPI | None = None
_LOCAL_EMBED_BACKEND = "local_default"


# ── ChromaDB client singleton ──────────────────────────────────────────

def _chroma_client() -> chromadb.ClientAPI:
    """Return (or create) a persistent ChromaDB client."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _LOCK:
        if _CLIENT is not None:
            return _CLIENT
        persist_dir = settings.chroma_persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        log.info("ChromaDB persistent client initialised at %s", persist_dir)
        return _CLIENT


def _embedding_backend_slug() -> str:
    model = str(settings.rag_embedding_model or "").strip().lower()
    if not model:
        return _LOCAL_EMBED_BACKEND
    slug = re.sub(r"[^a-z0-9]+", "_", model).strip("_")
    return slug[:64] or "provider_embedding"


def _collection_name_for_backend(base_name: str, backend_key: str) -> str:
    if backend_key == _LOCAL_EMBED_BACKEND:
        return base_name
    return f"{base_name}__{backend_key}"


def _active_chunk_collection_name(backend_key: str) -> str:
    return _collection_name_for_backend(settings.rag_collection_name, backend_key)


def _active_entity_collection_name(backend_key: str) -> str:
    return _collection_name_for_backend(settings.rag_entity_collection_name, backend_key)


def _existing_collection_names(base_name: str) -> list[str]:
    client = _chroma_client()
    names: list[str] = []
    for item in client.list_collections():
        name = getattr(item, "name", str(item))
        if name == base_name or str(name).startswith(f"{base_name}__"):
            names.append(str(name))
    if base_name not in names:
        names.append(base_name)
    return list(dict.fromkeys(names))


def _get_collection_if_exists(name: str) -> chromadb.Collection | None:
    client = _chroma_client()
    existing_names = {
        str(getattr(item, "name", str(item)))
        for item in client.list_collections()
    }
    if name not in existing_names:
        return None
    return client.get_collection(name)


def _collection(client: chromadb.ClientAPI | None = None, *, backend_key: str | None = None) -> chromadb.Collection:
    """Get or create the chunks collection for the active embedding backend."""
    client = client or _chroma_client()
    return client.get_or_create_collection(
        name=_active_chunk_collection_name(backend_key or _embedding_backend_slug()),
        metadata={"hnsw:space": "cosine"},
    )


def _entity_collection(client: chromadb.ClientAPI | None = None, *, backend_key: str | None = None) -> chromadb.Collection:
    """Get or create the entity-native collection for the active embedding backend."""
    client = client or _chroma_client()
    return client.get_or_create_collection(
        name=_active_entity_collection_name(backend_key or _embedding_backend_slug()),
        metadata={"hnsw:space": "cosine"},
    )


# ── Provider embeddings (optional) ─────────────────────────────────────

def _provider_embed(texts: list[str]) -> tuple[list[list[float]] | None, str]:
    """Try to compute embeddings via the GWDG OpenAI-compatible endpoint.

    Returns ``None`` if the provider is unavailable or the call fails,
    so the caller can fall back to ChromaDB's built-in embeddings.
    """
    model = (settings.rag_embedding_model or "").strip()
    if not model:
        return None, _LOCAL_EMBED_BACKEND  # no provider model configured → use local default

    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        return None, _LOCAL_EMBED_BACKEND

    # Reuse the same key/base-url resolution as chat_ai
    from app.services.chat_ai import _require_api_key, _base_url

    try:
        client = OpenAI(api_key=_require_api_key(), base_url=_base_url())
        response = client.embeddings.create(model=model, input=texts)
        return [list(d.embedding) for d in response.data], _embedding_backend_slug()
    except Exception as exc:  # noqa: BLE001
        log.warning("Provider embedding failed (%s), falling back to local: %s", model, exc)
        return None, _LOCAL_EMBED_BACKEND


# ── Indexing ────────────────────────────────────────────────────────────

def _upsert_collection(
    collection: chromadb.Collection,
    *,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    embeddings: list[list[float]] | None = None,
) -> None:
    if not ids:
        return
    if embeddings is not None:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    else:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


def _collection_has_run(collection: chromadb.Collection, run_id: str) -> bool:
    existing = collection.get(where={"run_id": run_id}, include=[])
    ids = existing["ids"] if existing and existing.get("ids") else []
    return bool(ids)


def _load_authority_aliases(entity_ids: Sequence[str]) -> dict[str, list[str]]:
    entity_ids = [str(entity_id or "").strip() for entity_id in entity_ids if str(entity_id or "").strip()]
    if not entity_ids:
        return {}
    placeholders = ",".join("?" for _ in entity_ids)
    sql = (
        f"SELECT entity_id, alias FROM authority_aliases "
        f"WHERE entity_id IN ({placeholders}) ORDER BY alias ASC"
    )
    out: dict[str, list[str]] = {entity_id: [] for entity_id in entity_ids}
    with pipeline_db._connect() as conn:
        rows = conn.execute(sql, entity_ids).fetchall()
    for row in rows:
        entity_id = str(row["entity_id"] or "")
        alias = str(row["alias"] or "").strip()
        if alias:
            out.setdefault(entity_id, []).append(alias)
    return out


def _load_authority_assertions(entity_ids: Sequence[str]) -> dict[str, list[str]]:
    entity_ids = [str(entity_id or "").strip() for entity_id in entity_ids if str(entity_id or "").strip()]
    if not entity_ids:
        return {}
    placeholders = ",".join("?" for _ in entity_ids)
    sql = (
        "SELECT entity_id, source_name, property_name, property_value "
        f"FROM authority_source_assertions WHERE entity_id IN ({placeholders}) "
        "ORDER BY source_name ASC, property_name ASC"
    )
    out: dict[str, list[str]] = {entity_id: [] for entity_id in entity_ids}
    with pipeline_db._connect() as conn:
        rows = conn.execute(sql, entity_ids).fetchall()
    for row in rows:
        entity_id = str(row["entity_id"] or "")
        source_name = str(row["source_name"] or "").strip()
        property_name = str(row["property_name"] or "").strip()
        property_value = str(row["property_value"] or "").strip()
        fact = " ".join(part for part in (source_name, property_name, property_value) if part)
        if fact:
            out.setdefault(entity_id, []).append(fact)
    return out


def _build_entity_index_payload(run_id: str, run: dict[str, Any]) -> tuple[list[str], list[str], list[dict[str, Any]], int]:
    rows = pipeline_db.list_mention_links_for_run(run_id)
    if not rows:
        return [], [], [], 0

    asset_ref = str(run.get("asset_ref") or "")
    linked_groups: dict[str, list[dict[str, Any]]] = {}
    unresolved_rows: list[dict[str, Any]] = []
    for row in rows:
        entity_id = str(row.get("entity_id") or "").strip()
        status = str(row.get("link_status") or "").strip().lower()
        if entity_id and status == "linked":
            linked_groups.setdefault(entity_id, []).append(row)
        else:
            unresolved_rows.append(row)

    aliases_by_entity = _load_authority_aliases(linked_groups.keys())
    assertions_by_entity = _load_authority_assertions(linked_groups.keys())

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    skipped = 0

    for entity_id, group_rows in linked_groups.items():
        head = group_rows[0]
        aliases = list(dict.fromkeys(aliases_by_entity.get(entity_id, [])))
        surfaces = list(dict.fromkeys(str(row.get("surface") or "").strip() for row in group_rows if str(row.get("surface") or "").strip()))
        evidence = list(
            dict.fromkeys(
                str(row.get("evidence_raw_text") or "").strip()
                for row in group_rows
                if str(row.get("evidence_raw_text") or "").strip()
            )
        )[:4]
        assertions = list(dict.fromkeys(assertions_by_entity.get(entity_id, [])))[:8]
        canonical_label = str(head.get("canonical_label") or head.get("surface") or entity_id).strip()
        authority_bits = [
            f"source: {head.get('authority_source') or ''}",
            f"authority_id: {head.get('authority_id') or ''}",
            f"wikidata_qid: {head.get('wikidata_qid') or ''}",
            f"viaf_id: {head.get('viaf_id') or ''}",
            f"geonames_id: {head.get('geonames_id') or ''}",
        ]
        doc = "\n".join(
            [
                f"entity_id: {entity_id}",
                f"canonical_label: {canonical_label}",
                f"entity_type: {head.get('entity_type') or head.get('ent_type') or ''}",
                f"description: {head.get('description') or ''}",
                f"aliases: {', '.join(aliases[:12])}",
                f"mention_surfaces: {', '.join(surfaces[:12])}",
                f"location: {' > '.join(part for part in [str(head.get('admin1_name') or ''), str(head.get('country_name') or '')] if part)}",
                f"confidence_max: {max(float(row.get('confidence') or 0.0) for row in group_rows):.4f}",
                f"evidence_snippets: {' || '.join(evidence)}",
                f"authority_assertions: {' || '.join(assertions)}",
                *authority_bits,
            ]
        ).strip()
        if not doc:
            skipped += 1
            continue
        ids.append(f"entity:{run_id}:{entity_id}")
        documents.append(doc)
        metadatas.append(
            {
                "record_type": "entity",
                "run_id": run_id,
                "asset_ref": asset_ref,
                "entity_id": entity_id,
                "canonical_label": canonical_label,
                "mention_surface": ", ".join(surfaces[:4]),
                "authority_source": str(head.get("authority_source") or ""),
                "authority_id": str(head.get("authority_id") or ""),
                "wikidata_qid": str(head.get("wikidata_qid") or ""),
                "viaf_id": str(head.get("viaf_id") or ""),
                "geonames_id": str(head.get("geonames_id") or ""),
                "entity_type": str(head.get("entity_type") or head.get("ent_type") or ""),
                "mention_count": len(group_rows),
                "confidence": round(max(float(row.get("confidence") or 0.0) for row in group_rows), 4),
            }
        )

    for row in unresolved_rows:
        mention_id = str(row.get("mention_id") or "").strip()
        if not mention_id:
            skipped += 1
            continue
        surface = str(row.get("surface") or "").strip()
        evidence_text = str(row.get("evidence_raw_text") or "").strip()
        reason = str(row.get("reason") or "").strip()
        doc = "\n".join(
            [
                f"mention_id: {mention_id}",
                f"surface: {surface}",
                f"probable_type: {row.get('ent_type') or ''}",
                f"link_status: {row.get('link_status') or ''}",
                f"reason_unresolved: {reason}",
                f"evidence_text: {evidence_text}",
                f"run_id: {run_id}",
                f"asset_ref: {asset_ref}",
            ]
        ).strip()
        if not doc:
            skipped += 1
            continue
        ids.append(f"unresolved:{run_id}:{mention_id}")
        documents.append(doc)
        metadatas.append(
            {
                "record_type": "unresolved_mention",
                "run_id": run_id,
                "asset_ref": asset_ref,
                "mention_id": mention_id,
                "mention_surface": surface,
                "reason_unresolved": reason,
                "chunk_id": str(row.get("chunk_id") or ""),
                "entity_type": str(row.get("ent_type") or ""),
                "link_status": str(row.get("link_status") or ""),
                "confidence": round(float(row.get("confidence") or 0.0), 4),
                "start_offset": int(row.get("evidence_start_offset") or row.get("start_offset") or 0),
                "end_offset": int(row.get("evidence_end_offset") or row.get("end_offset") or 0),
            }
        )

    return ids, documents, metadatas, skipped


def _index_chunks_for_run(run_id: str, run: dict[str, Any], *, backend_key: str | None = None) -> dict[str, Any]:
    chunks = pipeline_db.list_chunks(run_id)
    chunks_total = len(chunks)
    asset_ref = str(run.get("asset_ref") or "")
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    skipped = 0

    for ch in chunks:
        text = str(ch["text"] or "").strip()
        if not text:
            skipped += 1
            continue
        ids.append(str(ch["chunk_id"]))
        documents.append(text)
        metadatas.append(
            {
                "run_id": run_id,
                "asset_ref": asset_ref,
                "chunk_idx": int(ch["idx"]),
                "start_offset": int(ch["start_offset"]),
                "end_offset": int(ch["end_offset"]),
                "detected_language": str(run.get("detected_language") or ""),
            }
        )

    effective_backend_key = backend_key or _LOCAL_EMBED_BACKEND
    embeddings: list[list[float]] | None = None
    if ids:
        if backend_key is None:
            embeddings, effective_backend_key = _provider_embed(documents)
        elif backend_key != _LOCAL_EMBED_BACKEND:
            embeddings, resolved_backend_key = _provider_embed(documents)
            if embeddings is None or resolved_backend_key != backend_key:
                log.warning(
                    "Skipping chunk indexing for run %s in backend %s because provider embeddings are unavailable",
                    run_id,
                    backend_key,
                )
                return {
                    "run_id": run_id,
                    "asset_ref": asset_ref,
                    "chunks_total": chunks_total,
                    "chunks_indexed": 0,
                    "chunks_skipped": chunks_total,
                    "collection_name": _active_chunk_collection_name(backend_key),
                    "collection_count_after": 0,
                    "status": "embedding_unavailable",
                }
            effective_backend_key = backend_key
    col = _collection(backend_key=effective_backend_key)
    _upsert_collection(col, ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return {
        "run_id": run_id,
        "asset_ref": asset_ref,
        "chunks_total": chunks_total,
        "chunks_indexed": len(ids),
        "chunks_skipped": skipped,
        "collection_name": _active_chunk_collection_name(effective_backend_key),
        "collection_count_after": col.count(),
        "status": "ok" if ids else "empty",
    }


def index_entity_run(run_id: str, *, backend_key: str | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    run = pipeline_db.get_run(run_id)
    if run is None:
        raise ValueError(f"Pipeline run {run_id!r} not found.")

    ids, documents, metadatas, skipped = _build_entity_index_payload(run_id, run)
    effective_backend_key = backend_key or _LOCAL_EMBED_BACKEND
    embeddings: list[list[float]] | None = None
    if ids:
        if backend_key is None:
            embeddings, effective_backend_key = _provider_embed(documents)
        elif backend_key != _LOCAL_EMBED_BACKEND:
            embeddings, resolved_backend_key = _provider_embed(documents)
            if embeddings is None or resolved_backend_key != backend_key:
                log.warning(
                    "Skipping entity indexing for run %s in backend %s because provider embeddings are unavailable",
                    run_id,
                    backend_key,
                )
                return {
                    "run_id": run_id,
                    "asset_ref": str(run.get("asset_ref") or ""),
                    "entities_total": len(ids) + skipped,
                    "entities_indexed": 0,
                    "entities_skipped": len(ids) + skipped,
                    "collection_name": _active_entity_collection_name(backend_key),
                    "collection_count_after": 0,
                    "took_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "status": "embedding_unavailable",
                }
            effective_backend_key = backend_key
    col = _entity_collection(backend_key=effective_backend_key)
    _upsert_collection(col, ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    result = {
        "run_id": run_id,
        "asset_ref": str(run.get("asset_ref") or ""),
        "entities_total": len(ids) + skipped,
        "entities_indexed": len(ids),
        "entities_skipped": skipped,
        "collection_name": _active_entity_collection_name(effective_backend_key),
        "collection_count_after": col.count(),
        "took_ms": round((time.perf_counter() - t0) * 1000, 1),
        "status": "ok" if ids else "empty",
    }
    return result


def index_run(run_id: str) -> dict[str, Any]:
    """Index chunk and entity evidence for a pipeline run."""
    t0 = time.perf_counter()
    run = pipeline_db.get_run(run_id)
    if run is None:
        raise ValueError(f"Pipeline run {run_id!r} not found.")

    chunk_result = _index_chunks_for_run(run_id, run)
    entity_result = index_entity_run(run_id)
    result = {
        **chunk_result,
        "entity_collection_name": entity_result["collection_name"],
        "entities_total": entity_result["entities_total"],
        "entities_indexed": entity_result["entities_indexed"],
        "entities_skipped": entity_result["entities_skipped"],
        "entity_collection_count_after": entity_result["collection_count_after"],
        "entity_index_status": entity_result["status"],
        "took_ms": round((time.perf_counter() - t0) * 1000, 1),
        "status": "ok" if chunk_result["chunks_indexed"] or entity_result["entities_indexed"] else "empty",
    }
    log.info(
        "Indexed run %s with %d chunks and %d entity docs",
        run_id,
        chunk_result["chunks_indexed"],
        entity_result["entities_indexed"],
    )
    log_index_done(result)
    return result


def delete_run(run_id: str) -> int:
    """Remove all chunk and entity vectors belonging to *run_id* from the vector store."""
    deleted_total = 0
    collection_names = _existing_collection_names(settings.rag_collection_name) + _existing_collection_names(settings.rag_entity_collection_name)
    for name in list(dict.fromkeys(collection_names)):
        collection = _get_collection_if_exists(name)
        if collection is None:
            continue
        existing = collection.get(where={"run_id": run_id}, include=[])
        ids = existing["ids"] if existing and existing.get("ids") else []
        if ids:
            collection.delete(ids=ids)
        deleted_total += len(ids)
    log.info("Deleted %d vectors for run %s", deleted_total, run_id)
    return deleted_total


def is_run_indexed(run_id: str) -> dict[str, Any]:
    """Check whether a run has chunk and entity vectors in the stores."""
    indexed_chunk_ids: set[str] = set()
    for name in _existing_collection_names(settings.rag_collection_name):
        chunk_collection = _get_collection_if_exists(name)
        if chunk_collection is None:
            continue
        chunk_indexed_result = chunk_collection.get(where={"run_id": run_id}, include=[])
        if chunk_indexed_result and chunk_indexed_result.get("ids"):
            indexed_chunk_ids.update(str(item) for item in chunk_indexed_result["ids"])

    db_chunks = pipeline_db.list_chunks(run_id)
    all_chunk_ids = {str(ch["chunk_id"]) for ch in db_chunks}
    missing_chunk_ids = sorted(all_chunk_ids - indexed_chunk_ids)

    run = pipeline_db.get_run(run_id)
    asset_ref = str(run.get("asset_ref") or "") if run else ""
    entity_ids, _entity_docs, _entity_meta, _entity_skipped = _build_entity_index_payload(run_id, run or {})
    expected_entity_ids = set(entity_ids)
    indexed_entity_ids: set[str] = set()
    for name in _existing_collection_names(settings.rag_entity_collection_name):
        entity_collection = _get_collection_if_exists(name)
        if entity_collection is None:
            continue
        entity_indexed_result = entity_collection.get(where={"run_id": run_id}, include=[])
        if entity_indexed_result and entity_indexed_result.get("ids"):
            indexed_entity_ids.update(str(item) for item in entity_indexed_result["ids"])

    result: dict[str, Any] = {
        "run_id": run_id,
        "asset_ref": asset_ref,
        "chunks_total": len(all_chunk_ids),
        "chunks_indexed": len(indexed_chunk_ids),
        "missing_chunk_ids": missing_chunk_ids,
        "entities_total": len(expected_entity_ids),
        "entities_indexed": len(indexed_entity_ids),
        "missing_entity_ids": sorted(expected_entity_ids - indexed_entity_ids),
        "indexed": len(indexed_chunk_ids) > 0 or len(indexed_entity_ids) > 0,
        "chunk_indexed": len(indexed_chunk_ids) > 0,
        "entity_indexed": len(indexed_entity_ids) > 0,
    }
    log_index_status(result)
    return result


def _ensure_chunk_runs_indexed(run_ids: Sequence[str] | None, *, backend_key: str) -> None:
    if not run_ids or not settings.rag_auto_index:
        return
    collection = _collection(backend_key=backend_key)
    for run_id in run_ids:
        if _collection_has_run(collection, str(run_id)):
            continue
        run = pipeline_db.get_run(str(run_id))
        if run is None:
            continue
        _index_chunks_for_run(str(run_id), run, backend_key=backend_key)


def _ensure_entity_runs_indexed(run_ids: Sequence[str] | None, *, backend_key: str) -> None:
    if not run_ids or not settings.rag_auto_index:
        return
    collection = _entity_collection(backend_key=backend_key)
    for run_id in run_ids:
        if _collection_has_run(collection, str(run_id)):
            continue
        index_entity_run(str(run_id), backend_key=backend_key)


# ── Retrieval ───────────────────────────────────────────────────────────

def retrieve_chunks(
    query: str,
    *,
    top_k: int | None = None,
    run_ids: Sequence[str] | None = None,
    asset_ref: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the *top_k* most relevant chunks for *query*.

    Parameters
    ----------
    query:
        User question / search string.
    top_k:
        Number of results (default: ``settings.rag_top_k``).
    run_ids:
        Optional filter — only return chunks belonging to these runs.
    asset_ref:
        Optional filter — only return chunks from this asset.

    Returns
    -------
    List of dicts with keys ``chunk_id``, ``run_id``, ``asset_ref``,
    ``chunk_idx``, ``start_offset``, ``end_offset``, ``text``,
    ``distance``.
    """
    k = top_k or settings.rag_top_k
    query_embeddings, backend_key = _provider_embed([query])
    _ensure_chunk_runs_indexed(run_ids, backend_key=backend_key)
    col = _collection(backend_key=backend_key)

    where_clauses: list[dict[str, Any]] = []
    if run_ids:
        id_list = list(run_ids)
        if len(id_list) == 1:
            where_clauses.append({"run_id": id_list[0]})
        else:
            where_clauses.append({"run_id": {"$in": id_list}})
    if asset_ref:
        where_clauses.append({"asset_ref": asset_ref})

    where_filter: dict[str, Any] | None = None
    if len(where_clauses) == 1:
        where_filter = where_clauses[0]
    elif len(where_clauses) > 1:
        where_filter = {"$and": where_clauses}

    if query_embeddings is not None:
        results = col.query(
            query_embeddings=query_embeddings,
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    else:
        results = col.query(
            query_texts=[query],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

    return _flatten_results(results)


def retrieve_entities(
    query: str,
    *,
    top_k: int | None = None,
    run_ids: Sequence[str] | None = None,
    asset_ref: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the *top_k* most relevant entity-native records for *query*."""
    k = top_k or settings.rag_entity_top_k
    query_embeddings, backend_key = _provider_embed([query])
    _ensure_entity_runs_indexed(run_ids, backend_key=backend_key)
    col = _entity_collection(backend_key=backend_key)

    where_clauses: list[dict[str, Any]] = []
    if run_ids:
        id_list = list(run_ids)
        if len(id_list) == 1:
            where_clauses.append({"run_id": id_list[0]})
        else:
            where_clauses.append({"run_id": {"$in": id_list}})
    if asset_ref:
        where_clauses.append({"asset_ref": asset_ref})

    where_filter: dict[str, Any] | None = None
    if len(where_clauses) == 1:
        where_filter = where_clauses[0]
    elif len(where_clauses) > 1:
        where_filter = {"$and": where_clauses}

    if query_embeddings is not None:
        results = col.query(
            query_embeddings=query_embeddings,
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    else:
        results = col.query(
            query_texts=[query],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    return _flatten_entity_results(results)


def _flatten_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Chroma's nested result format into a flat list."""
    out: list[dict[str, Any]] = []
    ids_outer = results.get("ids") or [[]]
    docs_outer = results.get("documents") or [[]]
    metas_outer = results.get("metadatas") or [[]]
    dists_outer = results.get("distances") or [[]]

    for ids, docs, metas, dists in zip(ids_outer, docs_outer, metas_outer, dists_outer):
        for chunk_id, text, meta, distance in zip(ids, docs, metas, dists):
            out.append({
                "chunk_id": chunk_id,
                "run_id": (meta or {}).get("run_id", ""),
                "asset_ref": (meta or {}).get("asset_ref", ""),
                "chunk_idx": (meta or {}).get("chunk_idx", 0),
                "start_offset": (meta or {}).get("start_offset", 0),
                "end_offset": (meta or {}).get("end_offset", 0),
                "text": text or "",
                "distance": distance,
            })
    return out


def _flatten_entity_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ids_outer = results.get("ids") or [[]]
    docs_outer = results.get("documents") or [[]]
    metas_outer = results.get("metadatas") or [[]]
    dists_outer = results.get("distances") or [[]]

    for ids, docs, metas, dists in zip(ids_outer, docs_outer, metas_outer, dists_outer):
        for doc_id, text, meta, distance in zip(ids, docs, metas, dists):
            out.append(
                {
                    "doc_id": doc_id,
                    "record_type": (meta or {}).get("record_type", ""),
                    "run_id": (meta or {}).get("run_id", ""),
                    "asset_ref": (meta or {}).get("asset_ref", ""),
                    "entity_id": (meta or {}).get("entity_id", ""),
                    "canonical_label": (meta or {}).get("canonical_label", ""),
                    "mention_id": (meta or {}).get("mention_id", ""),
                    "mention_surface": (meta or {}).get("mention_surface", ""),
                    "reason_unresolved": (meta or {}).get("reason_unresolved", ""),
                    "authority_source": (meta or {}).get("authority_source", ""),
                    "authority_id": (meta or {}).get("authority_id", ""),
                    "wikidata_qid": (meta or {}).get("wikidata_qid", ""),
                    "viaf_id": (meta or {}).get("viaf_id", ""),
                    "geonames_id": (meta or {}).get("geonames_id", ""),
                    "entity_type": (meta or {}).get("entity_type", ""),
                    "chunk_id": (meta or {}).get("chunk_id", ""),
                    "start_offset": (meta or {}).get("start_offset", 0),
                    "end_offset": (meta or {}).get("end_offset", 0),
                    "link_status": (meta or {}).get("link_status", ""),
                    "confidence": (meta or {}).get("confidence", 0),
                    "text": text or "",
                    "distance": distance,
                }
            )
    return out


# ── Evidence formatting ─────────────────────────────────────────────────

def retrieve_debug(
    query: str,
    *,
    k: int = 5,
    run_id: str | None = None,
    asset_ref: str | None = None,
) -> dict[str, Any]:
    """Structured retrieval result for the debug endpoint."""
    run_ids = [run_id] if run_id else None
    hits = retrieve_chunks(query, top_k=k, run_ids=run_ids, asset_ref=asset_ref)
    entity_hits = retrieve_entities(query, top_k=min(k, settings.rag_entity_top_k), run_ids=run_ids, asset_ref=asset_ref)
    results = []
    for h in hits:
        results.append({
            "chunk_id": h["chunk_id"],
            "run_id": h["run_id"],
            "asset_ref": h["asset_ref"],
            "chunk_idx": h["chunk_idx"],
            "offsets": f"{h['start_offset']}-{h['end_offset']}",
            "score": round(1.0 - float(h.get("distance", 0)), 4),
            "text_preview": (h.get("text") or "")[:120],
        })
    entity_results = []
    for hit in entity_hits:
        entity_results.append(
            {
                "doc_id": hit["doc_id"],
                "record_type": hit["record_type"],
                "run_id": hit["run_id"],
                "asset_ref": hit["asset_ref"],
                "entity_id": hit["entity_id"],
                "mention_id": hit["mention_id"],
                "authority_source": hit["authority_source"],
                "authority_id": hit["authority_id"],
                "entity_type": hit["entity_type"],
                "score": round(1.0 - float(hit.get("distance", 0)), 4),
                "text_preview": (hit.get("text") or "")[:160],
            }
        )
    payload = {
        "query": query,
        "k": k,
        "filter": {"run_id": run_id, "asset_ref": asset_ref},
        "results": results,
        "entity_results": entity_results,
    }
    log_retrieve_debug(payload)
    return payload


def format_evidence_blocks(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as ``[OCR_CHUNK_EVIDENCE]…[/OCR_CHUNK_EVIDENCE]`` blocks.

    This is the exact citation format injected into the LLM system
    prompt so the model can ground its answers.
    """
    if not chunks:
        return ""

    blocks: list[str] = []
    for ch in chunks:
        block = (
            "[OCR_CHUNK_EVIDENCE]\n"
            f"run_id: {ch.get('run_id', '')}\n"
            f"asset_ref: {ch.get('asset_ref', '')}\n"
            f"chunk_id: {ch.get('chunk_id', '')}\n"
            f"offsets: {ch.get('start_offset', 0)}-{ch.get('end_offset', 0)}\n"
            f"retrieval_score: {round(1.0 - float(ch.get('distance', 1.0)), 4)}\n"
            f"text: {ch.get('text', '')}\n"
            "[/OCR_CHUNK_EVIDENCE]"
        )
        blocks.append(block)
    return "\n\n".join(blocks)
