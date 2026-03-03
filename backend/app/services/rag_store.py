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


def _collection(client: chromadb.ClientAPI | None = None) -> chromadb.Collection:
    """Get or create the chunks collection."""
    client = client or _chroma_client()
    return client.get_or_create_collection(
        name=settings.rag_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ── Provider embeddings (optional) ─────────────────────────────────────

def _provider_embed(texts: list[str]) -> list[list[float]] | None:
    """Try to compute embeddings via the GWDG OpenAI-compatible endpoint.

    Returns ``None`` if the provider is unavailable or the call fails,
    so the caller can fall back to ChromaDB's built-in embeddings.
    """
    model = (settings.rag_embedding_model or "").strip()
    if not model:
        return None  # no provider model configured → use local default

    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        return None

    # Reuse the same key/base-url resolution as chat_ai
    from app.services.chat_ai import _require_api_key, _base_url

    try:
        client = OpenAI(api_key=_require_api_key(), base_url=_base_url())
        response = client.embeddings.create(model=model, input=texts)
        return [list(d.embedding) for d in response.data]
    except Exception as exc:  # noqa: BLE001
        log.warning("Provider embedding failed (%s), falling back to local: %s", model, exc)
        return None


# ── Indexing ────────────────────────────────────────────────────────────

def index_run(run_id: str) -> dict[str, Any]:
    """Index all chunks of a pipeline run into ChromaDB.

    Reads chunks from the SQLite ``chunks`` table, embeds them,
    and upserts into the Chroma collection with provenance metadata.

    Returns a rich summary dict including ``took_ms``.
    """
    t0 = time.perf_counter()
    run = pipeline_db.get_run(run_id)
    if run is None:
        raise ValueError(f"Pipeline run {run_id!r} not found.")

    chunks = pipeline_db.list_chunks(run_id)
    chunks_total = len(chunks)
    asset_ref = str(run.get("asset_ref") or "")

    if not chunks:
        log.info("Run %s has no chunks to index.", run_id)
        result: dict[str, Any] = {
            "run_id": run_id,
            "asset_ref": asset_ref,
            "chunks_total": 0,
            "chunks_indexed": 0,
            "chunks_skipped": 0,
            "collection_name": settings.rag_collection_name,
            "collection_count_after": _collection().count(),
            "took_ms": round((time.perf_counter() - t0) * 1000, 1),
            "status": "empty",
        }
        log_index_done(result)
        return result

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
        metadatas.append({
            "run_id": run_id,
            "asset_ref": str(run.get("asset_ref") or ""),
            "chunk_idx": int(ch["idx"]),
            "start_offset": int(ch["start_offset"]),
            "end_offset": int(ch["end_offset"]),
            "detected_language": str(run.get("detected_language") or ""),
        })

    col = _collection()

    # Try provider embeddings first
    if ids:
        embeddings = _provider_embed(documents)
        if embeddings is not None:
            col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        else:
            # Let Chroma use its built-in default embedding function
            col.upsert(ids=ids, documents=documents, metadatas=metadatas)

    log.info("Indexed %d chunks for run %s", len(ids), run_id)
    result = {
        "run_id": run_id,
        "asset_ref": asset_ref,
        "chunks_total": chunks_total,
        "chunks_indexed": len(ids),
        "chunks_skipped": skipped,
        "collection_name": settings.rag_collection_name,
        "collection_count_after": col.count(),
        "took_ms": round((time.perf_counter() - t0) * 1000, 1),
        "status": "ok",
    }
    log_index_done(result)
    return result


def delete_run(run_id: str) -> int:
    """Remove all chunks belonging to *run_id* from the vector store."""
    col = _collection()
    existing = col.get(where={"run_id": run_id})
    ids = existing["ids"] if existing and existing.get("ids") else []
    if ids:
        col.delete(ids=ids)
    log.info("Deleted %d vectors for run %s", len(ids), run_id)
    return len(ids)


def is_run_indexed(run_id: str) -> dict[str, Any]:
    """Check whether a run has vectors in the store, with missing-ID detail."""
    col = _collection()
    indexed_result = col.get(where={"run_id": run_id}, include=[])
    indexed_ids = set(indexed_result["ids"]) if indexed_result and indexed_result.get("ids") else set()

    # Compare against SQLite chunks to find gaps
    db_chunks = pipeline_db.list_chunks(run_id)
    all_chunk_ids = {str(ch["chunk_id"]) for ch in db_chunks}
    missing = sorted(all_chunk_ids - indexed_ids)

    # Resolve asset_ref from the run record
    run = pipeline_db.get_run(run_id)
    asset_ref = str(run.get("asset_ref") or "") if run else ""

    result: dict[str, Any] = {
        "run_id": run_id,
        "asset_ref": asset_ref,
        "chunks_total": len(all_chunk_ids),
        "chunks_indexed": len(indexed_ids),
        "indexed": len(indexed_ids) > 0,
        "missing_chunk_ids": missing,
    }
    log_index_status(result)
    return result


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
    col = _collection()

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

    # Try provider embedding for the query
    query_embedding = _provider_embed([query])
    if query_embedding is not None:
        results = col.query(
            query_embeddings=query_embedding,
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
    payload = {
        "query": query,
        "k": k,
        "filter": {"run_id": run_id, "asset_ref": asset_ref},
        "results": results,
    }
    log_retrieve_debug(payload)
    return payload


def format_evidence_blocks(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as ``[EVIDENCE]…[/EVIDENCE]`` blocks.

    This is the exact citation format injected into the LLM system
    prompt so the model can ground its answers.
    """
    if not chunks:
        return ""

    blocks: list[str] = []
    for ch in chunks:
        block = (
            "[EVIDENCE]\n"
            f"run_id: {ch.get('run_id', '')}\n"
            f"asset_ref: {ch.get('asset_ref', '')}\n"
            f"chunk_id: {ch.get('chunk_id', '')}\n"
            f"chunk_idx: {ch.get('chunk_idx', 0)}\n"
            f"offsets: {ch.get('start_offset', 0)}-{ch.get('end_offset', 0)}\n"
            f"text: {ch.get('text', '')}\n"
            "[/EVIDENCE]"
        )
        blocks.append(block)
    return "\n\n".join(blocks)
