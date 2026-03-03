"""Centralised RAG debug flag and logging helpers.

Single source of truth: reads ``ARCHAI_DEBUG_RAG`` from the environment.

Values
------
* ``"1"`` / ``"true"`` / ``"yes"`` → standard debug logging (compact JSON lines)
* ``"verbose"``                    → also logs full evidence text in chat
* anything else / absent           → silent
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

log = logging.getLogger("archai.rag.debug")


def _env_value() -> str:
    return os.getenv("ARCHAI_DEBUG_RAG", "").strip().lower()


def is_rag_debug_enabled() -> bool:
    """``True`` when any level of RAG debug logging is active."""
    return _env_value() in {"1", "true", "yes", "verbose"}


def is_rag_debug_verbose() -> bool:
    """``True`` only when full evidence text should be logged."""
    return _env_value() == "verbose"


# ── compact logging helpers ─────────────────────────────────────────────

def _compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))


def log_index_done(payload: dict[str, Any]) -> None:
    if is_rag_debug_enabled():
        log.info("RAG_INDEX_DONE %s", _compact(payload))


def log_index_status(payload: dict[str, Any]) -> None:
    if is_rag_debug_enabled():
        log.info("RAG_INDEX_STATUS %s", _compact(payload))


def log_retrieve_debug(payload: dict[str, Any]) -> None:
    if is_rag_debug_enabled():
        log.info("RAG_RETRIEVE %s", _compact(payload))


def log_chat_evidence(
    *,
    run_id: str | None,
    asset_ref: str | None,
    k: int,
    evidence_ids: list[dict[str, Any]],
    query: str,
    full_evidence_text: str | None = None,
) -> None:
    """Log ONE compact line summarising the evidence injected into chat."""
    if not is_rag_debug_enabled():
        return
    qhash = hashlib.sha256(query.encode()).hexdigest()[:12]
    payload: dict[str, Any] = {
        "run_id": run_id,
        "asset_ref": asset_ref,
        "k": k,
        "evidence_ids": evidence_ids,
        "query_hash": qhash,
    }
    if is_rag_debug_verbose() and full_evidence_text:
        payload["evidence_text"] = full_evidence_text
    log.info("RAG_CHAT_EVIDENCE %s", _compact(payload))
