from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _DbConfig:
    path: Path


_DB_LOCK = threading.Lock()
_DB_READY = False


def _default_db_path() -> Path:
    app_dir = Path(__file__).resolve().parents[1]
    return app_dir / "archai.sqlite"


def _db_config() -> _DbConfig:
    raw = str(os.getenv("ARCHAI_DB_PATH", "")).strip()
    if raw:
        return _DbConfig(path=Path(raw).expanduser())
    return _DbConfig(path=_default_db_path())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    cfg = _db_config()
    cfg.path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cfg.path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db_if_needed() -> None:
    global _DB_READY
    if _DB_READY:
        return
    with _DB_LOCK:
        if _DB_READY:
            return
        with _connect() as conn:
            _migrate_entity_decisions_v2(conn)
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    asset_ref TEXT NOT NULL,
                    asset_sha256 TEXT NULL,
                    script_hint TEXT NULL,
                    detected_language TEXT NULL,
                    confidence REAL NULL,
                    warnings_json TEXT NULL,
                    ocr_lines_json TEXT NULL,
                    ocr_text TEXT NULL,
                    proofread_text TEXT NULL,
                    base_text_source TEXT NULL,
                    chunks_count INTEGER NULL,
                    mentions_count INTEGER NULL,
                    error TEXT NULL
                );

                CREATE TABLE IF NOT EXISTS pipeline_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    event TEXT NOT NULL,
                    message TEXT NULL,
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_pipeline_events_run_id ON pipeline_events(run_id);
                CREATE INDEX IF NOT EXISTS idx_pipeline_events_ts ON pipeline_events(ts);

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS entity_mentions (
                    mention_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    chunk_id TEXT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    surface TEXT NOT NULL,
                    norm TEXT NULL,
                    label TEXT NULL,
                    ent_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    method TEXT NOT NULL,
                    notes TEXT NULL,
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS entity_candidates (
                    cand_id TEXT PRIMARY KEY,
                    mention_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    candidate TEXT NOT NULL,
                    score REAL NOT NULL,
                    meta_json TEXT NULL,
                    FOREIGN KEY(mention_id) REFERENCES entity_mentions(mention_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS entity_decisions (
                    decision_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    span_key TEXT NOT NULL,
                    chunk_id TEXT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    surface TEXT NOT NULL,
                    norm TEXT NULL,
                    ent_type_guess TEXT NULL,
                    label TEXT NULL,
                    status TEXT NOT NULL,
                    method TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    confidence REAL NULL,
                    meta_json TEXT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS entity_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    attempt_idx INTEGER NOT NULL,
                    candidate_source TEXT NULL,
                    candidate TEXT NOT NULL,
                    candidate_label TEXT NULL,
                    candidate_type TEXT NULL,
                    nd REAL NULL,
                    bo REAL NULL,
                    threshold_nd REAL NULL,
                    threshold_bo REAL NULL,
                    attempt_decision TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    meta_json TEXT NULL,
                    FOREIGN KEY(decision_id) REFERENCES entity_decisions(decision_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_run_id ON chunks(run_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_run_idx ON chunks(run_id, idx);
                CREATE INDEX IF NOT EXISTS idx_entity_mentions_run_id ON entity_mentions(run_id);
                CREATE INDEX IF NOT EXISTS idx_entity_mentions_offsets ON entity_mentions(run_id, start_offset, end_offset);
                CREATE INDEX IF NOT EXISTS idx_entity_candidates_mention_id ON entity_candidates(mention_id);
                CREATE INDEX IF NOT EXISTS idx_entity_decisions_run_id ON entity_decisions(run_id);
                CREATE INDEX IF NOT EXISTS idx_entity_decisions_run_status ON entity_decisions(run_id, status);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_decisions_span_key
                    ON entity_decisions(run_id, span_key);
                CREATE INDEX IF NOT EXISTS idx_entity_attempts_decision_id ON entity_attempts(decision_id);

                CREATE TABLE IF NOT EXISTS ocr_quality_reports (
                    report_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    pass_idx INTEGER NOT NULL DEFAULT 0,
                    script_family TEXT NOT NULL DEFAULT 'unknown',
                    quality_label TEXT NOT NULL DEFAULT 'OK',
                    char_entropy REAL NOT NULL DEFAULT 0.0,
                    gibberish_score REAL NOT NULL DEFAULT 0.0,
                    non_wordlike_frac REAL NOT NULL DEFAULT 0.0,
                    rare_bigram_ratio REAL NOT NULL DEFAULT 0.0,
                    uncertainty_density REAL NOT NULL DEFAULT 0.0,
                    leading_fragment_ratio REAL NOT NULL DEFAULT 0.0,
                    trailing_fragment_ratio REAL NOT NULL DEFAULT 0.0,
                    cross_pass_stability REAL NOT NULL DEFAULT -1.0,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    line_count INTEGER NOT NULL DEFAULT 0,
                    sanitized_token_count INTEGER NOT NULL DEFAULT 0,
                    token_search_allowed INTEGER NOT NULL DEFAULT 1,
                    ner_allowed INTEGER NOT NULL DEFAULT 1,
                    seam_retry_required INTEGER NOT NULL DEFAULT 0,
                    detail_json TEXT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_ocr_quality_run_id ON ocr_quality_reports(run_id);
                CREATE INDEX IF NOT EXISTS idx_ocr_quality_run_pass ON ocr_quality_reports(run_id, pass_idx);

                CREATE TABLE IF NOT EXISTS tile_audit (
                    tile_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    pass_idx INTEGER NOT NULL DEFAULT 0,
                    tile_idx INTEGER NOT NULL,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    overlap_px INTEGER NOT NULL DEFAULT 0,
                    seam_merge_action TEXT NULL,
                    lines_before_merge INTEGER NULL,
                    lines_after_merge INTEGER NULL,
                    meta_json TEXT NULL,
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_tile_audit_run_id ON tile_audit(run_id);

                CREATE TABLE IF NOT EXISTS ocr_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    attempt_idx INTEGER NOT NULL DEFAULT 0,
                    tiling_strategy TEXT NULL,
                    tile_grid TEXT NULL,
                    overlap_pct REAL NULL,
                    tile_count INTEGER NULL,
                    tile_boxes_json TEXT NULL,
                    preproc_json TEXT NULL,
                    model_used TEXT NULL,
                    text_sha256 TEXT NULL,
                    text_hash TEXT NULL,
                    quality_label TEXT NULL,
                    effective_quality_json TEXT NULL,
                    gibberish_score REAL NULL,
                    leading_fragment_ratio REAL NULL,
                    seam_fragment_ratio REAL NULL,
                    non_wordlike_frac REAL NULL,
                    char_entropy REAL NULL,
                    uncertainty_density REAL NULL,
                    cross_pass_stability REAL NULL,
                    gates_passed INTEGER NOT NULL DEFAULT 0,
                    noop_detected INTEGER NOT NULL DEFAULT 0,
                    decision TEXT NULL,
                    ocr_text TEXT NULL,
                    detail_json TEXT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_ocr_attempts_run_id ON ocr_attempts(run_id);
                CREATE INDEX IF NOT EXISTS idx_ocr_attempts_run_idx ON ocr_attempts(run_id, attempt_idx);

                CREATE TABLE IF NOT EXISTS ocr_backend_results (
                    result_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    page_id TEXT NULL,
                    region_id TEXT NOT NULL,
                    backend_name TEXT NOT NULL,
                    model_name TEXT NULL,
                    confidence REAL NULL,
                    selected INTEGER NOT NULL DEFAULT 0,
                    text TEXT NULL,
                    raw_json TEXT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_ocr_backend_results_run_id ON ocr_backend_results(run_id);
                CREATE INDEX IF NOT EXISTS idx_ocr_backend_results_run_region ON ocr_backend_results(run_id, region_id);

                CREATE TABLE IF NOT EXISTS ocr_benchmark_references (
                    benchmark_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    page_id TEXT NULL,
                    source_label TEXT NULL,
                    reference_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_ocr_benchmark_references_run_id ON ocr_benchmark_references(run_id);
                """
            )
            _ensure_run_columns(conn)
            conn.commit()
        _DB_READY = True


def _ensure_run_columns(conn: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()}
    for column_def in (
        "base_text_source TEXT NULL",
        "chunks_count INTEGER NULL",
        "mentions_count INTEGER NULL",
    ):
        name = column_def.split(" ", 1)[0]
        if name in columns:
            continue
        conn.execute(f"ALTER TABLE pipeline_runs ADD COLUMN {column_def}")

    # Migrate ocr_attempts v1 → v2 (add new columns)
    try:
        att_cols = {str(row["name"]) for row in conn.execute("PRAGMA table_info(ocr_attempts)").fetchall()}
    except Exception:
        att_cols = set()
    for col_def in (
        "tile_grid TEXT NULL",
        "tile_boxes_json TEXT NULL",
        "preproc_json TEXT NULL",
        "text_sha256 TEXT NULL",
        "effective_quality_json TEXT NULL",
        "seam_fragment_ratio REAL NULL",
        "char_entropy REAL NULL",
        "uncertainty_density REAL NULL",
        "cross_pass_stability REAL NULL",
        "noop_detected INTEGER NOT NULL DEFAULT 0",
    ):
        name = col_def.split(" ", 1)[0]
        if name not in att_cols:
            try:
                conn.execute(f"ALTER TABLE ocr_attempts ADD COLUMN {col_def}")
            except Exception:
                pass


def _migrate_entity_decisions_v2(conn: sqlite3.Connection) -> None:
    """Drop v1 entity_decisions (no span_key) so CREATE TABLE IF NOT EXISTS creates v2.

    Also drops entity_attempts in case a partial migration left it behind.
    entity_decisions is derived data — it is fully reconstructed on each
    pipeline run, so dropping it is safe.
    """
    try:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(entity_decisions)").fetchall()}
    except Exception:
        return
    if not cols or "span_key" in cols:
        return  # Table absent or already v2
    conn.execute("DROP TABLE IF EXISTS entity_attempts")
    conn.execute("DROP TABLE IF EXISTS entity_decisions")


def make_span_key(start_offset: int, end_offset: int, surface: str) -> str:
    """Build deterministic span key: ``{start}:{end}:{surface}``."""
    return f"{start_offset}:{end_offset}:{surface}"


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = {key: row[key] for key in row.keys()}
    for key in ("warnings_json", "ocr_lines_json"):
        value = data.get(key)
        if isinstance(value, str):
            try:
                data[key] = json.loads(value)
            except Exception:
                pass
    return data


def create_run(asset_ref: str, asset_sha256: str | None = None) -> str:
    _init_db_if_needed()
    run_id = str(uuid.uuid4())
    ts = now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO pipeline_runs (
                run_id, created_at, updated_at, status, current_stage, asset_ref, asset_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, ts, ts, "RUNNING", "RECEIVED", str(asset_ref or ""), asset_sha256),
        )
        conn.commit()
    return run_id


def log_event(run_id: str, stage: str, event: str, message: str | None = None) -> None:
    _init_db_if_needed()
    ts = now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO pipeline_events (run_id, ts, stage, event, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, ts, stage, event, message),
        )
        conn.execute(
            "UPDATE pipeline_runs SET updated_at=?, current_stage=? WHERE run_id=?",
            (ts, stage, run_id),
        )
        conn.commit()


def update_run_fields(run_id: str, **fields: Any) -> None:
    _init_db_if_needed()
    if not fields:
        return

    allowed = {
        "updated_at",
        "status",
        "current_stage",
        "asset_ref",
        "asset_sha256",
        "script_hint",
        "detected_language",
        "confidence",
        "warnings_json",
        "ocr_lines_json",
        "ocr_text",
        "proofread_text",
        "base_text_source",
        "chunks_count",
        "mentions_count",
        "error",
    }

    updates: list[str] = []
    values: list[Any] = []
    for key, value in fields.items():
        if key not in allowed:
            continue
        updates.append(f"{key}=?")
        values.append(value)

    if "updated_at" not in {k for k in fields.keys() if k in allowed}:
        updates.append("updated_at=?")
        values.append(now_iso())

    if not updates:
        return

    values.append(run_id)
    sql = f"UPDATE pipeline_runs SET {', '.join(updates)} WHERE run_id=?"
    with _connect() as conn:
        conn.execute(sql, values)
        conn.commit()


def get_run(run_id: str) -> dict[str, Any] | None:
    _init_db_if_needed()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM pipeline_runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


def list_events(run_id: str) -> list[dict[str, Any]]:
    _init_db_if_needed()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, run_id, ts, stage, event, message FROM pipeline_events WHERE run_id=? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def _table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [str(row[1]) for row in rows]


def table_view_for_run(run_id: str) -> dict[str, Any]:
    _init_db_if_needed()
    with _connect() as conn:
        columns = _table_columns(conn, "pipeline_runs")
        rows = conn.execute("SELECT * FROM pipeline_runs WHERE run_id=?", (run_id,)).fetchall()
    values = [[row[col] for col in columns] for row in rows]
    return {"table": "pipeline_runs", "columns": columns, "rows": values}


def table_view_for_events(run_id: str) -> dict[str, Any]:
    _init_db_if_needed()
    with _connect() as conn:
        columns = _table_columns(conn, "pipeline_events")
        rows = conn.execute("SELECT * FROM pipeline_events WHERE run_id=? ORDER BY id ASC", (run_id,)).fetchall()
    values = [[row[col] for col in columns] for row in rows]
    return {"table": "pipeline_events", "columns": columns, "rows": values}


def clear_analysis_for_run(run_id: str) -> None:
    _init_db_if_needed()
    with _connect() as conn:
        conn.execute("DELETE FROM entity_attempts WHERE decision_id IN (SELECT decision_id FROM entity_decisions WHERE run_id=?)", (run_id,))
        conn.execute("DELETE FROM entity_candidates WHERE mention_id IN (SELECT mention_id FROM entity_mentions WHERE run_id=?)", (run_id,))
        conn.execute("DELETE FROM entity_mentions WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM entity_decisions WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM chunks WHERE run_id=?", (run_id,))
        conn.commit()


def insert_chunks(run_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _init_db_if_needed()
    inserted: list[dict[str, Any]] = []
    with _connect() as conn:
        for row in rows:
            chunk_id = str(row.get("chunk_id") or uuid.uuid4())
            idx = int(row.get("idx", 0))
            start_offset = int(row.get("start_offset", 0))
            end_offset = int(row.get("end_offset", 0))
            text = str(row.get("text") or "")
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, run_id, idx, start_offset, end_offset, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, run_id, idx, start_offset, end_offset, text),
            )
            inserted.append(
                {
                    "chunk_id": chunk_id,
                    "run_id": run_id,
                    "idx": idx,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "text": text,
                }
            )
        conn.commit()
    return inserted


def insert_entity_mentions(run_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _init_db_if_needed()
    inserted: list[dict[str, Any]] = []
    with _connect() as conn:
        for row in rows:
            mention_id = str(row.get("mention_id") or uuid.uuid4())
            chunk_id = row.get("chunk_id")
            start_offset = int(row.get("start_offset", 0))
            end_offset = int(row.get("end_offset", 0))
            surface = str(row.get("surface") or "")
            norm = row.get("norm")
            label = row.get("label")
            ent_type = str(row.get("ent_type") or "unknown")
            confidence = float(row.get("confidence", 0.0))
            method = str(row.get("method") or "rule:unknown")
            notes = row.get("notes")
            conn.execute(
                """
                INSERT INTO entity_mentions (
                    mention_id, run_id, chunk_id, start_offset, end_offset, surface,
                    norm, label, ent_type, confidence, method, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mention_id,
                    run_id,
                    chunk_id,
                    start_offset,
                    end_offset,
                    surface,
                    norm,
                    label,
                    ent_type,
                    confidence,
                    method,
                    notes,
                ),
            )
            inserted.append(
                {
                    "mention_id": mention_id,
                    "run_id": run_id,
                    "chunk_id": chunk_id,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "surface": surface,
                    "norm": norm,
                    "label": label,
                    "ent_type": ent_type,
                    "confidence": confidence,
                    "method": method,
                    "notes": notes,
                }
            )
        conn.commit()
    return inserted


def insert_entity_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _init_db_if_needed()
    inserted: list[dict[str, Any]] = []
    with _connect() as conn:
        for row in rows:
            cand_id = str(row.get("cand_id") or uuid.uuid4())
            mention_id = str(row.get("mention_id") or "")
            if not mention_id:
                continue
            source = str(row.get("source") or "heuristic")
            candidate = str(row.get("candidate") or "")
            if not candidate:
                continue
            score = float(row.get("score", 0.0))
            meta_json = row.get("meta_json")
            if meta_json is not None and not isinstance(meta_json, str):
                try:
                    meta_json = json.dumps(meta_json, ensure_ascii=False)
                except Exception:
                    meta_json = None
            conn.execute(
                """
                INSERT INTO entity_candidates (cand_id, mention_id, source, candidate, score, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cand_id, mention_id, source, candidate, score, meta_json),
            )
            inserted.append(
                {
                    "cand_id": cand_id,
                    "mention_id": mention_id,
                    "source": source,
                    "candidate": candidate,
                    "score": score,
                    "meta_json": meta_json,
                }
            )
        conn.commit()
    return inserted


def list_chunks(run_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    _init_db_if_needed()
    sql = "SELECT chunk_id, run_id, idx, start_offset, end_offset, text FROM chunks WHERE run_id=? ORDER BY idx ASC"
    params: list[Any] = [run_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def list_entity_mentions(run_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    _init_db_if_needed()
    sql = (
        "SELECT mention_id, run_id, chunk_id, start_offset, end_offset, surface, norm, label, ent_type, confidence, method, notes "
        "FROM entity_mentions WHERE run_id=? ORDER BY start_offset ASC, confidence DESC"
    )
    params: list[Any] = [run_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def count_chunks(run_id: str) -> int:
    _init_db_if_needed()
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM chunks WHERE run_id=?", (run_id,)).fetchone()
    return int(row["c"] if row else 0)


def count_entity_mentions(run_id: str) -> int:
    _init_db_if_needed()
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM entity_mentions WHERE run_id=?", (run_id,)).fetchone()
    return int(row["c"] if row else 0)


def table_view_for_chunks(run_id: str) -> dict[str, Any]:
    _init_db_if_needed()
    with _connect() as conn:
        columns = _table_columns(conn, "chunks")
        rows = conn.execute("SELECT * FROM chunks WHERE run_id=? ORDER BY idx ASC", (run_id,)).fetchall()
    values = [[row[col] for col in columns] for row in rows]
    return {"table": "chunks", "columns": columns, "rows": values}


def table_view_for_entity_mentions(run_id: str) -> dict[str, Any]:
    _init_db_if_needed()
    with _connect() as conn:
        columns = _table_columns(conn, "entity_mentions")
        rows = conn.execute(
            "SELECT * FROM entity_mentions WHERE run_id=? ORDER BY start_offset ASC, confidence DESC",
            (run_id,),
        ).fetchall()
    values = [[row[col] for col in columns] for row in rows]
    return {"table": "entity_mentions", "columns": columns, "rows": values}


def table_view_for_entity_candidates(run_id: str) -> dict[str, Any]:
    _init_db_if_needed()
    with _connect() as conn:
        columns = _table_columns(conn, "entity_candidates")
        rows = conn.execute(
            """
            SELECT c.* FROM entity_candidates c
            JOIN entity_mentions m ON m.mention_id = c.mention_id
            WHERE m.run_id=?
            ORDER BY c.score DESC, c.cand_id ASC
            """,
            (run_id,),
        ).fetchall()
    values = [[row[col] for col in columns] for row in rows]
    return {"table": "entity_candidates", "columns": columns, "rows": values}


# ── entity_decisions CRUD ─────────────────────────────────────────────


def insert_entity_decisions(run_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist final extraction decisions.

    Valid statuses: ACCEPT_LINKABLE, REJECT_LINKABLE, SKIP_NON_LINKABLE,
    FILTERED_OUT.

    Each row must include at least: surface, status, method, reason.
    ``span_key`` is auto-generated from ``{start_offset}:{end_offset}:{surface}``.
    Optional: decision_id, chunk_id, start_offset, end_offset, norm,
              ent_type_guess (or ent_type), label, confidence, meta_json.
    """
    _init_db_if_needed()
    ts = now_iso()
    inserted: list[dict[str, Any]] = []
    with _connect() as conn:
        for row in rows:
            decision_id = str(row.get("decision_id") or uuid.uuid4())
            chunk_id = row.get("chunk_id")
            start_offset = int(row.get("start_offset", 0))
            end_offset = int(row.get("end_offset", 0))
            surface = str(row.get("surface") or "")
            span_key = make_span_key(start_offset, end_offset, surface)
            norm = row.get("norm")
            ent_type_guess = row.get("ent_type_guess") or row.get("ent_type")
            label = row.get("label")
            status = str(row.get("status") or "ACCEPT_LINKABLE")
            method = str(row.get("method") or "unknown")
            reason = str(row.get("reason") or "no_reason")
            confidence = row.get("confidence")
            if confidence is not None:
                confidence = float(confidence)
            meta_json = row.get("meta_json")
            if meta_json is not None and not isinstance(meta_json, str):
                try:
                    meta_json = json.dumps(meta_json, ensure_ascii=False)
                except Exception:
                    meta_json = None
            conn.execute(
                """
                INSERT OR REPLACE INTO entity_decisions (
                    decision_id, run_id, span_key, chunk_id, start_offset, end_offset,
                    surface, norm, ent_type_guess, label, status,
                    method, reason, confidence, meta_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id, run_id, span_key, chunk_id, start_offset, end_offset,
                    surface, norm, ent_type_guess, label, status,
                    method, reason, confidence, meta_json, ts,
                ),
            )
            rec = {
                "decision_id": decision_id,
                "run_id": run_id,
                "span_key": span_key,
                "chunk_id": chunk_id,
                "start_offset": start_offset,
                "end_offset": end_offset,
                "surface": surface,
                "norm": norm,
                "ent_type_guess": ent_type_guess,
                "label": label,
                "status": status,
                "method": method,
                "reason": reason,
                "confidence": confidence,
                "meta_json": meta_json,
                "created_at": ts,
            }
            inserted.append(rec)
        conn.commit()
    return inserted


def list_entity_decisions(
    run_id: str,
    *,
    status: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Query persisted decisions, optionally filtered by status."""
    _init_db_if_needed()
    sql = (
        "SELECT decision_id, run_id, span_key, chunk_id, start_offset, end_offset, "
        "surface, norm, ent_type_guess, label, status, "
        "method, reason, confidence, meta_json, created_at "
        "FROM entity_decisions WHERE run_id=?"
    )
    params: list[Any] = [run_id]
    if status is not None:
        sql += " AND status=?"
        params.append(status)
    sql += " ORDER BY start_offset ASC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        d = {key: row[key] for key in row.keys()}
        if d.get("meta_json") and isinstance(d["meta_json"], str):
            try:
                d["meta_json"] = json.loads(d["meta_json"])
            except Exception:
                pass
        results.append(d)
    return results


def count_entity_decisions(run_id: str, *, status: str | None = None) -> int:
    _init_db_if_needed()
    sql = "SELECT COUNT(*) AS c FROM entity_decisions WHERE run_id=?"
    params: list[Any] = [run_id]
    if status is not None:
        sql += " AND status=?"
        params.append(status)
    with _connect() as conn:
        row = conn.execute(sql, params).fetchone()
    return int(row["c"] if row else 0)


def table_view_for_entity_decisions(run_id: str) -> dict[str, Any]:
    _init_db_if_needed()
    with _connect() as conn:
        columns = _table_columns(conn, "entity_decisions")
        rows = conn.execute(
            "SELECT * FROM entity_decisions WHERE run_id=? ORDER BY start_offset ASC",
            (run_id,),
        ).fetchall()
    values = [[row[col] for col in columns] for row in rows]
    return {"table": "entity_decisions", "columns": columns, "rows": values}


# ── entity_attempts CRUD ─────────────────────────────────────────────


def insert_entity_attempts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist comparison attempts for entity decisions.

    Each row must include: decision_id, candidate, attempt_decision, reason.
    Optional: attempt_idx, candidate_source, candidate_label, candidate_type,
              nd, bo, threshold_nd, threshold_bo, meta_json.
    """
    _init_db_if_needed()
    inserted: list[dict[str, Any]] = []
    with _connect() as conn:
        for row in rows:
            attempt_id = str(row.get("attempt_id") or uuid.uuid4())
            decision_id = str(row.get("decision_id") or "")
            if not decision_id:
                continue
            attempt_idx = int(row.get("attempt_idx", 0))
            candidate_source = row.get("candidate_source")
            candidate = str(row.get("candidate") or "")
            candidate_label = row.get("candidate_label")
            candidate_type = row.get("candidate_type")
            nd = row.get("nd")
            bo = row.get("bo")
            threshold_nd = row.get("threshold_nd")
            threshold_bo = row.get("threshold_bo")
            attempt_decision = str(row.get("attempt_decision") or "REJECT")
            reason = str(row.get("reason") or "no_reason")
            meta_json = row.get("meta_json")
            if meta_json is not None and not isinstance(meta_json, str):
                try:
                    meta_json = json.dumps(meta_json, ensure_ascii=False)
                except Exception:
                    meta_json = None
            conn.execute(
                """
                INSERT INTO entity_attempts (
                    attempt_id, decision_id, attempt_idx, candidate_source,
                    candidate, candidate_label, candidate_type,
                    nd, bo, threshold_nd, threshold_bo,
                    attempt_decision, reason, meta_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id, decision_id, attempt_idx, candidate_source,
                    candidate, candidate_label, candidate_type,
                    nd, bo, threshold_nd, threshold_bo,
                    attempt_decision, reason, meta_json,
                ),
            )
            inserted.append({
                "attempt_id": attempt_id,
                "decision_id": decision_id,
                "attempt_idx": attempt_idx,
                "candidate_source": candidate_source,
                "candidate": candidate,
                "candidate_label": candidate_label,
                "candidate_type": candidate_type,
                "nd": nd,
                "bo": bo,
                "threshold_nd": threshold_nd,
                "threshold_bo": threshold_bo,
                "attempt_decision": attempt_decision,
                "reason": reason,
                "meta_json": meta_json,
            })
        conn.commit()
    return inserted


def list_entity_attempts_for_run(run_id: str) -> list[dict[str, Any]]:
    """Get all attempts for all decisions in a run (joined with decision metadata)."""
    _init_db_if_needed()
    sql = """
        SELECT a.attempt_id, a.decision_id, a.attempt_idx, a.candidate_source,
               a.candidate, a.candidate_label, a.candidate_type,
               a.nd, a.bo, a.threshold_nd, a.threshold_bo,
               a.attempt_decision, a.reason, a.meta_json,
               d.surface AS decision_surface, d.status AS decision_status
        FROM entity_attempts a
        JOIN entity_decisions d ON d.decision_id = a.decision_id
        WHERE d.run_id = ?
        ORDER BY a.decision_id, a.attempt_idx
    """
    with _connect() as conn:
        rows = conn.execute(sql, (run_id,)).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        d = {key: row[key] for key in row.keys()}
        if d.get("meta_json") and isinstance(d["meta_json"], str):
            try:
                d["meta_json"] = json.loads(d["meta_json"])
            except Exception:
                pass
        results.append(d)
    return results


def list_entity_attempts(decision_id: str) -> list[dict[str, Any]]:
    """Get attempts for a specific decision."""
    _init_db_if_needed()
    sql = (
        "SELECT attempt_id, decision_id, attempt_idx, candidate_source, "
        "candidate, candidate_label, candidate_type, "
        "nd, bo, threshold_nd, threshold_bo, "
        "attempt_decision, reason, meta_json "
        "FROM entity_attempts WHERE decision_id=? ORDER BY attempt_idx"
    )
    with _connect() as conn:
        rows = conn.execute(sql, (decision_id,)).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def table_view_for_entity_attempts(run_id: str) -> dict[str, Any]:
    _init_db_if_needed()
    with _connect() as conn:
        columns = _table_columns(conn, "entity_attempts")
        rows = conn.execute(
            """
            SELECT a.* FROM entity_attempts a
            JOIN entity_decisions d ON d.decision_id = a.decision_id
            WHERE d.run_id=?
            ORDER BY a.decision_id, a.attempt_idx
            """,
            (run_id,),
        ).fetchall()
    values = [[row[col] for col in columns] for row in rows]
    return {"table": "entity_attempts", "columns": columns, "rows": values}


def table_view_for_ocr_backend_results(run_id: str) -> dict[str, Any]:
    _init_db_if_needed()
    with _connect() as conn:
        columns = _table_columns(conn, "ocr_backend_results")
        rows = conn.execute(
            "SELECT * FROM ocr_backend_results WHERE run_id=? ORDER BY created_at ASC, region_id ASC",
            (run_id,),
        ).fetchall()
    values = [[row[col] for col in columns] for row in rows]
    return {"table": "ocr_backend_results", "columns": columns, "rows": values}


# ── OCR quality reports ──────────────────────────────────────────────


def insert_ocr_quality_report(run_id: str, report: dict[str, Any]) -> str:
    """Persist an OCR quality report and return its report_id."""
    _init_db_if_needed()
    report_id = str(report.get("report_id") or uuid.uuid4())
    pass_idx = int(report.get("pass_idx", 0))
    detail_json = None
    line_details = report.get("line_details")
    if line_details:
        try:
            detail_json = json.dumps(line_details, ensure_ascii=False)
        except Exception:
            pass

    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO ocr_quality_reports (
                report_id, run_id, pass_idx, script_family, quality_label,
                char_entropy, gibberish_score, non_wordlike_frac,
                rare_bigram_ratio, uncertainty_density,
                leading_fragment_ratio, trailing_fragment_ratio,
                cross_pass_stability, token_count, line_count,
                sanitized_token_count, token_search_allowed, ner_allowed,
                seam_retry_required, detail_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                run_id,
                pass_idx,
                str(report.get("script_family", "unknown")),
                str(report.get("quality_label", "OK")),
                float(report.get("char_entropy", 0.0)),
                float(report.get("gibberish_score", 0.0)),
                float(report.get("non_wordlike_frac", 0.0)),
                float(report.get("rare_bigram_ratio", 0.0)),
                float(report.get("uncertainty_density", 0.0)),
                float(report.get("leading_fragment_ratio", 0.0)),
                float(report.get("trailing_fragment_ratio", 0.0)),
                float(report.get("cross_pass_stability", -1.0)),
                int(report.get("token_count", 0)),
                int(report.get("line_count", 0)),
                int(report.get("sanitized_token_count", 0)),
                1 if report.get("token_search_allowed", True) else 0,
                1 if report.get("ner_allowed", True) else 0,
                1 if report.get("seam_retry_required", False) else 0,
                detail_json,
                now_iso(),
            ),
        )
        conn.commit()
    return report_id


def get_ocr_quality_report(run_id: str, pass_idx: int = 0) -> dict[str, Any] | None:
    """Get the quality report for a specific run and pass."""
    _init_db_if_needed()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM ocr_quality_reports WHERE run_id=? AND pass_idx=? "
            "ORDER BY created_at DESC LIMIT 1",
            (run_id, pass_idx),
        ).fetchone()
    if row is None:
        return None
    d = {key: row[key] for key in row.keys()}
    d["token_search_allowed"] = bool(d.get("token_search_allowed", 1))
    d["ner_allowed"] = bool(d.get("ner_allowed", 1))
    d["seam_retry_required"] = bool(d.get("seam_retry_required", 0))
    if d.get("detail_json") and isinstance(d["detail_json"], str):
        try:
            d["detail_json"] = json.loads(d["detail_json"])
        except Exception:
            pass
    return d


def list_ocr_quality_reports(run_id: str) -> list[dict[str, Any]]:
    """List all quality reports for a run (all passes)."""
    _init_db_if_needed()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ocr_quality_reports WHERE run_id=? ORDER BY pass_idx",
            (run_id,),
        ).fetchall()
    results = []
    for row in rows:
        d = {key: row[key] for key in row.keys()}
        d["token_search_allowed"] = bool(d.get("token_search_allowed", 1))
        d["ner_allowed"] = bool(d.get("ner_allowed", 1))
        d["seam_retry_required"] = bool(d.get("seam_retry_required", 0))
        results.append(d)
    return results


# ── Tile audit ───────────────────────────────────────────────────────


def insert_tile_audit(run_id: str, tiles: list[dict[str, Any]]) -> list[str]:
    """Persist tile grid info for a run. Returns list of tile_ids."""
    _init_db_if_needed()
    ids: list[str] = []
    with _connect() as conn:
        for tile in tiles:
            tile_id = str(tile.get("tile_id") or uuid.uuid4())
            meta = tile.get("meta_json")
            if meta is not None and not isinstance(meta, str):
                try:
                    meta = json.dumps(meta, ensure_ascii=False)
                except Exception:
                    meta = None
            conn.execute(
                """
                INSERT OR REPLACE INTO tile_audit (
                    tile_id, run_id, pass_idx, tile_idx,
                    x, y, width, height, overlap_px,
                    seam_merge_action, lines_before_merge, lines_after_merge,
                    meta_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tile_id,
                    run_id,
                    int(tile.get("pass_idx", 0)),
                    int(tile.get("tile_idx", 0)),
                    int(tile.get("x", 0)),
                    int(tile.get("y", 0)),
                    int(tile.get("width", 0)),
                    int(tile.get("height", 0)),
                    int(tile.get("overlap_px", 0)),
                    tile.get("seam_merge_action"),
                    tile.get("lines_before_merge"),
                    tile.get("lines_after_merge"),
                    meta,
                ),
            )
            ids.append(tile_id)
        conn.commit()
    return ids


def list_tile_audit(run_id: str) -> list[dict[str, Any]]:
    """List tile audit records for a run."""
    _init_db_if_needed()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tile_audit WHERE run_id=? ORDER BY pass_idx, tile_idx",
            (run_id,),
        ).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


# ── OCR attempts ─────────────────────────────────────────────────────


def insert_ocr_attempt(run_id: str, attempt: dict[str, Any]) -> str:
    """Persist one OCR attempt record and return its attempt_id."""
    _init_db_if_needed()
    attempt_id = str(attempt.get("attempt_id") or uuid.uuid4())
    detail = attempt.get("detail_json")
    if detail is not None and not isinstance(detail, str):
        try:
            detail = json.dumps(detail, ensure_ascii=False)
        except Exception:
            detail = None
    tile_boxes = attempt.get("tile_boxes_json")
    if tile_boxes is not None and not isinstance(tile_boxes, str):
        try:
            tile_boxes = json.dumps(tile_boxes, ensure_ascii=False)
        except Exception:
            tile_boxes = None
    preproc = attempt.get("preproc_json")
    if preproc is not None and not isinstance(preproc, str):
        try:
            preproc = json.dumps(preproc, ensure_ascii=False)
        except Exception:
            preproc = None
    eff_q = attempt.get("effective_quality_json")
    if eff_q is not None and not isinstance(eff_q, str):
        try:
            eff_q = json.dumps(eff_q, ensure_ascii=False)
        except Exception:
            eff_q = None

    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO ocr_attempts (
                attempt_id, run_id, attempt_idx, tiling_strategy,
                tile_grid, overlap_pct, tile_count, tile_boxes_json,
                preproc_json, model_used, text_sha256, text_hash,
                quality_label, effective_quality_json,
                gibberish_score, leading_fragment_ratio, seam_fragment_ratio,
                non_wordlike_frac, char_entropy, uncertainty_density,
                cross_pass_stability,
                gates_passed, noop_detected, decision,
                ocr_text, detail_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                run_id,
                int(attempt.get("attempt_idx", 0)),
                attempt.get("tiling_strategy"),
                attempt.get("tile_grid"),
                attempt.get("overlap_pct"),
                attempt.get("tile_count"),
                tile_boxes,
                preproc,
                attempt.get("model_used"),
                attempt.get("text_sha256"),
                attempt.get("text_hash"),
                attempt.get("quality_label"),
                eff_q,
                attempt.get("gibberish_score"),
                attempt.get("leading_fragment_ratio"),
                attempt.get("seam_fragment_ratio"),
                attempt.get("non_wordlike_frac"),
                attempt.get("char_entropy"),
                attempt.get("uncertainty_density"),
                attempt.get("cross_pass_stability"),
                1 if attempt.get("gates_passed") else 0,
                1 if attempt.get("noop_detected") else 0,
                attempt.get("decision"),
                attempt.get("ocr_text"),
                detail,
                now_iso(),
            ),
        )
        conn.commit()
    return attempt_id


def list_ocr_attempts(run_id: str) -> list[dict[str, Any]]:
    """List all OCR attempts for a run, ordered by attempt_idx."""
    _init_db_if_needed()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ocr_attempts WHERE run_id=? ORDER BY attempt_idx",
            (run_id,),
        ).fetchall()
    results = []
    for row in rows:
        d = {key: row[key] for key in row.keys()}
        d["gates_passed"] = bool(d.get("gates_passed", 0))
        d["noop_detected"] = bool(d.get("noop_detected", 0))
        results.append(d)
    return results


def get_best_ocr_attempt(run_id: str) -> dict[str, Any] | None:
    """Get the best (gates_passed=1) or latest OCR attempt for a run."""
    _init_db_if_needed()
    with _connect() as conn:
        # Prefer a passing attempt
        row = conn.execute(
            "SELECT * FROM ocr_attempts WHERE run_id=? AND gates_passed=1 "
            "ORDER BY attempt_idx DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is None:
            # Fallback: latest attempt regardless
            row = conn.execute(
                "SELECT * FROM ocr_attempts WHERE run_id=? "
                "ORDER BY attempt_idx DESC LIMIT 1",
                (run_id,),
            ).fetchone()
    if row is None:
        return None
    d = {key: row[key] for key in row.keys()}
    d["gates_passed"] = bool(d.get("gates_passed", 0))
    d["noop_detected"] = bool(d.get("noop_detected", 0))
    return d


def insert_ocr_backend_results(run_id: str, rows: list[dict[str, Any]]) -> list[str]:
    _init_db_if_needed()
    inserted: list[str] = []
    with _connect() as conn:
        for row in rows:
            result_id = str(row.get("result_id") or uuid.uuid4())
            raw_json = row.get("raw_json")
            if raw_json is not None and not isinstance(raw_json, str):
                try:
                    raw_json = json.dumps(raw_json, ensure_ascii=False)
                except Exception:
                    raw_json = None
            conn.execute(
                """
                INSERT OR REPLACE INTO ocr_backend_results (
                    result_id, run_id, page_id, region_id, backend_name,
                    model_name, confidence, selected, text, raw_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    run_id,
                    row.get("page_id"),
                    row.get("region_id"),
                    row.get("backend_name"),
                    row.get("model_name"),
                    row.get("confidence"),
                    1 if row.get("selected") else 0,
                    row.get("text"),
                    raw_json,
                    now_iso(),
                ),
            )
            inserted.append(result_id)
        conn.commit()
    return inserted


def insert_ocr_benchmark_reference(
    run_id: str,
    *,
    page_id: str | None,
    source_label: str | None,
    reference_text: str,
) -> str:
    _init_db_if_needed()
    benchmark_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO ocr_benchmark_references (
                benchmark_id, run_id, page_id, source_label, reference_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                benchmark_id,
                run_id,
                page_id,
                source_label,
                reference_text,
                now_iso(),
            ),
        )
        conn.commit()
    return benchmark_id


def list_ocr_benchmark_references(run_id: str) -> list[dict[str, Any]]:
    _init_db_if_needed()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ocr_benchmark_references WHERE run_id=? ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


def list_ocr_backend_results(run_id: str) -> list[dict[str, Any]]:
    _init_db_if_needed()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ocr_backend_results WHERE run_id=? ORDER BY created_at ASC, region_id ASC",
            (run_id,),
        ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        data = {key: row[key] for key in row.keys()}
        data["selected"] = bool(data.get("selected", 0))
        results.append(data)
    return results
