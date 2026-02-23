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

                CREATE INDEX IF NOT EXISTS idx_chunks_run_id ON chunks(run_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_run_idx ON chunks(run_id, idx);
                CREATE INDEX IF NOT EXISTS idx_entity_mentions_run_id ON entity_mentions(run_id);
                CREATE INDEX IF NOT EXISTS idx_entity_mentions_offsets ON entity_mentions(run_id, start_offset, end_offset);
                CREATE INDEX IF NOT EXISTS idx_entity_candidates_mention_id ON entity_candidates(mention_id);
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
        conn.execute("DELETE FROM entity_candidates WHERE mention_id IN (SELECT mention_id FROM entity_mentions WHERE run_id=?)", (run_id,))
        conn.execute("DELETE FROM entity_mentions WHERE run_id=?", (run_id,))
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
