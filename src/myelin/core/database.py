"""Database connection manager and low-level operations."""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Any

from .schema import init_db

DEFAULT_DB_PATH = Path.home() / ".myelin" / "memory.db"


def _serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_f32(blob: bytes, dim: int = 768) -> list[float]:
    return list(struct.unpack(f"{dim}f", blob))


class Database:
    """Thread-local SQLite connection with FTS5 and optional sqlite-vec."""

    def __init__(self, path: str | Path | None = None, enable_vec: bool = True):
        self.path = Path(path) if path else DEFAULT_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._enable_vec = enable_vec
        self._vec_available = False

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")

            if self._enable_vec:
                try:
                    import sqlite_vec  # noqa: F401

                    self._conn.enable_load_extension(True)
                    sqlite_vec.load(self._conn)
                    self._vec_available = True
                except (ImportError, Exception):
                    self._vec_available = False

            init_db(self._conn)
        return self._conn

    @property
    def vec_available(self) -> bool:
        _ = self.conn
        return self._vec_available

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Generic operations ─────────────────────────────────────

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params: list) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params)

    def commit(self) -> None:
        self.conn.commit()

    def fetchone(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def fetchall(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # ── Insert helpers ─────────────────────────────────────────

    def insert(self, table: str, data: dict[str, Any]) -> None:
        processed: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                processed[k] = json.dumps(v)
            elif isinstance(v, bool):
                processed[k] = int(v)
            else:
                processed[k] = v

        cols = ", ".join(processed.keys())
        placeholders = ", ".join(f":{k}" for k in processed)
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        self.conn.execute(sql, processed)
        self.conn.commit()

    def update(self, table: str, id_value: str, data: dict[str, Any]) -> None:
        processed: dict[str, Any] = {}
        for k, v in data.items():
            if k == "id":
                continue
            if isinstance(v, (list, dict)):
                processed[k] = json.dumps(v)
            elif isinstance(v, bool):
                processed[k] = int(v)
            else:
                processed[k] = v

        sets = ", ".join(f"{k} = :{k}" for k in processed)
        processed["_id"] = id_value
        sql = f"UPDATE {table} SET {sets} WHERE id = :_id"
        self.conn.execute(sql, processed)
        self.conn.commit()

    def delete(self, table: str, id_value: str) -> None:
        self.conn.execute(f"DELETE FROM {table} WHERE id = ?", (id_value,))
        self.conn.commit()

    # ── FTS5 search ────────────────────────────────────────────

    def fts_search(
        self,
        table: str,
        fts_table: str,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        sql = f"""
            SELECT t.*, rank
            FROM {fts_table} fts
            JOIN {table} t ON t.rowid = fts.rowid
            WHERE {fts_table} MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        return self.fetchall(sql, (query, limit))

    # ── Vector search (requires sqlite-vec) ────────────────────

    def vec_search(
        self,
        table: str,
        embedding_col: str,
        query_vec: list[float],
        limit: int = 10,
        where: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self._vec_available:
            return []

        query_blob = _serialize_f32(query_vec)
        where_clause = f"AND {where}" if where else ""

        sql = f"""
            SELECT t.*, vec_distance_cosine(t.{embedding_col}, ?) as distance
            FROM {table} t
            WHERE t.{embedding_col} IS NOT NULL {where_clause}
            ORDER BY distance ASC
            LIMIT ?
        """
        return self.fetchall(sql, (query_blob, limit))

    # ── Hybrid search (FTS + vector) ───────────────────────────

    def hybrid_search(
        self,
        table: str,
        fts_table: str,
        text_query: str,
        query_vec: list[float] | None = None,
        limit: int = 10,
        text_weight: float = 0.4,
        vec_weight: float = 0.6,
    ) -> list[dict[str, Any]]:
        fts_results = self.fts_search(table, fts_table, text_query, limit=limit * 2)

        if not query_vec or not self._vec_available:
            return fts_results[:limit]

        fts_ids = {r["id"]: i for i, r in enumerate(fts_results)}
        embedding_col = "embedding"
        vec_results = self.vec_search(table, embedding_col, query_vec, limit=limit * 2)
        vec_ids = {r["id"]: i for i, r in enumerate(vec_results)}

        all_ids = set(fts_ids.keys()) | set(vec_ids.keys())
        scored: list[tuple[float, dict]] = []

        for item_id in all_ids:
            fts_rank = 1.0 - (fts_ids[item_id] / len(fts_results)) if item_id in fts_ids else 0.0
            vec_rank = 1.0 - (vec_ids[item_id] / len(vec_results)) if item_id in vec_ids else 0.0
            combined = text_weight * fts_rank + vec_weight * vec_rank

            row = next(
                (r for r in fts_results if r["id"] == item_id),
                next((r for r in vec_results if r["id"] == item_id), None),
            )
            if row:
                scored.append((combined, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [row for _, row in scored[:limit]]
