"""Database connection manager and low-level operations."""

from __future__ import annotations

import json
import re
import sqlite3
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .schema import init_db

DEFAULT_DB_PATH = Path.home() / ".myelin" / "memory.db"


def _serialize_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_f32(blob: bytes, dim: int = 768) -> list[float]:
    return list(struct.unpack(f"{dim}f", blob))


def _normalize_fts_token(token: str) -> str:
    """NFKC-normalize a single token and FTS5-safe-quote it."""
    import unicodedata

    normalized = unicodedata.normalize("NFKC", token)
    safe = normalized.replace('"', '""')
    return f'"{safe}"'


def _contains_injection(token: str) -> bool:
    """Reject structural query markers while allowing literal SQL words.

    FTS terms are double-quoted and passed through a ``MATCH ?`` parameter, so
    words such as ``create`` and ``update`` are data, not executable SQL. Keep
    structural markers blocked as defense in depth for direct callers.
    """
    if any(marker in token for marker in (";", "/*", "*/")):
        return True
    # Command-line flags such as ``--force`` are legitimate search text. Treat
    # ``--`` as a comment only by itself or when followed by whitespace.
    return token == "--" or bool(re.search(r"--\s", token))


def _tokenize_fts_query(query: str) -> list[str]:
    """Tokenize a query string into FTS5-safe tokens.

    Splits on non-alphanumeric characters (except . _ / @ : + -).
    Each token is NFKC-normalized, FTS5-safe-quoted, and validated
    for injection. Short tokens (<3 chars) are dropped unless they
    are the only token produced.
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in query:
        if char.isalnum() or char in "._/@:+-":
            current.append(char)
        elif current:
            token = "".join(current)
            if len(token) >= 3:
                tokens.append(token)
            current = []
    if current:
        token = "".join(current)
        if len(token) >= 3 or not tokens:
            tokens.append(token)

    return tokens


def build_fts_where(
    tokens: list[str],
    operator: str = "OR",
    max_len: int = 200,
) -> str:
    """Build a parameterized FTS5 MATCH expression from token list.

    Each token is NFKC-normalized, injection-scanned, and FTS5-safe-quoted.
    Empty token lists produce a no-op sentinel. Total expression length
    is capped at *max_len* characters (ValueError raised if exceeded).

    *operator* is ``"OR"`` or ``"AND"``.
    """
    if not tokens:
        return '"__myelin_no_match__"'

    quoted: list[str] = []
    for token in tokens:
        if _contains_injection(token):
            raise ValueError(f"Rejected potential SQL injection in FTS token: {token!r}")
        quoted.append(_normalize_fts_token(token))

    expr = f" {operator} ".join(quoted)
    if len(expr) > max_len:
        raise ValueError(f"FTS query expression too long ({len(expr)} chars, max {max_len})")
    return expr


def escape_fts_query(query: str) -> str:
    """Convert user text into a safe FTS5 MATCH expression (OR mode).

    Uses OR between tokens instead of AND so partial matches still return
    results. FTS5's default AND means a single missing stopword (e.g. "my",
    "with") kills the entire query — agent queries are almost always broad
    enough that any single token match is meaningful, and the multi-signal
    retriever handles ranking via composite scores.

    Tokens under 3 chars are treated as stopwords and dropped unless they're
    the only token.
    """
    tokens = _tokenize_fts_query(query)
    return build_fts_where(tokens, operator="OR")


def validate_where_clause(where: str | None) -> None:
    """Validate that a WHERE clause fragment does not contain injection.

    Checks for multi-statement, DDL/DML keywords, and comment markers.
    Raises ValueError if dangerous patterns are detected.
    """
    if not where:
        return
    upper = where.upper()
    dangerous = (";", "--", "/*", "*/")
    for pattern in dangerous:
        if pattern in where:
            raise ValueError(f"Rejected dangerous WHERE clause pattern {pattern!r}")
    for kw in (
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "CREATE",
        "EXEC",
        "ATTACH",
        "PRAGMA",
        "REINDEX",
        "REPLACE",
        "VACUUM",
        "UNION",
    ):
        if kw in upper.split():
            raise ValueError(f"Rejected SQL keyword {kw!r} in WHERE clause")


class Database:
    """Thread-local SQLite connection with FTS5 and optional sqlite-vec."""

    def __init__(self, path: str | Path | None = None, enable_vec: bool = True):
        self.path = Path(path) if path else DEFAULT_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._enable_vec = enable_vec
        self._vec_available = False
        self._transaction_depth = 0

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            # Larger page cache keeps the FTS5 index and base-table pages warm
            # across queries; 16MiB is ample for a ~19k-episode corpus.
            self._conn.execute("PRAGMA cache_size=-16384")
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

    @contextmanager
    def transaction(self):
        """Group many operations into one SQLite transaction."""
        self._transaction_depth += 1
        try:
            if self._transaction_depth == 1:
                self.conn.execute("BEGIN")
            yield self
            if self._transaction_depth == 1:
                self.conn.commit()
        except Exception:
            if self._transaction_depth == 1:
                self.conn.rollback()
            raise
        finally:
            self._transaction_depth -= 1

    def _commit_if_needed(self) -> None:
        if self._transaction_depth == 0:
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
        self._commit_if_needed()

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
        self._commit_if_needed()

    def delete(self, table: str, id_value: str) -> None:
        self.conn.execute(f"DELETE FROM {table} WHERE id = ?", (id_value,))
        self._commit_if_needed()

    # ── FTS5 search ────────────────────────────────────────────

    def fts_search(
        self,
        table: str,
        fts_table: str,
        query: str,
        limit: int = 10,
        where: str | None = None,
        where_params: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        where_clause = f"AND ({where})" if where else ""
        if where:
            validate_where_clause(where)
        sql = f"""\
            SELECT t.*, rank
            FROM {fts_table} fts
            JOIN {table} t ON t.rowid = fts.rowid
            WHERE {fts_table} MATCH ? {where_clause}
            ORDER BY rank
            LIMIT ?
        """
        safe_query = escape_fts_query(query)
        params: list[Any] = [safe_query]
        params.extend(where_params)
        params.append(limit)
        return self.fetchall(sql, tuple(params))

    # ── Vector search (requires sqlite-vec) ────────────────────

    def vec_search(
        self,
        table: str,
        embedding_col: str,
        query_vec: list[float],
        limit: int = 10,
        where: str | None = None,
        where_params: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        if not self._vec_available:
            return []

        query_blob = _serialize_f32(query_vec)
        where_clause = f"AND ({where})" if where else ""

        sql = f"""
            SELECT t.*, vec_distance_cosine(t.{embedding_col}, ?) as distance
            FROM {table} t
            WHERE t.{embedding_col} IS NOT NULL {where_clause}
            ORDER BY distance ASC
            LIMIT ?
        """
        params: list[Any] = [query_blob]
        params.extend(where_params)
        params.append(limit)
        return self.fetchall(sql, tuple(params))

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
        embedding_col: str = "embedding",
        where: str | None = None,
        where_params: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        fts_results = self.fts_search(
            table, fts_table, text_query, limit=limit * 2, where=where, where_params=where_params
        )

        if not query_vec or not self._vec_available:
            return fts_results[:limit]

        fts_ids = {r["id"]: i for i, r in enumerate(fts_results)}
        vec_results = self.vec_search(
            table, embedding_col, query_vec, limit=limit * 2, where=where, where_params=where_params
        )
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
