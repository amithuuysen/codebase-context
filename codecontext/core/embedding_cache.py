"""
Embedding cache — skip re-embedding unchanged chunks.

Cursor caches embeddings by chunk content hash so that re-indexing the same
codebase (or unchanged files after a Merkle diff) does not re-call the
embedding API.  This module replicates that behavior locally.

Storage: ``<data_dir>/embedding_cache/<provider>_<model>.db``  (SQLite)

Each entry maps a SHA-256 hash of the chunk text to its embedding vector.

Performance note:
  The original JSON implementation loaded the entire cache into memory on
  startup (1.2 GB+ for large repos) and rewrote it in full on every save.
  The SQLite backend provides:
    - Near-instant startup (no full load)
    - Incremental writes (INSERT per new entry, periodic COMMIT)
    - O(1) lookups via indexed hash key
    - ~10x less I/O on save (only dirty rows flushed)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import struct
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """SQLite-backed cache mapping chunk text hash → embedding vector.

    Uses an in-memory read-through cache so that repeated lookups during
    indexing are as fast as dict.get() (~50ns) instead of hitting SQLite
    on every call (~50-100μs).  New entries are batched in memory and
    flushed periodically.
    """

    def __init__(self, cache_dir: str | Path, provider: str, model: str):
        self._cache_dir = Path(cache_dir) / "embedding_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        key = f"{provider}_{model}".replace("/", "_").replace(":", "_")
        self._db_path = self._cache_dir / f"{key}.db"
        self._json_path = self._cache_dir / f"{key}.json"  # legacy

        self._pending: dict[str, list[float]] = {}  # dirty entries not yet flushed
        # In-memory read-through cache — avoids repeated SQLite round-trips
        # during indexing.  Populated on-demand by get() and get_batch().
        self._mem_cache: dict[str, list[float]] = {}
        self._count: int = 0
        self._conn: sqlite3.Connection | None = None

        self._init_db()

    # ------------------------------------------------------------------
    # SQLite setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Open (or create) the SQLite database and migrate from JSON if needed."""
        self._conn = sqlite3.connect(str(self._db_path), timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")  # faster concurrent reads
        self._conn.execute("PRAGMA synchronous=NORMAL")  # safe + fast
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  hash TEXT PRIMARY KEY,"
            "  vector BLOB NOT NULL"
            ")"
        )
        self._conn.commit()

        # Count existing entries
        row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        self._count = row[0] if row else 0

        # Auto-migrate from legacy JSON if SQLite is empty and JSON exists
        if self._count == 0 and self._json_path.exists():
            self._migrate_from_json()
        elif self._count > 0:
            logger.info(
                "Loaded embedding cache (SQLite): %d entries from %s",
                self._count, self._db_path.name,
            )

    def _migrate_from_json(self) -> None:
        """One-time migration: import legacy JSON cache into SQLite."""
        try:
            import json as _json
            data = _json.loads(self._json_path.read_text())
            if not isinstance(data, dict) or not data:
                return

            logger.info(
                "Migrating embedding cache: %d entries from JSON → SQLite…",
                len(data),
            )
            assert self._conn is not None
            self._conn.execute("BEGIN")
            for text_hash, embedding in data.items():
                blob = self._encode_vector(embedding)
                self._conn.execute(
                    "INSERT OR IGNORE INTO embeddings (hash, vector) VALUES (?, ?)",
                    (text_hash, blob),
                )
            self._conn.commit()

            row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            self._count = row[0] if row else 0

            # Rename legacy file so we don't re-migrate
            backup = self._json_path.with_suffix(".json.bak")
            self._json_path.rename(backup)
            logger.info(
                "Migration complete: %d entries. Legacy file renamed to %s",
                self._count, backup.name,
            )
        except Exception as exc:
            logger.warning("JSON → SQLite migration failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Vector encoding (compact binary — 4 bytes per float32)
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_vector(vec: list[float]) -> bytes:
        """Pack a float list into compact binary (struct of float32)."""
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _decode_vector(blob: bytes) -> list[float]:
        """Unpack binary blob back to float list."""
        n = len(blob) // 4
        return list(struct.unpack(f"{n}f", blob))

    # ------------------------------------------------------------------
    # Public API (unchanged interface)
    # ------------------------------------------------------------------

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text_hash: str) -> list[float] | None:
        """Lookup embedding by content hash.

        Checks: pending buffer → in-memory cache → SQLite.
        Results from SQLite are promoted to the in-memory cache so
        subsequent lookups for the same hash are instant.
        """
        # 1. Check pending (new entries not yet flushed)
        cached = self._pending.get(text_hash)
        if cached is not None:
            return cached

        # 2. Check in-memory read-through cache
        cached = self._mem_cache.get(text_hash)
        if cached is not None:
            return cached

        # 3. Fall through to SQLite
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE hash = ?", (text_hash,)
        ).fetchone()
        if row is not None:
            vec = self._decode_vector(row[0])
            self._mem_cache[text_hash] = vec  # promote to memory
            return vec
        return None

    def get_batch(self, text_hashes: list[str]) -> dict[str, list[float]]:
        """Bulk lookup — fetches many hashes in a single SQLite query.

        Much faster than calling get() in a loop when processing a batch
        of chunks (e.g., 100 chunks → 1 SQL query instead of 100).
        Returns {hash: embedding} for all found entries.
        """
        result: dict[str, list[float]] = {}
        to_query: list[str] = []

        for h in text_hashes:
            # Check pending + mem_cache first
            vec = self._pending.get(h) or self._mem_cache.get(h)
            if vec is not None:
                result[h] = vec
            else:
                to_query.append(h)

        if to_query and self._conn is not None:
            # SQLite supports up to ~999 bind params; batch in groups
            batch_size = 900
            for i in range(0, len(to_query), batch_size):
                batch = to_query[i:i + batch_size]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT hash, vector FROM embeddings WHERE hash IN ({placeholders})",
                    batch,
                ).fetchall()
                for row_hash, blob in rows:
                    vec = self._decode_vector(blob)
                    result[row_hash] = vec
                    self._mem_cache[row_hash] = vec  # promote to memory

        return result

    def put(self, text_hash: str, embedding: list[float]) -> None:
        """Buffer a new entry. Call save() to flush to disk."""
        self._pending[text_hash] = embedding
        self._mem_cache[text_hash] = embedding  # also in read-through cache

    def save(self) -> None:
        """Flush pending entries to SQLite in a single transaction."""
        if not self._pending:
            return
        assert self._conn is not None
        try:
            self._conn.execute("BEGIN")
            for text_hash, embedding in self._pending.items():
                blob = self._encode_vector(embedding)
                self._conn.execute(
                    "INSERT OR REPLACE INTO embeddings (hash, vector) VALUES (?, ?)",
                    (text_hash, blob),
                )
            self._conn.commit()
            self._count += len(self._pending)
            logger.info(
                "Flushed %d embeddings to cache (%d total)",
                len(self._pending), self._count,
            )
            self._pending.clear()
        except Exception as exc:
            logger.warning("Failed to save embedding cache: %s", exc)
            try:
                self._conn.rollback()
            except Exception:
                pass

    def __len__(self) -> int:
        return self._count + len(self._pending)

    def clear(self) -> None:
        self._pending.clear()
        self._mem_cache.clear()
        self._count = 0
        assert self._conn is not None
        self._conn.execute("DELETE FROM embeddings")
        self._conn.commit()

    def close(self) -> None:
        """Flush pending and close the database connection."""
        self.save()
        if self._conn:
            self._conn.close()
            self._conn = None
