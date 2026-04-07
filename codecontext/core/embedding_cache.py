"""
Embedding cache — skip re-embedding unchanged chunks.

Cursor caches embeddings by chunk content hash so that re-indexing the same
codebase (or unchanged files after a Merkle diff) does not re-call the
embedding API.  This module replicates that behavior locally.

Storage: ``<data_dir>/embedding_cache/<provider>_<model_hash>.json``

Each entry maps a SHA-256 hash of the chunk text to its embedding vector.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """In-memory + on-disk cache mapping chunk text hash → embedding vector."""

    def __init__(self, cache_dir: str | Path, provider: str, model: str):
        self._cache_dir = Path(cache_dir) / "embedding_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # One cache file per provider+model combination
        key = f"{provider}_{model}".replace("/", "_").replace(":", "_")
        self._path = self._cache_dir / f"{key}.json"

        self._cache: dict[str, list[float]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._cache = data
                logger.info(
                    "Loaded embedding cache: %d entries from %s",
                    len(self._cache), self._path.name,
                )
            except Exception as exc:
                logger.warning("Failed to load embedding cache: %s", exc)
                self._cache = {}

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self._path.write_text(json.dumps(self._cache))
            logger.info("Saved embedding cache: %d entries", len(self._cache))
            self._dirty = False
        except Exception as exc:
            logger.warning("Failed to save embedding cache: %s", exc)

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text_hash: str) -> list[float] | None:
        return self._cache.get(text_hash)

    def put(self, text_hash: str, embedding: list[float]) -> None:
        self._cache[text_hash] = embedding
        self._dirty = True

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self._dirty = False
        if self._path.exists():
            self._path.unlink()
