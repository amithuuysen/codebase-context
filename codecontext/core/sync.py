"""
File synchronizer — tracks file changes between sessions using SHA-256 hashes.

Mirrors packages/core/src/sync/synchronizer.ts.

Stores snapshots to ~/.context/merkle/<hash>.json so that re-indexing only
processes added / modified / deleted files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from fnmatch import fnmatch
from pathlib import Path

from .types import MERKLE_DIR

logger = logging.getLogger(__name__)


class FileSynchronizer:
    """Detects added / modified / deleted files between runs."""

    def __init__(self, root_dir: str, ignore_patterns: list[str] | None = None):
        self.root_dir = os.path.abspath(root_dir)
        self.ignore_patterns = ignore_patterns or []
        self._saved_hashes: dict[str, str] = {}
        self._snapshot_path = self._get_snapshot_path(self.root_dir)

    async def initialize(self) -> None:
        self._load_snapshot()

    async def check_for_changes(self) -> dict[str, list[str]]:
        """
        Return {"added": [...], "modified": [...], "removed": [...]}.
        All paths are *relative* to root_dir.
        """
        current = self._scan_hashes()

        added: list[str] = []
        modified: list[str] = []
        removed: list[str] = []

        for rel, h in current.items():
            if rel not in self._saved_hashes:
                added.append(rel)
            elif self._saved_hashes[rel] != h:
                modified.append(rel)

        for rel in self._saved_hashes:
            if rel not in current:
                removed.append(rel)

        self._saved_hashes = current
        self._save_snapshot()

        return {"added": added, "modified": modified, "removed": removed}

    def save_current_state(self) -> None:
        """Force-save the current file hashes as the baseline snapshot."""
        self._saved_hashes = self._scan_hashes()
        self._save_snapshot()

    # ---- snapshot persistence ----

    @staticmethod
    def _get_snapshot_path(root_dir: str) -> str:
        h = hashlib.md5(os.path.abspath(root_dir).encode()).hexdigest()
        MERKLE_DIR.mkdir(parents=True, exist_ok=True)
        return str(MERKLE_DIR / f"{h}.json")

    def _load_snapshot(self) -> None:
        if os.path.exists(self._snapshot_path):
            try:
                with open(self._snapshot_path) as f:
                    self._saved_hashes = json.load(f)
            except Exception:
                self._saved_hashes = {}
        else:
            self._saved_hashes = {}

    def _save_snapshot(self) -> None:
        try:
            MERKLE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self._snapshot_path, "w") as f:
                json.dump(self._saved_hashes, f)
        except Exception as exc:
            logger.warning("Failed to save snapshot: %s", exc)

    @staticmethod
    async def delete_snapshot(root_dir: str) -> None:
        sp = FileSynchronizer._get_snapshot_path(root_dir)
        if os.path.exists(sp):
            os.remove(sp)

    # ---- file scanning ----

    def _scan_hashes(self) -> dict[str, str]:
        """Walk root_dir and return {relative_path: sha256}."""
        hashes: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            dirnames[:] = [
                d for d in dirnames
                if not self._should_ignore(os.path.join(dirpath, d), is_dir=True)
            ]
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                if self._should_ignore(full, is_dir=False):
                    continue
                rel = os.path.relpath(full, self.root_dir)
                try:
                    hashes[rel] = self._hash_file(full)
                except Exception:
                    pass
        return hashes

    @staticmethod
    def _hash_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _should_ignore(self, full_path: str, is_dir: bool) -> bool:
        rel = os.path.relpath(full_path, self.root_dir)
        name = os.path.basename(full_path)

        if name.startswith("."):
            return True

        for pattern in self.ignore_patterns:
            if is_dir and pattern == name:
                return True
            if fnmatch(rel, pattern):
                return True
            if fnmatch(name, pattern):
                return True
            if pattern.endswith("/**") and rel.startswith(pattern[:-3]):
                return True
        return False
