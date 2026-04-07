"""
Snapshot manager — persists indexing state to disk.

Tracks which codebases are indexed / indexing / failed so the MCP server
can report status and resume after restarts.

Mirrors packages/mcp/src/snapshot.ts — SnapshotManager class.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codecontext.core.types import SNAPSHOT_FILE

logger = logging.getLogger(__name__)

_LOCK_STALE_SECONDS = 10
_LOCK_MAX_RETRIES = 5
_LOCK_RETRY_MS = 100


@dataclass
class CodebaseInfo:
    status: str  # "indexed" | "indexing" | "indexfailed"
    indexed_files: int = 0
    total_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    index_status: str = "completed"  # "completed" | "limit_reached"
    indexing_percentage: int = 0
    error_message: str = ""
    last_updated: str = ""


class SnapshotManager:
    """
    Manages the persistent codebase snapshot file.

    Mirrors the TS SnapshotManager:
      - Directory-based file locking (stale-lock detection)
      - V1 → V2 format migration on load
      - Re-reads from disk on every getter for multi-process safety
      - Tracks recently-removed entries to prevent re-addition during merge
    """

    def __init__(self, snapshot_path: str | Path | None = None):
        self._path = Path(snapshot_path) if snapshot_path else SNAPSHOT_FILE
        self._lock_path = Path(str(self._path) + ".lock")
        self._codebases: dict[str, CodebaseInfo] = {}
        self._recently_removed: set[str] = set()
        self.load_snapshot()

    # ------------------------------------------------------------------
    # Getters — re-read from disk each time (TS pattern)
    # ------------------------------------------------------------------

    def get_indexed_codebases(self) -> list[str]:
        self._refresh_from_disk()
        return [p for p, info in self._codebases.items() if info.status == "indexed"]

    def get_indexing_codebases(self) -> list[str]:
        self._refresh_from_disk()
        return [p for p, info in self._codebases.items() if info.status == "indexing"]

    def get_failed_codebases(self) -> list[str]:
        self._refresh_from_disk()
        return [p for p, info in self._codebases.items() if info.status == "indexfailed"]

    def get_codebase_info(self, path: str) -> CodebaseInfo | None:
        return self._codebases.get(path)

    def get_codebase_status(self, path: str) -> str:
        """Return 'indexed' | 'indexing' | 'indexfailed' | 'not_found'."""
        info = self._codebases.get(path)
        return info.status if info else "not_found"

    def get_indexing_progress(self, path: str) -> int | None:
        info = self._codebases.get(path)
        if info and info.status == "indexing":
            return info.indexing_percentage
        return None

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def set_codebase_indexing(self, path: str, percentage: int = 0) -> None:
        self._codebases[path] = CodebaseInfo(
            status="indexing",
            indexing_percentage=percentage,
            last_updated=_now(),
        )
        self.save_snapshot()

    def set_codebase_indexed(
        self,
        path: str,
        indexed_files: int = 0,
        total_chunks: int = 0,
        index_status: str = "completed",
        total_files: int = 0,
        skipped_files: int = 0,
    ) -> None:
        self._codebases[path] = CodebaseInfo(
            status="indexed",
            indexed_files=indexed_files,
            total_files=total_files or indexed_files,
            skipped_files=skipped_files,
            total_chunks=total_chunks,
            index_status=index_status,
            last_updated=_now(),
        )
        self.save_snapshot()

    def set_codebase_index_failed(
        self, path: str, error: str, last_pct: int = 0
    ) -> None:
        self._codebases[path] = CodebaseInfo(
            status="indexfailed",
            error_message=error,
            indexing_percentage=last_pct,
            last_updated=_now(),
        )
        self.save_snapshot()

    def remove_codebase(self, path: str) -> None:
        """Remove from all internal state and prevent re-add during merge."""
        self._codebases.pop(path, None)
        self._recently_removed.add(path)
        self.save_snapshot()

    # ------------------------------------------------------------------
    # Persistence — with directory-based locking
    # ------------------------------------------------------------------

    def load_snapshot(self) -> None:
        """Load snapshot from disk (V1 or V2 format). Always saves back as V2."""
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                raw = json.load(f)

            if self._is_v2(raw):
                self._load_v2(raw)
            else:
                self._load_v1(raw)

            # Always persist back as V2 (migration)
            self.save_snapshot()
        except Exception as exc:
            logger.warning("Failed to load snapshot: %s", exc)

    def save_snapshot(self) -> None:
        """Write V2 snapshot to disk with file locking and merge."""
        if not self._acquire_lock():
            logger.warning("Could not acquire snapshot lock — skipping save")
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)

            # Read-merge: incorporate entries from disk we don't know about
            self._merge_external_entries()

            data: dict[str, Any] = {
                "formatVersion": "v2",
                "codebases": {},
                "lastUpdated": _now(),
            }
            for p, info in self._codebases.items():
                data["codebases"][p] = {
                    "status": info.status,
                    "indexedFiles": info.indexed_files,
                    "totalChunks": info.total_chunks,
                    "indexStatus": info.index_status,
                    "indexingPercentage": info.indexing_percentage,
                    "errorMessage": info.error_message,
                    "lastUpdated": info.last_updated,
                }
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)

            self._recently_removed.clear()
        except Exception as exc:
            logger.warning("Failed to save snapshot: %s", exc)
        finally:
            self._release_lock()

    # Alias for backward compatibility
    save = save_snapshot

    # ------------------------------------------------------------------
    # Locking — directory-based (mirrors TS acquireLock / releaseLock)
    # ------------------------------------------------------------------

    def _acquire_lock(self) -> bool:
        for _ in range(_LOCK_MAX_RETRIES):
            try:
                # Check for stale lock
                if self._lock_path.exists():
                    try:
                        age = time.time() - self._lock_path.stat().st_mtime
                        if age > _LOCK_STALE_SECONDS:
                            logger.info("Removing stale snapshot lock (%.1fs old)", age)
                            self._lock_path.rmdir()
                        else:
                            time.sleep(_LOCK_RETRY_MS / 1000)
                            continue
                    except FileNotFoundError:
                        pass  # lock was released between check and remove
                self._lock_path.mkdir(parents=True, exist_ok=False)
                return True
            except FileExistsError:
                time.sleep(_LOCK_RETRY_MS / 1000)
        return False

    def _release_lock(self) -> None:
        try:
            if self._lock_path.exists():
                self._lock_path.rmdir()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal format handling
    # ------------------------------------------------------------------

    @staticmethod
    def _is_v2(raw: dict) -> bool:
        return raw.get("formatVersion") == "v2"

    def _load_v1(self, raw: dict) -> None:
        """Load legacy V1 format and convert to internal state."""
        indexed = raw.get("indexedCodebases", [])
        for p in indexed:
            if not os.path.isdir(p):
                continue
            self._codebases[p] = CodebaseInfo(
                status="indexed",
                index_status="completed",
                last_updated=raw.get("lastUpdated", ""),
            )
        # Any previously-indexing codebases are treated as interrupted
        indexing = raw.get("indexingCodebases", [])
        if isinstance(indexing, list):
            for p in indexing:
                if not os.path.isdir(p):
                    continue
                self._codebases[p] = CodebaseInfo(
                    status="indexfailed",
                    error_message="Indexing was interrupted (V1 migration)",
                    last_updated=raw.get("lastUpdated", ""),
                )

    def _load_v2(self, raw: dict) -> None:
        """Load V2 structured format."""
        codebases_raw = raw.get("codebases", {})
        for p, info in codebases_raw.items():
            if not os.path.isdir(p):
                continue
            status = info.get("status", "indexed")
            # Interrupted indexing → mark as failed
            if status == "indexing":
                self._codebases[p] = CodebaseInfo(
                    status="indexfailed",
                    error_message="Indexing was interrupted",
                    indexing_percentage=info.get("indexingPercentage", 0),
                    last_updated=info.get("lastUpdated", ""),
                )
            else:
                self._codebases[p] = CodebaseInfo(
                    status=status,
                    indexed_files=info.get("indexedFiles", 0),
                    total_chunks=info.get("totalChunks", 0),
                    index_status=info.get("indexStatus", "completed"),
                    indexing_percentage=info.get("indexingPercentage", 0),
                    error_message=info.get("errorMessage", ""),
                    last_updated=info.get("lastUpdated", ""),
                )

    def _refresh_from_disk(self) -> None:
        """Re-read snapshot from disk to pick up changes from other processes."""
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                raw = json.load(f)
            if not self._is_v2(raw):
                return
            codebases_raw = raw.get("codebases", {})
            for p, info in codebases_raw.items():
                if p in self._recently_removed:
                    continue
                if not os.path.isdir(p):
                    continue
                # Only update entries we don't have in memory
                # (memory is authoritative for entries we're managing)
                if p not in self._codebases:
                    self._codebases[p] = CodebaseInfo(
                        status=info.get("status", "indexed"),
                        indexed_files=info.get("indexedFiles", 0),
                        total_chunks=info.get("totalChunks", 0),
                        index_status=info.get("indexStatus", "completed"),
                        indexing_percentage=info.get("indexingPercentage", 0),
                        error_message=info.get("errorMessage", ""),
                        last_updated=info.get("lastUpdated", ""),
                    )
        except Exception:
            pass

    def _merge_external_entries(self) -> None:
        """Merge entries from disk that this process doesn't know about."""
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                raw = json.load(f)
            if not self._is_v2(raw):
                return
            codebases_raw = raw.get("codebases", {})
            for p, info in codebases_raw.items():
                if p in self._recently_removed:
                    continue
                if p not in self._codebases:
                    self._codebases[p] = CodebaseInfo(
                        status=info.get("status", "indexed"),
                        indexed_files=info.get("indexedFiles", 0),
                        total_chunks=info.get("totalChunks", 0),
                        index_status=info.get("indexStatus", "completed"),
                        indexing_percentage=info.get("indexingPercentage", 0),
                        error_message=info.get("errorMessage", ""),
                        last_updated=info.get("lastUpdated", ""),
                    )
        except Exception:
            pass


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
