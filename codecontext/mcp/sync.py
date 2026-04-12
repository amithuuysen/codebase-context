"""
SyncManager — background file-change synchronisation.

Mirrors packages/mcp/src/sync.ts.
"""

from __future__ import annotations

import asyncio
import logging
import os

from codecontext.core.context import Context
from .snapshot import SnapshotManager

logger = logging.getLogger(__name__)


class SyncManager:
    """
    Background file-change synchronisation.

    Mirrors TS SyncManager:
      - isSyncing guard prevents concurrent syncs
      - startBackgroundSync() runs an initial sync after 5 s, then every 5 min
    """

    def __init__(self, context: Context, snapshot: SnapshotManager):
        self._ctx = context
        self._snap = snapshot
        self._is_syncing = False
        self._background_started = False

    async def handle_sync_index(self) -> None:
        """Sync all indexed codebases with filesystem changes."""
        if self._is_syncing:
            return
        self._is_syncing = True
        try:
            for codebase_path in list(self._snap.get_indexed_codebases()):
                if not os.path.isdir(codebase_path):
                    continue
                try:
                    changes = await self._ctx.reindex_by_change(codebase_path)
                    total = changes["added"] + changes["modified"] + changes["removed"]
                    if total > 0:
                        logger.info(
                            "Sync %s: +%d ~%d -%d",
                            codebase_path,
                            changes["added"], changes["modified"], changes["removed"],
                        )
                except Exception as exc:
                    logger.warning("Sync failed for %s: %s", codebase_path, exc)
        except Exception as exc:
            logger.warning("Sync error: %s", exc)
        finally:
            self._is_syncing = False

    def start_background_sync(self, interval: int = 300) -> None:
        """Kick off two-phase background sync (mirrors TS startBackgroundSync).

        Phase 1: initial sync after 5 s.
        Phase 2: periodic sync every *interval* seconds (default 300 = 5 min).
        Only starts once — subsequent calls are no-ops.
        """
        if self._background_started:
            return
        self._background_started = True
        logger.info("Starting background sync (interval=%ds)", interval)

        async def _phase1():
            await asyncio.sleep(5)
            try:
                await self.handle_sync_index()
            except Exception as exc:
                logger.warning("Initial sync error (expected for new installs): %s", exc)

        async def _phase2():
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.handle_sync_index()
                except Exception:
                    pass

        asyncio.ensure_future(_phase1())
        asyncio.ensure_future(_phase2())
