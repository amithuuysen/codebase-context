"""
Background indexing handler — runs indexing as an async task.

Mirrors packages/mcp/src/handlers.ts (startBackgroundIndexing).
Tool handler logic now lives directly in server.py via @mcp.tool() decorators.
"""

from __future__ import annotations

import logging

from codecontext.core.context import Context
from codecontext.core.sync import FileSynchronizer  # noqa: F401 (used in background_indexing)
from .snapshot import SnapshotManager

logger = logging.getLogger(__name__)


async def background_indexing(
    ctx: Context, snap: SnapshotManager, abs_path: str, force: bool
) -> None:
    """Run indexing in the background (mirrors TS startBackgroundIndexing)."""
    try:
        # 1. Load ignore patterns
        await ctx.get_loaded_ignore_patterns(abs_path)

        # 2. Initialize FileSynchronizer
        col_name = ctx.get_collection_name(abs_path)
        sync = FileSynchronizer(abs_path, ctx.get_ignore_patterns())
        await sync.initialize()
        ctx.set_synchronizer(col_name, sync)

        # 3. Index with progress callback
        last_save = [0.0]

        def _progress(phase: str, current: int, total: int, pct: int) -> None:
            snap.set_codebase_indexing(abs_path, pct)
            import time
            now = time.time()
            if now - last_save[0] >= 2.0:
                snap.save_snapshot()
                last_save[0] = now

        result = await ctx.index_codebase(
            abs_path, progress=_progress, force_reindex=force
        )

        # 4. Mark as indexed
        snap.set_codebase_indexed(
            abs_path,
            result["indexed_files"],
            result["total_chunks"],
            index_status=result["status"],
            total_files=result.get("total_files", result["indexed_files"]),
            skipped_files=result.get("skipped_files", 0),
        )
        snap.save_snapshot()
        logger.info(
            "Indexing complete for %s: %d/%d files, %d skipped, %d chunks, status=%s",
            abs_path, result["indexed_files"], result.get("total_files", result["indexed_files"]),
            result.get("skipped_files", 0), result["total_chunks"], result["status"],
        )
    except Exception as exc:
        pct = snap.get_indexing_progress(abs_path) or 0
        snap.set_codebase_index_failed(abs_path, str(exc), last_pct=pct)
        snap.save_snapshot()
        logger.error("Indexing failed for %s: %s", abs_path, exc)
