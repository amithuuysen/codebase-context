"""
MCP server (remote proxy mode) — when INDEX_SERVER_URL is set, all indexing
and search operations are forwarded to the remote index server.

No local Ollama or FAISS needed on the client machine.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

from codecontext.core.types import Config
from codecontext.local.snapshot import SnapshotManager
from codecontext.local.utils import ensure_absolute, install_shutdown_handlers, log_config, truncate
from .remote_search import RemoteSearchProxy

logger = logging.getLogger("codecontext")


# ---------------------------------------------------------------------------
# Module-level state — filled in by main()
# ---------------------------------------------------------------------------

_remote_proxy: RemoteSearchProxy | None = None
_snap: SnapshotManager

# Background remote indexing tasks: {abs_path: {"status": str, "phase": str, "progress": int, ...}}
_remote_tasks: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _workspace_id_for_path(abs_path: str) -> str:
    """Derive workspace_id from an absolute path (matches SyncClient logic)."""
    import hashlib as _hl
    return os.path.basename(abs_path) + "_" + _hl.md5(abs_path.encode()).hexdigest()[:8]


async def _background_remote_index(abs_path: str, force: bool = False) -> None:
    """Background task: upload local files to remote server, trigger indexing, track progress."""
    from .sync_client import SyncClient

    task = _remote_tasks[abs_path]
    workspace_id = task["workspace_id"]
    client = SyncClient(_remote_proxy.server_url, abs_path)

    try:
        # Phase 1: Upload
        task.update({"phase": "uploading", "progress": 0})
        logger.info("Remote index [%s]: uploading files...", workspace_id)

        files = client._collect_files()
        total_files = len(files)
        task["total_files"] = total_files

        if total_files == 0:
            task.update({"status": "failed", "phase": "upload", "error": "No files found to index"})
            return

        # Upload in parallel batches, track progress
        batch_size = 100
        max_concurrent = 4
        rel_paths = list(files.keys())
        batches = []
        for i in range(0, len(rel_paths), batch_size):
            batch_paths = rel_paths[i:i + batch_size]
            batches.append({p: files[p] for p in batch_paths})

        sem = asyncio.Semaphore(max_concurrent)
        uploaded = 0

        async def upload_batch(batch: dict[str, str]) -> None:
            nonlocal uploaded
            async with sem:
                resp = await client._client.post(
                    f"{client.server_url}/api/upload-json",
                    json={"workspace_id": workspace_id, "files": batch},
                )
                resp.raise_for_status()
                uploaded += len(batch)
                pct = int(uploaded / total_files * 50)  # Upload = 0-50%
                task.update({"progress": pct, "files_uploaded": uploaded})

        await asyncio.gather(*[upload_batch(b) for b in batches])
        logger.info("Remote index [%s]: %d files uploaded", workspace_id, uploaded)

        # Phase 2: Trigger indexing
        task.update({"phase": "indexing", "progress": 50})
        resp = await client._client.post(
            f"{client.server_url}/api/index",
            json={"workspace_id": workspace_id, "force": force},
        )
        resp.raise_for_status()

        # Phase 3: Poll remote server for indexing progress
        while True:
            await asyncio.sleep(3.0)
            try:
                status = await _remote_proxy.get_status(workspace_id)
            except Exception:
                continue
            s = status.get("status", "unknown")
            if s == "indexed":
                task.update({"status": "indexed", "phase": "done", "progress": 100})
                logger.info("Remote index [%s]: complete", workspace_id)
                return
            elif s == "failed":
                task.update({"status": "failed", "phase": "indexing", "error": status.get("error", "unknown")})
                logger.error("Remote index [%s]: failed — %s", workspace_id, status.get("error"))
                return
            elif s == "indexing":
                remote_pct = status.get("progress", 0)
                # Map remote 0-100% to our 50-100% range
                pct = 50 + int(remote_pct / 2)
                task.update({"progress": pct, "remote_progress": remote_pct, "remote_phase": status.get("phase", "")})
    except Exception as exc:
        logger.error("Remote index [%s]: error — %s", workspace_id, exc)
        task.update({"status": "failed", "error": str(exc)})
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# FastMCP instance — proxy tools
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "codecontext",
    host="0.0.0.0",
    port=8877,
    streamable_http_path="/mcp",
)


@mcp.tool(description="Index a codebase directory to enable semantic code search using AST or text splitter.")
async def index_codebase(
    path: str,
    force: bool = False,
    customExtensions: Optional[list[str]] = None,
    ignorePatterns: Optional[list[str]] = None,
) -> str:
    """Index a codebase directory for semantic code search (remote proxy).

    Args:
        path: ABSOLUTE path to the codebase directory to index.
        force: Force re-indexing even if already indexed.
        customExtensions: Additional file extensions to index (e.g. ['.vue', '.svelte']).
        ignorePatterns: Additional glob patterns to ignore.
    """
    if not path:
        return "Error: 'path' is required."
    abs_path = ensure_absolute(path)
    if not os.path.isdir(abs_path):
        return f"Error: '{abs_path}' is not a directory."

    workspace_id = _workspace_id_for_path(abs_path)

    # Already running?
    existing = _remote_tasks.get(abs_path)
    if existing and existing.get("status") == "indexing":
        return (
            f"Codebase '{abs_path}' is already being indexed.\n"
            f"Workspace ID: {workspace_id}\n"
            f"Phase: {existing.get('phase', 'unknown')}\n"
            f"Progress: {existing.get('progress', 0)}%\n"
            f"Use get_indexing_status to check progress."
        )

    # Already indexed on remote? (skip unless force)
    if not force:
        try:
            status = await _remote_proxy.get_status(workspace_id)
            if status.get("status") == "indexed":
                return (
                    f"Codebase '{abs_path}' is already indexed on remote server.\n"
                    f"Workspace ID: {workspace_id}\n"
                    f"Use force=True to re-index."
                )
        except Exception:
            pass

    # Launch background upload + index
    _remote_tasks[abs_path] = {
        "status": "indexing",
        "workspace_id": workspace_id,
        "phase": "starting",
        "progress": 0,
        "files_uploaded": 0,
        "total_files": 0,
    }
    asyncio.get_event_loop().create_task(
        _background_remote_index(abs_path, force)
    )

    return (
        f"Remote indexing initiated for '{abs_path}'.\n"
        f"Workspace ID: {workspace_id}\n"
        f"Files are being uploaded and indexed in the background.\n"
        f"Use get_indexing_status to check progress."
    )


@mcp.tool(description="Search indexed codebase using natural language queries. Returns relevant code snippets with file locations ranked by similarity.")
async def search_code(
    path: str,
    query: str,
    extensionFilter: Optional[list[str]] = None,
    compact: bool = False,
) -> str:
    """Search an indexed codebase using natural language (remote proxy).

    Args:
        path: ABSOLUTE path to the indexed codebase directory.
        query: Natural language search query.
        extensionFilter: Filter results by file extensions (e.g. ['.py', '.ts']).
        compact: If true, return only file locations without code snippets.
    """
    if not path:
        return "Error: 'path' is required."
    if not query:
        return "Error: 'query' is required."
    abs_path = ensure_absolute(path)
    if not os.path.isdir(abs_path):
        return f"Error: '{abs_path}' is not a directory."

    limit = 5
    workspace_id = _workspace_id_for_path(abs_path)

    # Check if currently uploading/indexing locally
    local_task = _remote_tasks.get(abs_path)
    if local_task and local_task.get("status") == "indexing":
        phase = local_task.get("phase", "unknown")
        pct = local_task.get("progress", 0)
        return (
            f"Codebase '{abs_path}' is still being indexed.\n"
            f"Phase: {phase} ({pct}% complete)\n"
            f"Use get_indexing_status to check progress. Search will be available once indexing completes."
        )

    # Check remote status
    try:
        status = await _remote_proxy.get_status(workspace_id)
    except Exception:
        status = {"status": "not_found"}

    if status.get("status") == "not_found":
        # Kick off background indexing, don't wait
        _remote_tasks[abs_path] = {
            "status": "indexing",
            "workspace_id": workspace_id,
            "phase": "starting",
            "progress": 0,
            "files_uploaded": 0,
            "total_files": 0,
        }
        asyncio.get_event_loop().create_task(
            _background_remote_index(abs_path, False)
        )
        return (
            f"Codebase '{abs_path}' is not yet indexed on the remote server.\n"
            f"Indexing has been initiated in the background.\n"
            f"Workspace ID: {workspace_id}\n"
            f"Use get_indexing_status to check progress, then search again once complete."
        )

    if status.get("status") == "indexing":
        return (
            f"Codebase '{abs_path}' is currently being indexed on the remote server.\n"
            f"Progress: {status.get('progress', 0)}%\n"
            f"Use get_indexing_status to check progress. Search will be available once indexing completes."
        )

    try:
        results = await _remote_proxy.search(workspace_id, query, top_k=limit)
    except Exception as exc:
        return f"Error: Remote search failed: {exc}"

    if not results:
        return f'No results found for query: "{query}" (remote index)'

    lines: list[str] = []
    lines.append(f'Found {len(results)} results for query: "{query}" (remote index)\n')
    base = os.path.basename(abs_path)
    for i, r in enumerate(results, 1):
        loc = f"{r.relative_path}:{r.start_line}-{r.end_line}"
        if compact:
            lines.append(f"{i}. [{r.language}] {loc}")
        else:
            content = truncate(r.content)
            lines.append(
                f"{i}. Code snippet ({r.language}) [{base}]\n"
                f"   Location: {loc}\n"
                f"   Context:\n```{r.language}\n{content}\n```\n"
            )
    return "\n".join(lines)


@mcp.tool(description="Clear the search index for a codebase.")
async def clear_index(path: str) -> str:
    """Clear the search index for a codebase (remote proxy).

    Args:
        path: ABSOLUTE path to the codebase whose index should be cleared.
    """
    if not path:
        return "Error: 'path' is required."
    abs_path = ensure_absolute(path)

    workspace_id = _workspace_id_for_path(abs_path)
    # Cancel local background task if running
    _remote_tasks.pop(abs_path, None)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(f"{_remote_proxy.server_url}/api/clear/{workspace_id}")
            if resp.status_code == 404:
                return f"Workspace '{workspace_id}' not found on remote server."
            resp.raise_for_status()
        return f"Successfully cleared remote index for '{abs_path}'."
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool(description="Get indexing status and progress for a codebase.")
async def get_indexing_status(path: str) -> str:
    """Get indexing status and progress for a codebase (remote proxy).

    Args:
        path: ABSOLUTE path to the codebase to check status for.
    """
    if not path:
        return "Error: 'path' is required."
    abs_path = ensure_absolute(path)
    workspace_id = _workspace_id_for_path(abs_path)

    # Check local background task first
    local_task = _remote_tasks.get(abs_path)
    if local_task and local_task.get("status") == "indexing":
        phase = local_task.get("phase", "unknown")
        pct = local_task.get("progress", 0)
        uploaded = local_task.get("files_uploaded", 0)
        total = local_task.get("total_files", 0)
        lines = [
            f"**Codebase:** {abs_path}",
            f"**Workspace ID:** {workspace_id}",
            f"**Status:** Indexing 🔄",
            f"**Phase:** {phase}",
            f"**Overall progress:** {pct}%",
        ]
        if phase == "uploading":
            lines.append(f"**Files uploaded:** {uploaded}/{total}")
        elif phase == "indexing":
            remote_pct = local_task.get("remote_progress", 0)
            remote_phase = local_task.get("remote_phase", "")
            lines.append(f"**Remote indexing:** {remote_pct}%")
            if remote_phase:
                lines.append(f"**Remote phase:** {remote_phase}")
        return "\n".join(lines)

    if local_task and local_task.get("status") == "failed":
        return (
            f"**Codebase:** {abs_path}\n"
            f"**Workspace ID:** {workspace_id}\n"
            f"**Status:** Failed ❌\n"
            f"**Error:** {local_task.get('error', 'unknown')}"
        )

    if local_task and local_task.get("status") == "indexed":
        return (
            f"**Codebase:** {abs_path}\n"
            f"**Workspace ID:** {workspace_id}\n"
            f"**Status:** Indexed ✅\n"
            f"**Progress:** 100%\n"
            f"Ready for search."
        )

    # Fall back to remote server status
    try:
        status = await _remote_proxy.get_status(workspace_id)
        s = status.get("status", "unknown")
        lines = [
            f"**Codebase:** {abs_path}",
            f"**Workspace ID:** {workspace_id}",
            f"**Status:** {s}",
        ]
        if s == "indexing":
            lines.append(f"**Progress:** {status.get('progress', 0)}%")
            if status.get("phase"):
                lines.append(f"**Phase:** {status['phase']}")
        elif s == "indexed":
            lines.append("Ready for search.")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Main entry point (called from local/server.py when INDEX_SERVER_URL is set)
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server in remote proxy mode."""
    global _remote_proxy, _snap

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    index_server_url = os.getenv("INDEX_SERVER_URL")
    if not index_server_url:
        raise RuntimeError("INDEX_SERVER_URL must be set for remote proxy mode")

    _remote_proxy = RemoteSearchProxy(index_server_url)
    _snap = SnapshotManager()
    _snap.load_snapshot()

    install_shutdown_handlers()

    logger.info("Remote proxy mode: all operations forwarded to %s", index_server_url)
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    logger.info("MCP server starting on %s (remote proxy)...", transport)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
