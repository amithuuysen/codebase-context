"""
MCP server entry point — exposes index_codebase, search_code, clear_index,
get_indexing_status as MCP tools via @mcp.tool() decorators, runnable over stdio.

Mirrors packages/mcp/src/index.ts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from enum import Enum
from typing import Optional

from mcp.server.fastmcp import FastMCP

from pathlib import Path

from codecontext.core.context import Context
from codecontext.core.embedding import create_embedding
from codecontext.core.reranker import Reranker
from codecontext.core.splitter import AstSplitter, TextSplitter
from codecontext.core.sync import FileSynchronizer
from codecontext.core.types import Config
from codecontext.core.vectordb import FaissVectorDB
from .handlers import background_indexing
from .snapshot import SnapshotManager
from .sync import SyncManager
from .utils import ensure_absolute, install_shutdown_handlers, log_config, show_help, truncate

logger = logging.getLogger("codecontext")

# Remote proxy mode: when INDEX_SERVER_URL is set, search is forwarded
# to a remote index server instead of local FAISS.
_remote_proxy = None  # RemoteSearchProxy | None

# Background remote indexing tasks: {abs_path: {"status": str, "phase": str, "progress": int, ...}}
_remote_tasks: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _background_remote_index(abs_path: str, force: bool = False) -> None:
    """Background task: upload local files to remote server, trigger indexing, track progress."""
    from codecontext.client.sync_client import SyncClient

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


def _workspace_id_for_path(abs_path: str) -> str:
    """Derive workspace_id from an absolute path (matches SyncClient logic)."""
    import hashlib as _hl
    return os.path.basename(abs_path) + "_" + _hl.md5(abs_path.encode()).hexdigest()[:8]


def _build_context(cfg: Config) -> Context:
    """Wire up embedding → FAISS vector DB → BM25 → splitter → reranker → Context."""
    emb_kwargs: dict = {"model": cfg.embedding_model}
    if cfg.embedding_provider == "openai":
        emb_kwargs["api_key"] = cfg.openai_api_key
        if cfg.openai_base_url:
            emb_kwargs["base_url"] = cfg.openai_base_url
    elif cfg.embedding_provider == "ollama":
        emb_kwargs["host"] = cfg.ollama_host
    embed_model = create_embedding(cfg.embedding_provider, **emb_kwargs)

    persist_dir = Path(cfg.data_dir) / "faiss_store"
    vector_db = FaissVectorDB(persist_dir=persist_dir, embed_model=embed_model)
    splitter = AstSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    # Optional cross-encoder reranker (env: RERANKER_PROVIDER, RERANKER_MODEL)
    reranker_provider = os.getenv("RERANKER_PROVIDER", "none")
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = Reranker(provider=reranker_provider, model=reranker_model)

    return Context(
        vector_db=vector_db, splitter=splitter, config=cfg, reranker=reranker
    )


# ---------------------------------------------------------------------------
# FastMCP instance — tools are registered via @mcp.tool() below
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "codecontext",
    host="0.0.0.0",
    port=8877,
    streamable_http_path="/mcp",
)

# Module-level state filled in by main() before the server starts
_ctx: Context | None = None
_snap: SnapshotManager
_sync: SyncManager | None = None


# ---------------------------------------------------------------------------
# Splitter enum for the tool parameter
# ---------------------------------------------------------------------------

class SplitterType(str, Enum):
    ast = "ast"
    text = "text"


# ---------------------------------------------------------------------------
# MCP tools — registered with @mcp.tool()
# ---------------------------------------------------------------------------

@mcp.tool(description="Index a codebase directory to enable semantic code search using AST or text splitter.")
async def index_codebase(
    path: str,
    force: bool = False,
    splitter: SplitterType = SplitterType.ast,
    customExtensions: Optional[list[str]] = None,
    ignorePatterns: Optional[list[str]] = None,
) -> str:
    """Index a codebase directory for semantic code search.

    Args:
        path: ABSOLUTE path to the codebase directory to index.
        force: Force re-indexing even if already indexed.
        splitter: Splitting strategy — 'ast' (tree-sitter) or 'text' (character-based).
        customExtensions: Additional file extensions to index (e.g. ['.vue', '.svelte']).
        ignorePatterns: Additional glob patterns to ignore.
    """
    if not path:
        return "Error: 'path' is required."
    abs_path = ensure_absolute(path)
    if not os.path.isdir(abs_path):
        return f"Error: '{abs_path}' is not a directory."

    # --- Remote proxy mode: upload files to index server ---
    if _remote_proxy:
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

    # --- Local mode ---
    # Guard: already indexing?
    if abs_path in _snap.get_indexing_codebases():
        pct = _snap.get_indexing_progress(abs_path)
        return (
            f"Codebase '{abs_path}' is already being indexed "
            f"({pct or 0}% complete). Please wait."
        )

    # Splitter choice
    splitter_type = splitter.value
    if splitter_type == "text":
        _ctx.update_splitter(TextSplitter(chunk_size=_ctx._cfg.chunk_size, chunk_overlap=_ctx._cfg.chunk_overlap))
    else:
        _ctx.update_splitter(AstSplitter(chunk_size=_ctx._cfg.chunk_size, chunk_overlap=_ctx._cfg.chunk_overlap))

    # Custom extensions / ignore patterns
    if customExtensions:
        _ctx.add_custom_extensions(customExtensions)
    if ignorePatterns:
        _ctx.add_custom_ignore_patterns(ignorePatterns)

    # Mark as indexing and kick off background task
    _snap.set_codebase_indexing(abs_path, 0)
    asyncio.get_event_loop().create_task(background_indexing(_ctx, _snap, abs_path, force, _sync))

    return (
        f"Indexing started for '{abs_path}'.\n"
        f"Splitter: {splitter_type}\n"
        f"Use get_indexing_status to check progress.\n"
        f"You can search while indexing is in progress (results may be partial)."
    )


@mcp.tool(description="Search indexed codebase using natural language queries. Returns relevant code snippets with file locations ranked by similarity.")
async def search_code(
    path: str,
    query: str,
    extensionFilter: Optional[list[str]] = None,
    compact: bool = False,
) -> str:
    """Search an indexed codebase using natural language.

    Args:
        path: ABSOLUTE path to the indexed codebase directory.
        query: Natural language search query.
        extensionFilter: Filter results by file extensions (e.g. ['.py', '.ts']).
        compact: If true, return only file locations without code snippets (much lower token usage).
    """
    if not path:
        return "Error: 'path' is required."
    if not query:
        return "Error: 'query' is required."
    abs_path = ensure_absolute(path)
    if not os.path.isdir(abs_path):
        return f"Error: '{abs_path}' is not a directory."

    limit = 5

    # --- Remote proxy mode: forward search to index server ---
    if _remote_proxy:
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

    # --- Local mode ---
    # Check index status
    is_indexed = abs_path in _snap.get_indexed_codebases()
    is_indexing = abs_path in _snap.get_indexing_codebases()

    if not is_indexed and not is_indexing:
        has = await _ctx.has_index(abs_path)
        if has:
            _snap.set_codebase_indexed(abs_path)
            is_indexed = True
        else:
            return (
                f"Error: Codebase '{abs_path}' is not indexed. "
                "Please index it first using the index_codebase tool."
            )

    # Extension filter
    filter_expr = None
    if extensionFilter:
        quoted = ", ".join(f"'{e}'" for e in extensionFilter)
        filter_expr = f"file_extension in [{quoted}]"

    results = await _ctx.semantic_search(
        abs_path, query, top_k=limit, threshold=0.5, filter_expr=filter_expr,
    )

    if not results:
        msg = f'No results found for query: "{query}" in codebase \'{abs_path}\''
        if is_indexing:
            msg += "\n\nNote: This codebase is still being indexed. Try again later."
        return msg

    # Format results
    lines: list[str] = []
    indexing_note = ""
    if is_indexing:
        indexing_note = "\n⚠️  **Indexing in Progress** — results may be incomplete.\n"

    lines.append(
        f'Found {len(results)} results for query: "{query}" '
        f'in codebase \'{abs_path}\'{indexing_note}\n'
    )
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
    """Clear the search index for a codebase.

    Args:
        path: ABSOLUTE path to the codebase whose index should be cleared.
    """
    if not path:
        return "Error: 'path' is required."
    abs_path = ensure_absolute(path)

    # --- Remote proxy mode ---
    if _remote_proxy:
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

    # --- Local mode ---
    status = _snap.get_codebase_status(abs_path)
    if status == "not_found":
        has = await _ctx.has_index(abs_path)
        if not has:
            return f"Codebase '{abs_path}' is not indexed."

    await _ctx.clear_index(abs_path)
    _snap.remove_codebase(abs_path)
    _snap.save_snapshot()
    return f"Successfully cleared index for '{abs_path}'."


@mcp.tool(description="Get indexing status and progress for a codebase.")
async def get_indexing_status(path: str) -> str:
    """Get indexing status and progress for a codebase.

    Args:
        path: ABSOLUTE path to the codebase to check status for.
    """
    if not path:
        return "Error: 'path' is required."
    abs_path = ensure_absolute(path)

    # --- Remote proxy mode ---
    if _remote_proxy:
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

    # --- Local mode ---
    status = _snap.get_codebase_status(abs_path)
    info = _snap.get_codebase_info(abs_path)

    if status == "indexed" and info:
        return (
            f"**Codebase:** {abs_path}\n"
            f"**Status:** Indexed ✅\n"
            f"**Files indexed:** {info.indexed_files}/{info.total_files}"
            f"{f' ({info.skipped_files} skipped)' if info.skipped_files else ''}\n"
            f"**Total chunks:** {info.total_chunks}\n"
            f"**Index status:** {info.index_status}\n"
            f"**Last updated:** {info.last_updated}"
        )
    elif status == "indexing" and info:
        return (
            f"**Codebase:** {abs_path}\n"
            f"**Status:** Indexing 🔄\n"
            f"**Progress:** {info.indexing_percentage}%\n"
            f"**Last updated:** {info.last_updated}"
        )
    elif status == "indexfailed" and info:
        return (
            f"**Codebase:** {abs_path}\n"
            f"**Status:** Failed ❌\n"
            f"**Error:** {info.error_message}\n"
            f"**Progress at failure:** {info.indexing_percentage}%\n"
            f"**Last updated:** {info.last_updated}"
        )
    else:
        return (
            f"**Codebase:** {abs_path}\n"
            f"**Status:** Not indexed\n\n"
            f"Use the index_codebase tool to index this codebase."
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — run the MCP server over stdio."""
    global _ctx, _snap, _sync, _remote_proxy

    # --help support
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit(0)

    # Redirect logging to stderr (stdout reserved for MCP JSON protocol)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    cfg = Config.from_env()
    log_config(cfg)

    # Remote proxy mode: skip local Ollama/FAISS entirely
    index_server_url = os.getenv("INDEX_SERVER_URL")
    if index_server_url:
        from codecontext.client.remote_search import RemoteSearchProxy
        _remote_proxy = RemoteSearchProxy(index_server_url)
        _snap = SnapshotManager()
        _snap.load_snapshot()
        logger.info("Remote proxy mode: all operations forwarded to %s", index_server_url)
    else:
        _ctx = _build_context(cfg)
        _snap = SnapshotManager()
        _snap.load_snapshot()
        _sync = SyncManager(_ctx, _snap)

    install_shutdown_handlers()

    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    logger.info("MCP server starting on %s...", transport)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
