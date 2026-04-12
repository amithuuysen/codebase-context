"""
MCP server entry point — exposes index_codebase, search_code, clear_index,
get_indexing_status as MCP tools via @mcp.tool() decorators.

This module handles LOCAL indexing only.  When INDEX_SERVER_URL is set,
main() delegates to codecontext.remote.server instead.
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
    """Entry point — run the MCP server.

    When INDEX_SERVER_URL is set, delegates to the remote proxy server instead.
    """
    global _ctx, _snap, _sync

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

    # Remote proxy mode: delegate to remote.server
    index_server_url = os.getenv("INDEX_SERVER_URL")
    if index_server_url:
        from codecontext.remote.server import main as remote_main
        remote_main()
        return

    cfg = Config.from_env()
    log_config(cfg)

    _ctx = _build_context(cfg)
    _snap = SnapshotManager()
    _snap.load_snapshot()
    _sync = SyncManager(_ctx, _snap)

    install_shutdown_handlers()

    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    logger.info("MCP server starting on %s (local mode)...", transport)
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
