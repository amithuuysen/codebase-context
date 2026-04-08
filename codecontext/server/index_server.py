"""Index Server — centralized HTTP API for remote codebase indexing.

Architecture:
  - User sends files from local machine to this server
  - Server indexes them into FAISS + BM25
  - User searches via their local MCP proxy

Endpoints:
  POST /api/upload       — Upload files for indexing
  POST /api/index        — Trigger indexing of uploaded files
  POST /api/search       — Search indexed codebase
  GET  /api/status/:id   — Get indexing status
  GET  /api/collections  — List all indexed codebases
  DELETE /api/clear/:id  — Clear an index
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from mcp.server.fastmcp import FastMCP

from mcp.server.transport_security import TransportSecuritySettings

from codecontext.core.context import Context
from codecontext.core.embedding import create_embedding
from codecontext.core.reranker import Reranker
from codecontext.core.splitter import AstSplitter
from codecontext.core.types import Config
from codecontext.core.vectordb import FaissVectorDB

logger = logging.getLogger("codecontext.server")

# Server state
_ctx: Context | None = None
_cfg: Config | None = None

# Upload staging: {workspace_id: staging_dir_path}
_upload_staging: dict[str, str] = {}

# Indexing tasks: {workspace_id: {"status": str, "progress": int, ...}}
_index_status: dict[str, dict[str, Any]] = {}


def _get_ctx() -> Context:
    if _ctx is None:
        raise RuntimeError("Server not initialized")
    return _ctx


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

async def upload_files(request: Request) -> JSONResponse:
    """Receive files from a developer's machine for indexing.

    Expects multipart form data with:
      - workspace_id: unique identifier for this codebase (e.g. SimHash or repo name)
      - files: one or more files with relative paths preserved in filename
    """
    form = await request.form()
    workspace_id = form.get("workspace_id")
    if not workspace_id:
        return JSONResponse({"error": "workspace_id required"}, status_code=400)

    workspace_id = str(workspace_id)
    # Create staging directory for this workspace
    staging_base = Path(_cfg.data_dir) / "staging"  # type: ignore
    staging_dir = staging_base / workspace_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    _upload_staging[workspace_id] = str(staging_dir)

    file_count = 0
    for key in form:
        if key == "workspace_id":
            continue
        upload = form[key]
        if hasattr(upload, "read"):
            # The field name is the relative path
            rel_path = key
            content = await upload.read()
            file_path = staging_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)
            file_count += 1

    logger.info("Received %d files for workspace %s", file_count, workspace_id)
    return JSONResponse({
        "workspace_id": workspace_id,
        "files_received": file_count,
        "staging_dir": str(staging_dir),
    })


async def upload_files_json(request: Request) -> JSONResponse:
    """Receive files as JSON payload (simpler for programmatic use).

    Expects JSON body:
    {
        "workspace_id": "my-project",
        "files": {
            "src/main.py": "file content...",
            "src/utils.py": "file content..."
        }
    }
    """
    body = await request.json()
    workspace_id = body.get("workspace_id")
    files = body.get("files", {})

    if not workspace_id:
        return JSONResponse({"error": "workspace_id required"}, status_code=400)
    if not files:
        return JSONResponse({"error": "files dict required"}, status_code=400)

    staging_base = Path(_cfg.data_dir) / "staging"  # type: ignore
    staging_dir = staging_base / workspace_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    _upload_staging[workspace_id] = str(staging_dir)

    file_count = 0
    for rel_path, content in files.items():
        # Prevent path traversal
        safe_path = Path(rel_path)
        if safe_path.is_absolute() or ".." in safe_path.parts:
            continue
        file_path = staging_dir / safe_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        file_count += 1

    logger.info("Received %d files (JSON) for workspace %s", file_count, workspace_id)
    return JSONResponse({
        "workspace_id": workspace_id,
        "files_received": file_count,
    })


async def trigger_index(request: Request) -> JSONResponse:
    """Trigger indexing of uploaded files for a workspace.

    Expects JSON body:
    {
        "workspace_id": "my-project",
        "force": false
    }
    """
    body = await request.json()
    workspace_id = body.get("workspace_id")
    force = body.get("force", False)

    if not workspace_id:
        return JSONResponse({"error": "workspace_id required"}, status_code=400)

    staging_dir = _upload_staging.get(workspace_id)
    if not staging_dir or not os.path.isdir(staging_dir):
        return JSONResponse(
            {"error": f"No staged files for workspace '{workspace_id}'. Upload files first."},
            status_code=400,
        )

    # Check if already indexing
    status = _index_status.get(workspace_id, {})
    if status.get("status") == "indexing":
        return JSONResponse({
            "workspace_id": workspace_id,
            "status": "already_indexing",
            "progress": status.get("progress", 0),
        })

    _index_status[workspace_id] = {
        "status": "indexing",
        "progress": 0,
        "started": time.time(),
    }

    # Run indexing in background
    asyncio.get_event_loop().create_task(
        _background_index(workspace_id, staging_dir, force)
    )

    return JSONResponse({
        "workspace_id": workspace_id,
        "status": "indexing_started",
        "staging_dir": staging_dir,
    })


async def _background_index(workspace_id: str, staging_dir: str, force: bool) -> None:
    """Background task: index the staged files."""
    ctx = _get_ctx()
    try:
        def progress_cb(phase: str, current: int, total: int, pct: int) -> None:
            _index_status[workspace_id] = {
                "status": "indexing",
                "progress": pct,
                "phase": phase,
                "current": current,
                "total": total,
                "started": _index_status[workspace_id].get("started", time.time()),
            }

        result = await ctx.index_codebase(
            staging_dir, progress=progress_cb, force_reindex=force
        )

        _index_status[workspace_id] = {
            "status": "indexed",
            "progress": 100,
            "result": result,
            "completed": time.time(),
            "started": _index_status[workspace_id].get("started", time.time()),
        }

        logger.info("Indexing complete for workspace %s: %s", workspace_id, result)
    except Exception as exc:
        logger.error("Indexing failed for workspace %s: %s", workspace_id, exc)
        _index_status[workspace_id] = {
            "status": "failed",
            "error": str(exc),
            "completed": time.time(),
            "started": _index_status[workspace_id].get("started", time.time()),
        }


async def search(request: Request) -> JSONResponse:
    """Search indexed codebase.

    Expects JSON body:
    {
        "workspace_id": "my-project",
        "query": "how does authentication work",
        "limit": 10
    }
    """
    body = await request.json()
    workspace_id = body.get("workspace_id")
    query = body.get("query")
    limit = min(body.get("limit", 10), 50)

    if not workspace_id or not query:
        return JSONResponse(
            {"error": "workspace_id and query required"}, status_code=400
        )

    staging_dir = _upload_staging.get(workspace_id)
    if not staging_dir:
        return JSONResponse(
            {"error": f"Workspace '{workspace_id}' not found"}, status_code=404
        )

    ctx = _get_ctx()
    try:
        results = await ctx.semantic_search(
            staging_dir, query, top_k=limit, threshold=0.5
        )

        return JSONResponse({
            "workspace_id": workspace_id,
            "query": query,
            "results": [
                {
                    "content": r.content,
                    "relative_path": r.relative_path,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "language": r.language,
                    "score": r.score,
                }
                for r in results
            ],
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


async def get_status(request: Request) -> JSONResponse:
    """Get indexing status for a workspace."""
    workspace_id = request.path_params["workspace_id"]
    status = _index_status.get(workspace_id)
    if not status:
        return JSONResponse(
            {"workspace_id": workspace_id, "status": "not_found"}, status_code=404
        )
    return JSONResponse({"workspace_id": workspace_id, **status})


async def list_collections(request: Request) -> JSONResponse:
    """List all indexed workspaces."""
    ctx = _get_ctx()
    collections = ctx.vector_db.list_collections()
    workspaces = []
    for ws_id, staging in _upload_staging.items():
        col_name = ctx.get_collection_name(staging)
        status = _index_status.get(ws_id, {}).get("status", "unknown")
        workspaces.append({
            "workspace_id": ws_id,
            "collection": col_name,
            "status": status,
            "has_index": col_name in collections,
        })
    return JSONResponse({"workspaces": workspaces})


async def clear(request: Request) -> JSONResponse:
    """Clear index for a workspace."""
    workspace_id = request.path_params["workspace_id"]
    staging_dir = _upload_staging.get(workspace_id)
    if not staging_dir:
        return JSONResponse(
            {"error": f"Workspace '{workspace_id}' not found"}, status_code=404
        )

    ctx = _get_ctx()
    await ctx.clear_index(staging_dir)
    _index_status.pop(workspace_id, None)

    # Clean up staging directory
    shutil.rmtree(staging_dir, ignore_errors=True)
    _upload_staging.pop(workspace_id, None)

    return JSONResponse({"workspace_id": workspace_id, "status": "cleared"})


# ---------------------------------------------------------------------------
# MCP tools — exposed at /mcp so VS Code can connect directly
# ---------------------------------------------------------------------------

_mcp = FastMCP(
    "codecontext-server",
    streamable_http_path="/mcp",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)


@_mcp.tool(description="Index a codebase directory to enable semantic code search.")
async def index_codebase(
    path: str,
    force: bool = False,
) -> str:
    """Index a codebase. Upload files first via SyncClient, then call this.

    Args:
        path: Workspace ID or codebase path to index.
        force: Force re-indexing even if already indexed.
    """
    workspace_id = _workspace_id_from_path(path)
    staging_dir = _upload_staging.get(workspace_id)
    if not staging_dir or not os.path.isdir(staging_dir):
        return f"Error: No staged files for workspace '{workspace_id}'. Upload files first via /api/upload-json."

    status = _index_status.get(workspace_id, {})
    if status.get("status") == "indexing":
        return f"Workspace '{workspace_id}' is already being indexed ({status.get('progress', 0)}% complete)."

    _index_status[workspace_id] = {
        "status": "indexing",
        "progress": 0,
        "started": time.time(),
    }

    asyncio.get_event_loop().create_task(
        _background_index(workspace_id, staging_dir, force)
    )

    return f"Indexing started for workspace '{workspace_id}'. Use get_indexing_status to check progress."


@_mcp.tool(description="Search indexed codebase using natural language queries. Returns relevant code snippets ranked by similarity.")
async def search_code(
    path: str,
    query: str,
    limit: int = 10,
) -> str:
    """Search an indexed codebase using natural language.

    Args:
        path: Workspace ID or codebase path to search.
        query: Natural language search query.
        limit: Max results to return (default 10, max 50).
    """
    if not query:
        return "Error: 'query' is required."

    workspace_id = _workspace_id_from_path(path)
    staging_dir = _upload_staging.get(workspace_id)
    if not staging_dir:
        return f"Error: Workspace '{workspace_id}' not found. Upload and index files first."

    limit = min(limit, 50)
    ctx = _get_ctx()
    try:
        results = await ctx.semantic_search(
            staging_dir, query, top_k=limit, threshold=0.5
        )
    except Exception as exc:
        return f"Error: Search failed: {exc}"

    if not results:
        return f'No results found for query: "{query}" in workspace \'{workspace_id}\''

    lines: list[str] = []
    lines.append(f'Found {len(results)} results for query: "{query}"\n')
    for i, r in enumerate(results, 1):
        loc = f"{r.relative_path}:{r.start_line}-{r.end_line}"
        lines.append(
            f"{i}. Code snippet ({r.language})\n"
            f"   Location: {loc}\n"
            f"   Rank: {i}\n"
            f"   Context:\n```{r.language}\n{r.content}\n```\n"
        )
    return "\n".join(lines)


@_mcp.tool(description="Clear the search index for a workspace.")
async def clear_index(path: str) -> str:
    """Clear the search index for a workspace.

    Args:
        path: Workspace ID or codebase path to clear.
    """
    workspace_id = _workspace_id_from_path(path)
    staging_dir = _upload_staging.get(workspace_id)
    if not staging_dir:
        return f"Error: Workspace '{workspace_id}' not found."

    ctx = _get_ctx()
    await ctx.clear_index(staging_dir)
    _index_status.pop(workspace_id, None)
    shutil.rmtree(staging_dir, ignore_errors=True)
    _upload_staging.pop(workspace_id, None)

    return f"Successfully cleared index for workspace '{workspace_id}'."


@_mcp.tool(description="Get indexing status and progress for a workspace.")
async def get_indexing_status(path: str) -> str:
    """Get indexing status for a workspace.

    Args:
        path: Workspace ID or codebase path to check.
    """
    workspace_id = _workspace_id_from_path(path)
    status = _index_status.get(workspace_id)
    if not status:
        return f"Workspace '{workspace_id}' is not indexed."

    s = status.get("status", "unknown")
    if s == "indexed":
        result = status.get("result", {})
        return (
            f"**Workspace:** {workspace_id}\n"
            f"**Status:** Indexed\n"
            f"**Result:** {result}"
        )
    elif s == "indexing":
        return (
            f"**Workspace:** {workspace_id}\n"
            f"**Status:** Indexing\n"
            f"**Progress:** {status.get('progress', 0)}%\n"
            f"**Phase:** {status.get('phase', 'unknown')}"
        )
    elif s == "failed":
        return (
            f"**Workspace:** {workspace_id}\n"
            f"**Status:** Failed\n"
            f"**Error:** {status.get('error', 'unknown')}"
        )
    return f"Workspace '{workspace_id}' status: {s}"


def _workspace_id_from_path(path: str) -> str:
    """Convert a path to a workspace_id, or return as-is if already an ID."""
    # If it looks like a workspace_id already (no slashes), return as-is
    if "/" not in path and "\\" not in path:
        return path
    # Otherwise derive from path like the MCP proxy does
    base = os.path.basename(path.rstrip("/\\"))
    h = hashlib.md5(path.encode()).hexdigest()[:8]
    return f"{base}_{h}"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(cfg: Config | None = None) -> Starlette:
    """Create the Starlette ASGI app for the index server."""
    global _ctx, _cfg

    _cfg = cfg or Config.from_env()

    # Build context (same wiring as MCP server)
    emb_kwargs: dict = {"model": _cfg.embedding_model}
    if _cfg.embedding_provider == "openai":
        emb_kwargs["api_key"] = _cfg.openai_api_key
        if _cfg.openai_base_url:
            emb_kwargs["base_url"] = _cfg.openai_base_url
    elif _cfg.embedding_provider == "ollama":
        emb_kwargs["host"] = _cfg.ollama_host
    embed_model = create_embedding(_cfg.embedding_provider, **emb_kwargs)

    persist_dir = Path(_cfg.data_dir) / "faiss_store"
    vector_db = FaissVectorDB(persist_dir=persist_dir, embed_model=embed_model)
    splitter = AstSplitter(
        chunk_size=_cfg.chunk_size, chunk_overlap=_cfg.chunk_overlap
    )
    reranker_provider = os.getenv("RERANKER_PROVIDER", "none")
    reranker_model = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker = Reranker(provider=reranker_provider, model=reranker_model)

    _ctx = Context(
        vector_db=vector_db, splitter=splitter, config=_cfg, reranker=reranker
    )

    routes = [
        Route("/api/upload", upload_files, methods=["POST"]),
        Route("/api/upload-json", upload_files_json, methods=["POST"]),
        Route("/api/index", trigger_index, methods=["POST"]),
        Route("/api/search", search, methods=["POST"]),
        Route("/api/status/{workspace_id}", get_status, methods=["GET"]),
        Route("/api/collections", list_collections, methods=["GET"]),
        Route("/api/clear/{workspace_id}", clear, methods=["DELETE"]),
    ]

    # Use MCP's Starlette app as the primary app (it manages the task group
    # lifecycle via its lifespan), then prepend the REST API routes.
    app = _mcp.streamable_http_app()
    for route in reversed(routes):
        app.routes.insert(0, route)

    logger.info("Index server initialized with %d REST routes + MCP at /mcp", len(routes))
    return app


def run_server(host: str = "0.0.0.0", port: int = 8878) -> None:
    """Run the index server standalone."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = Config.from_env()
    app = create_app(cfg)

    logger.info("Starting index server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
