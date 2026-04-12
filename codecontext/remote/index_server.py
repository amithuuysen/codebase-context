"""Index Server — centralized HTTP API for remote codebase indexing.

Architecture:
  - User sends files from local machine to this server
  - Server indexes them into FAISS + BM25
  - User searches via their local MCP proxy
  - SimHash-based index reuse: new users get a copy of a similar teammate's
    index instead of indexing from scratch (Architecture §4)

Endpoints:
  POST /api/upload       — Upload files for indexing
  POST /api/index        — Trigger indexing of uploaded files
  POST /api/search       — Search indexed codebase
  GET  /api/status/:id   — Get indexing status
  GET  /api/collections  — List all indexed codebases
  DELETE /api/clear/:id  — Clear an index
  POST /api/register-simhash  — Register SimHash for a workspace
  POST /api/find-similar      — Find the most similar existing index
"""

from __future__ import annotations

import asyncio
import hashlib
import json as _json
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
from codecontext.core.simhash import compute_simhash_from_directory, simhash_similarity
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

# SimHash registry: {workspace_id: simhash_hex_string}
# Persisted to <data_dir>/simhash_registry.json
_simhash_registry: dict[str, str] = {}

# Similarity threshold for index reuse (Architecture §4: ~92% for teammates)
_SIMHASH_REUSE_THRESHOLD = 0.85

def _get_ctx() -> Context:
    if _ctx is None:
        raise RuntimeError("Server not initialized")
    return _ctx


# ---------------------------------------------------------------------------
# SimHash registry persistence
# ---------------------------------------------------------------------------

def _simhash_registry_path() -> Path:
    return Path(_cfg.data_dir) / "simhash_registry.json"  # type: ignore


def _load_simhash_registry() -> None:
    """Load SimHash registry from disk."""
    global _simhash_registry
    path = _simhash_registry_path()
    if path.exists():
        try:
            with open(path) as f:
                _simhash_registry = _json.load(f)
            logger.info("Loaded SimHash registry: %d entries", len(_simhash_registry))
        except Exception as exc:
            logger.warning("Failed to load SimHash registry: %s", exc)
            _simhash_registry = {}


def _save_simhash_registry() -> None:
    """Persist SimHash registry to disk."""
    path = _simhash_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        _json.dump(_simhash_registry, f, indent=2)


def _find_similar_index(simhash: str, exclude_workspace: str | None = None) -> tuple[str | None, float]:
    """Find the most similar existing index by SimHash comparison.

    Returns (workspace_id, similarity) of the best match, or (None, 0.0)
    if no match exceeds the reuse threshold.
    """
    best_ws: str | None = None
    best_sim = 0.0

    for ws_id, ws_hash in _simhash_registry.items():
        if ws_id == exclude_workspace:
            continue
        # Only consider workspaces that are actually indexed
        status = _index_status.get(ws_id, {})
        if status.get("status") != "indexed":
            continue

        sim = simhash_similarity(simhash, ws_hash)
        if sim > best_sim:
            best_sim = sim
            best_ws = ws_id

    if best_sim >= _SIMHASH_REUSE_THRESHOLD:
        return best_ws, best_sim
    return None, best_sim


def _copy_index(source_workspace: str, target_workspace: str, target_staging_dir: str) -> bool:
    """Copy FAISS + BM25 index from source to target workspace.

    Returns True if the copy succeeded.
    """
    ctx = _get_ctx()
    source_staging = _upload_staging.get(source_workspace)
    if not source_staging:
        return False

    source_col = ctx.get_collection_name(source_staging)
    target_col = ctx.get_collection_name(target_staging_dir)

    # Copy FAISS collection
    if not ctx.vector_db.copy_collection(source_col, target_col):
        return False

    # Copy BM25 index file if it exists
    bm25_dir = Path(ctx._cfg.data_dir) / "bm25_store"
    source_bm25 = bm25_dir / f"{source_col}.json"
    target_bm25 = bm25_dir / f"{target_col}.json"
    if source_bm25.exists():
        shutil.copy2(source_bm25, target_bm25)
        logger.info("Copied BM25 index %s → %s", source_col, target_col)

    return True


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
    """Background task: index the staged files.

    Architecture §4 — Index Reuse: Before indexing from scratch, compute
    a SimHash of the staged files and check if a similar index already
    exists on this server. If so, copy it as a starting point and only
    re-index the files that differ.
    """
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

        reused_from: str | None = None

        # --- SimHash-based index reuse (Architecture §4) ---
        if not force:
            _index_status[workspace_id]["phase"] = "computing_simhash"
            simhash = compute_simhash_from_directory(
                staging_dir,
                supported_extensions=ctx.get_supported_extensions(),
            )
            # Register this workspace's SimHash
            _simhash_registry[workspace_id] = simhash
            _save_simhash_registry()
            logger.info("Workspace %s SimHash: %s", workspace_id, simhash[:16])

            # Find a similar existing index
            similar_ws, similarity = _find_similar_index(simhash, exclude_workspace=workspace_id)
            if similar_ws is not None:
                _index_status[workspace_id]["phase"] = "copying_index"
                logger.info(
                    "Found similar index: %s (%.1f%% similar) — copying as starting point",
                    similar_ws, similarity * 100,
                )
                if _copy_index(similar_ws, workspace_id, staging_dir):
                    reused_from = similar_ws
                    logger.info(
                        "Index copied from %s. Will re-index only changed files.",
                        similar_ws,
                    )
                else:
                    logger.warning("Index copy failed, proceeding with full index")

        # --- Index (full or incremental over the copied base) ---
        result = await ctx.index_codebase(
            staging_dir, progress=progress_cb, force_reindex=force
        )

        # Register/update SimHash after successful indexing
        if force or workspace_id not in _simhash_registry:
            simhash = compute_simhash_from_directory(
                staging_dir,
                supported_extensions=ctx.get_supported_extensions(),
            )
            _simhash_registry[workspace_id] = simhash
            _save_simhash_registry()

        status_data: dict[str, Any] = {
            "status": "indexed",
            "progress": 100,
            "result": result,
            "completed": time.time(),
            "started": _index_status[workspace_id].get("started", time.time()),
        }
        if reused_from:
            status_data["reused_from"] = reused_from
        _index_status[workspace_id] = status_data

        logger.info("Indexing complete for workspace %s: %s", workspace_id, result)
    except Exception as exc:
        logger.error("Indexing failed for workspace %s: %s", workspace_id, exc)
        _index_status[workspace_id] = {
            "status": "failed",
            "error": str(exc),
            "completed": time.time(),
            "started": _index_status[workspace_id].get("started", time.time()),
        }


async def register_simhash(request: Request) -> JSONResponse:
    """Register or update a workspace's SimHash fingerprint.

    Expects JSON body:
    {
        "workspace_id": "my-project",
        "simhash": "a3f2c1..."   // optional — if omitted, computed from staged files
    }
    """
    body = await request.json()
    workspace_id = body.get("workspace_id")
    if not workspace_id:
        return JSONResponse({"error": "workspace_id required"}, status_code=400)

    simhash = body.get("simhash")
    if not simhash:
        # Compute from staged files
        staging_dir = _upload_staging.get(workspace_id)
        if not staging_dir or not os.path.isdir(staging_dir):
            return JSONResponse(
                {"error": f"No staged files for '{workspace_id}'. Upload files or provide simhash."},
                status_code=400,
            )
        ctx = _get_ctx()
        simhash = compute_simhash_from_directory(
            staging_dir, supported_extensions=ctx.get_supported_extensions()
        )

    _simhash_registry[workspace_id] = simhash
    _save_simhash_registry()

    return JSONResponse({
        "workspace_id": workspace_id,
        "simhash": simhash,
        "registry_size": len(_simhash_registry),
    })


async def find_similar(request: Request) -> JSONResponse:
    """Find the most similar existing index for a workspace.

    Expects JSON body:
    {
        "workspace_id": "new-user-project",
        "simhash": "a3f2c1..."   // optional — looked up from registry if omitted
    }

    Returns the best matching workspace and similarity score.
    """
    body = await request.json()
    workspace_id = body.get("workspace_id")
    if not workspace_id:
        return JSONResponse({"error": "workspace_id required"}, status_code=400)

    simhash = body.get("simhash") or _simhash_registry.get(workspace_id)
    if not simhash:
        return JSONResponse(
            {"error": f"No SimHash for '{workspace_id}'. Register it first or provide simhash."},
            status_code=400,
        )

    match_ws, similarity = _find_similar_index(simhash, exclude_workspace=workspace_id)

    result: dict[str, Any] = {
        "workspace_id": workspace_id,
        "simhash": simhash,
        "threshold": _SIMHASH_REUSE_THRESHOLD,
    }
    if match_ws:
        result["match"] = {
            "workspace_id": match_ws,
            "similarity": round(similarity, 4),
            "simhash": _simhash_registry.get(match_ws, ""),
        }
    else:
        result["match"] = None
        result["best_similarity"] = round(similarity, 4)

    return JSONResponse(result)


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

    # Remove from SimHash registry
    if workspace_id in _simhash_registry:
        _simhash_registry.pop(workspace_id)
        _save_simhash_registry()

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
    """Index a codebase directory or previously uploaded workspace.

    Args:
        path: Absolute path to a codebase directory on the server, or a workspace ID.
        force: Force re-indexing even if already indexed.
    """
    workspace_id = _workspace_id_from_path(path)

    # Check for uploaded staging files first
    staging_dir = _upload_staging.get(workspace_id)

    # If no staged files, check if path is a directory on this server's filesystem
    if (not staging_dir or not os.path.isdir(staging_dir)) and os.path.isdir(path):
        staging_dir = os.path.abspath(path)
        _upload_staging[workspace_id] = staging_dir

    if not staging_dir or not os.path.isdir(staging_dir):
        return (
            f"Error: No staged files for workspace '{workspace_id}' and "
            f"'{path}' is not a directory on this server. "
            f"Either provide a valid server path or upload files first via /api/upload-json."
        )

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

    return f"Indexing started for workspace '{workspace_id}' from '{staging_dir}'. Use get_indexing_status to check progress."


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

    # If no staged files, check if path exists on this server's filesystem
    if not staging_dir and os.path.isdir(path):
        staging_dir = os.path.abspath(path)
        _upload_staging[workspace_id] = staging_dir

    if not staging_dir:
        return f"Error: Workspace '{workspace_id}' not found. Provide a valid server path or upload files first."

    # Check indexing status
    status = _index_status.get(workspace_id, {})
    if status.get("status") == "indexing":
        pct = status.get("progress", 0)
        phase = status.get("phase", "unknown")
        return (
            f"Workspace '{workspace_id}' is still being indexed ({pct}% — {phase}).\n"
            f"Use get_indexing_status to check progress. Search will be available once indexing completes."
        )
    if status.get("status") == "failed":
        return (
            f"Workspace '{workspace_id}' indexing failed: {status.get('error', 'unknown')}.\n"
            f"Use index_codebase to retry."
        )
    if status.get("status") not in ("indexed",):
        # Not indexed yet — auto-trigger indexing
        _upload_staging[workspace_id] = staging_dir
        _index_status[workspace_id] = {
            "status": "indexing",
            "progress": 0,
            "started": time.time(),
        }
        asyncio.get_event_loop().create_task(
            _background_index(workspace_id, staging_dir, False)
        )
        return (
            f"Workspace '{workspace_id}' was not indexed. Indexing has been initiated.\n"
            f"Use get_indexing_status to check progress, then search again once complete."
        )

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

    # Also check if path is a directory on this server
    if not staging_dir and os.path.isdir(path):
        staging_dir = os.path.abspath(path)

    if not staging_dir:
        return f"Error: Workspace '{workspace_id}' not found."

    ctx = _get_ctx()
    await ctx.clear_index(staging_dir)
    _index_status.pop(workspace_id, None)
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

    # Load SimHash registry from disk (Architecture §4)
    _load_simhash_registry()

    routes = [
        Route("/api/upload", upload_files, methods=["POST"]),
        Route("/api/upload-json", upload_files_json, methods=["POST"]),
        Route("/api/index", trigger_index, methods=["POST"]),
        Route("/api/search", search, methods=["POST"]),
        Route("/api/status/{workspace_id}", get_status, methods=["GET"]),
        Route("/api/collections", list_collections, methods=["GET"]),
        Route("/api/clear/{workspace_id}", clear, methods=["DELETE"]),
        # SimHash-based index reuse (Architecture §4)
        Route("/api/register-simhash", register_simhash, methods=["POST"]),
        Route("/api/find-similar", find_similar, methods=["POST"]),
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
