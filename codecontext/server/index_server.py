"""
Index Server — centralized HTTP API for team codebase indexing.

Architecture:
  - Developers send files (or file diffs) to this server
  - Server indexes them into FAISS + BM25 (shared indexes)
  - Developers search via their local MCP proxy
  - SimHash registry enables index reuse across teammates

Endpoints:
  POST /api/upload       — Upload files for indexing
  POST /api/index        — Trigger indexing of uploaded files
  POST /api/search       — Search indexed codebase
  GET  /api/status/:id   — Get indexing status
  POST /api/simhash      — Register/lookup SimHash for index sharing
  GET  /api/collections  — List all indexed codebases
  DELETE /api/clear/:id  — Clear an index
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from codecontext.core.context import Context
from codecontext.core.embedding import create_embedding
from codecontext.core.reranker import Reranker
from codecontext.core.simhash import compute_simhash, simhash_similarity
from codecontext.core.splitter import AstSplitter
from codecontext.core.types import Config
from codecontext.core.vectordb import FaissVectorDB

logger = logging.getLogger("codecontext.server")

# Server state
_ctx: Context | None = None
_cfg: Config | None = None

# SimHash registry: {workspace_id: {"simhash": str, "collection": str, "updated": float}}
_simhash_registry: dict[str, dict[str, Any]] = {}
_simhash_registry_path: Path | None = None

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

        # Compute SimHash for this workspace
        _register_simhash(workspace_id, staging_dir)

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
        simhash = _simhash_registry.get(ws_id, {}).get("simhash", "")
        workspaces.append({
            "workspace_id": ws_id,
            "collection": col_name,
            "status": status,
            "simhash": simhash,
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
    _simhash_registry.pop(workspace_id, None)
    _save_simhash_registry()

    # Clean up staging directory
    shutil.rmtree(staging_dir, ignore_errors=True)
    _upload_staging.pop(workspace_id, None)

    return JSONResponse({"workspace_id": workspace_id, "status": "cleared"})


async def simhash_lookup(request: Request) -> JSONResponse:
    """Find the most similar existing index by SimHash.

    Expects JSON body:
    {
        "simhash": "abc123...",
        "threshold": 0.85
    }

    Returns the most similar workspace if similarity >= threshold,
    enabling index reuse instead of re-indexing from scratch.
    """
    body = await request.json()
    query_hash = body.get("simhash")
    threshold = body.get("threshold", 0.85)

    if not query_hash:
        return JSONResponse({"error": "simhash required"}, status_code=400)

    best_match = None
    best_similarity = 0.0

    for ws_id, entry in _simhash_registry.items():
        sim = simhash_similarity(query_hash, entry["simhash"])
        if sim > best_similarity:
            best_similarity = sim
            best_match = ws_id

    if best_match and best_similarity >= threshold:
        return JSONResponse({
            "match": True,
            "workspace_id": best_match,
            "similarity": round(best_similarity, 4),
            "collection": _simhash_registry[best_match].get("collection", ""),
        })
    else:
        return JSONResponse({
            "match": False,
            "best_similarity": round(best_similarity, 4) if best_match else 0,
        })


async def register_simhash(request: Request) -> JSONResponse:
    """Register a SimHash for a workspace (called by sync client)."""
    body = await request.json()
    workspace_id = body.get("workspace_id")
    simhash = body.get("simhash")

    if not workspace_id or not simhash:
        return JSONResponse(
            {"error": "workspace_id and simhash required"}, status_code=400
        )

    staging_dir = _upload_staging.get(workspace_id, "")
    ctx = _get_ctx()
    col_name = ctx.get_collection_name(staging_dir) if staging_dir else ""

    _simhash_registry[workspace_id] = {
        "simhash": simhash,
        "collection": col_name,
        "updated": time.time(),
    }
    _save_simhash_registry()

    return JSONResponse({"workspace_id": workspace_id, "registered": True})


# ---------------------------------------------------------------------------
# SimHash registry persistence
# ---------------------------------------------------------------------------

def _register_simhash(workspace_id: str, staging_dir: str) -> None:
    """Compute and register SimHash after indexing."""
    from codecontext.core.simhash import compute_simhash_from_directory
    from codecontext.core.types import DEFAULT_SUPPORTED_EXTENSIONS, DEFAULT_IGNORE_PATTERNS

    ctx = _get_ctx()
    simhash = compute_simhash_from_directory(
        staging_dir, DEFAULT_SUPPORTED_EXTENSIONS, DEFAULT_IGNORE_PATTERNS
    )
    col_name = ctx.get_collection_name(staging_dir)
    _simhash_registry[workspace_id] = {
        "simhash": simhash,
        "collection": col_name,
        "updated": time.time(),
    }
    _save_simhash_registry()
    logger.info("SimHash registered for %s: %s", workspace_id, simhash[:16])


def _save_simhash_registry() -> None:
    if _simhash_registry_path:
        _simhash_registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_simhash_registry_path, "w") as f:
            json.dump(_simhash_registry, f, indent=2)


def _load_simhash_registry() -> None:
    global _simhash_registry
    if _simhash_registry_path and _simhash_registry_path.exists():
        with open(_simhash_registry_path) as f:
            _simhash_registry = json.load(f)
        logger.info("Loaded %d SimHash entries", len(_simhash_registry))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(cfg: Config | None = None) -> Starlette:
    """Create the Starlette ASGI app for the index server."""
    global _ctx, _cfg, _simhash_registry_path

    _cfg = cfg or Config.from_env()
    _simhash_registry_path = Path(_cfg.data_dir) / "simhash_registry.json"

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

    _load_simhash_registry()

    routes = [
        Route("/api/upload", upload_files, methods=["POST"]),
        Route("/api/upload-json", upload_files_json, methods=["POST"]),
        Route("/api/index", trigger_index, methods=["POST"]),
        Route("/api/search", search, methods=["POST"]),
        Route("/api/status/{workspace_id}", get_status, methods=["GET"]),
        Route("/api/collections", list_collections, methods=["GET"]),
        Route("/api/clear/{workspace_id}", clear, methods=["DELETE"]),
        Route("/api/simhash", simhash_lookup, methods=["POST"]),
        Route("/api/simhash/register", register_simhash, methods=["POST"]),
    ]

    app = Starlette(routes=routes)
    logger.info("Index server initialized with %d routes", len(routes))
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
