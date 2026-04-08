"""
Sync Client — sends local files to a remote CodeContext index server.

Flow:
  1. Compute SimHash of local codebase
  2. Check server for a similar existing index (SimHash lookup)
  3. If similar index found → only send changed files (Merkle diff)
  4. If no match → send all files
  5. Trigger remote indexing
  6. Search via server API

Usage:
    client = SyncClient("http://index-server:8878", "/path/to/codebase")
    await client.sync()            # Upload files + trigger indexing
    results = await client.search("how does auth work")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

from codecontext.core.simhash import compute_simhash_from_directory
from codecontext.core.types import DEFAULT_IGNORE_PATTERNS, DEFAULT_SUPPORTED_EXTENSIONS

logger = logging.getLogger("codecontext.client")


class SyncClient:
    """Client that syncs a local codebase to a remote index server."""

    def __init__(
        self,
        server_url: str,
        codebase_path: str,
        workspace_id: str | None = None,
        supported_extensions: set[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.codebase_path = os.path.abspath(codebase_path)
        self.workspace_id = workspace_id or self._default_workspace_id()
        self.extensions = supported_extensions or DEFAULT_SUPPORTED_EXTENSIONS
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS
        self._client = httpx.AsyncClient(timeout=300.0)

    def _default_workspace_id(self) -> str:
        """Generate workspace ID from codebase path."""
        h = hashlib.md5(self.codebase_path.encode()).hexdigest()[:8]
        name = os.path.basename(self.codebase_path)
        return f"{name}_{h}"

    async def sync(self, force: bool = False) -> dict[str, Any]:
        """Full sync: compute SimHash → check for similar index → upload files → index.

        Args:
            force: Force re-index even if a similar index exists.

        Returns:
            Dict with sync results.
        """
        logger.info("Starting sync for %s → %s", self.codebase_path, self.server_url)

        # Step 1: Compute local SimHash
        logger.info("Computing SimHash...")
        local_simhash = compute_simhash_from_directory(
            self.codebase_path, self.extensions, self.ignore_patterns
        )
        logger.info("Local SimHash: %s", local_simhash[:16])

        # Step 2: Check server for similar index
        similar = await self._find_similar_index(local_simhash)
        if similar and not force:
            logger.info(
                "Found similar index: workspace=%s, similarity=%.1f%%",
                similar["workspace_id"],
                similar["similarity"] * 100,
            )
            # Could optimize: only send divergent files
            # For now, still send all and let server handle diff

        # Step 3: Collect and upload files
        files = self._collect_files()
        logger.info("Uploading %d files to server...", len(files))

        upload_result = await self._upload_files(files)
        logger.info("Upload complete: %s", upload_result)

        # Step 4: Register SimHash
        await self._register_simhash(local_simhash)

        # Step 5: Trigger indexing
        index_result = await self._trigger_index(force)
        logger.info("Indexing triggered: %s", index_result)

        return {
            "workspace_id": self.workspace_id,
            "files_uploaded": len(files),
            "simhash": local_simhash,
            "similar_index": similar,
            "index_status": index_result,
        }

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search the remote index."""
        resp = await self._client.post(
            f"{self.server_url}/api/search",
            json={
                "workspace_id": self.workspace_id,
                "query": query,
                "limit": limit,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    async def get_status(self) -> dict[str, Any]:
        """Get indexing status from server."""
        resp = await self._client.get(
            f"{self.server_url}/api/status/{self.workspace_id}"
        )
        if resp.status_code == 404:
            return {"status": "not_found"}
        resp.raise_for_status()
        return resp.json()

    async def wait_for_indexing(self, poll_interval: float = 2.0) -> dict[str, Any]:
        """Poll server until indexing is complete."""
        while True:
            status = await self.get_status()
            s = status.get("status", "unknown")
            if s in ("indexed", "failed", "not_found"):
                return status
            progress = status.get("progress", 0)
            logger.info("Indexing in progress: %d%%", progress)
            await asyncio.sleep(poll_interval)

    async def clear(self) -> dict[str, Any]:
        """Clear the remote index for this workspace."""
        resp = await self._client.delete(
            f"{self.server_url}/api/clear/{self.workspace_id}"
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _collect_files(self) -> dict[str, str]:
        """Walk codebase and collect file contents as {relative_path: content}."""
        from fnmatch import fnmatch

        files: dict[str, str] = {}
        root = Path(self.codebase_path)

        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            rel_dir = os.path.relpath(dirpath, root)
            skip_dir = False
            for pat in self.ignore_patterns:
                if fnmatch(rel_dir, pat) or fnmatch(os.path.basename(dirpath), pat):
                    skip_dir = True
                    break
            if skip_dir:
                dirnames.clear()
                continue

            for fname in filenames:
                if fname.startswith("."):
                    continue
                ext = os.path.splitext(fname)[1]
                if ext not in self.extensions:
                    continue

                rel_path = os.path.relpath(os.path.join(dirpath, fname), root)
                skip_file = False
                for pat in self.ignore_patterns:
                    if fnmatch(rel_path, pat) or fnmatch(fname, pat):
                        skip_file = True
                        break
                if skip_file:
                    continue

                fpath = os.path.join(dirpath, fname)
                try:
                    content = Path(fpath).read_text(errors="replace")
                    files[rel_path] = content
                except (OSError, PermissionError):
                    continue

        return files

    async def _upload_files(self, files: dict[str, str]) -> dict[str, Any]:
        """Upload files to server in batches (JSON endpoint)."""
        batch_size = 100  # files per request
        total = 0

        rel_paths = list(files.keys())
        for i in range(0, len(rel_paths), batch_size):
            batch_paths = rel_paths[i:i + batch_size]
            batch = {p: files[p] for p in batch_paths}

            resp = await self._client.post(
                f"{self.server_url}/api/upload-json",
                json={
                    "workspace_id": self.workspace_id,
                    "files": batch,
                },
            )
            resp.raise_for_status()
            total += len(batch)
            logger.info("Uploaded batch: %d/%d files", total, len(files))

        return {"total_uploaded": total}

    async def _find_similar_index(self, simhash: str) -> dict[str, Any] | None:
        """Check server for a similar existing index."""
        try:
            resp = await self._client.post(
                f"{self.server_url}/api/simhash",
                json={"simhash": simhash, "threshold": 0.85},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("match"):
                return data
        except Exception as exc:
            logger.debug("SimHash lookup failed: %s", exc)
        return None

    async def _register_simhash(self, simhash: str) -> None:
        """Register local SimHash with the server."""
        try:
            await self._client.post(
                f"{self.server_url}/api/simhash/register",
                json={
                    "workspace_id": self.workspace_id,
                    "simhash": simhash,
                },
            )
        except Exception as exc:
            logger.debug("SimHash registration failed: %s", exc)

    async def _trigger_index(self, force: bool = False) -> dict[str, Any]:
        """Trigger indexing on the server."""
        resp = await self._client.post(
            f"{self.server_url}/api/index",
            json={
                "workspace_id": self.workspace_id,
                "force": force,
            },
        )
        resp.raise_for_status()
        return resp.json()
