"""
Sync Client — sends local files to a remote CodeContext index server.

Flow:
  1. Check if already indexed on server
  2. If not indexed or force → upload all files
  3. Trigger remote indexing (server's Merkle tree skips unchanged files)
  4. Search via server API

Usage:
    client = SyncClient("http://index-server:8878", "/path/to/codebase")
    await client.sync()            # Upload files + trigger indexing
    results = await client.search("how does auth work")
"""

from __future__ import annotations

import asyncio
import hashlib
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
        """Upload local files to remote server and trigger indexing.

        Skips upload if already indexed (unless force=True).
        Server's Merkle tree handles per-file diff during indexing.

        Args:
            force: Force re-index even if already indexed.

        Returns:
            Dict with sync results.
        """
        logger.info("Starting sync for %s → %s", self.codebase_path, self.server_url)

        # Step 1: Check if already indexed on server
        remote_status = await self.get_status()
        already_indexed = remote_status.get("status") == "indexed"

        if already_indexed and not force:
            logger.info("Codebase already indexed on server. Skipping upload.")
            return {
                "workspace_id": self.workspace_id,
                "files_uploaded": 0,
                "skipped": True,
                "reason": "Already indexed on server — use force=True to re-index",
                "index_status": remote_status,
            }

        if remote_status.get("status") == "indexing":
            logger.info("Codebase is currently being indexed. Skipping upload.")
            return {
                "workspace_id": self.workspace_id,
                "files_uploaded": 0,
                "skipped": True,
                "reason": "Indexing already in progress",
                "index_status": remote_status,
            }

        # Step 2: Collect and upload files
        files = self._collect_files()
        logger.info("Uploading %d files to server...", len(files))
        upload_result = await self._upload_files(files)
        logger.info("Upload complete: %s", upload_result)

        # Step 3: Register SimHash for index reuse (Architecture §4)
        # Allows server to copy a similar teammate's index as starting point
        simhash = compute_simhash_from_directory(
            self.codebase_path, supported_extensions=self.extensions
        )
        try:
            await self._register_simhash(simhash)
            logger.info("SimHash registered: %s", simhash[:16])
        except Exception as exc:
            logger.warning("SimHash registration failed (non-fatal): %s", exc)

        # Step 4: Trigger indexing
        index_result = await self._trigger_index(force)
        logger.info("Indexing triggered: %s", index_result)

        return {
            "workspace_id": self.workspace_id,
            "files_uploaded": len(files),
            "skipped": False,
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
        """Upload files to server in parallel batches (JSON endpoint)."""
        batch_size = 100  # files per request
        max_concurrent = 4  # parallel uploads

        rel_paths = list(files.keys())
        batches = []
        for i in range(0, len(rel_paths), batch_size):
            batch_paths = rel_paths[i:i + batch_size]
            batches.append({p: files[p] for p in batch_paths})

        sem = asyncio.Semaphore(max_concurrent)
        total_uploaded = 0
        total_files = len(files)
        lock = asyncio.Lock()

        async def upload_batch(batch: dict[str, str]) -> None:
            nonlocal total_uploaded
            async with sem:
                resp = await self._client.post(
                    f"{self.server_url}/api/upload-json",
                    json={
                        "workspace_id": self.workspace_id,
                        "files": batch,
                    },
                )
                resp.raise_for_status()
                async with lock:
                    total_uploaded += len(batch)
                    logger.info("Uploaded: %d/%d files", total_uploaded, total_files)

        await asyncio.gather(*[upload_batch(b) for b in batches])

        return {"total_uploaded": total_uploaded}

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

    async def _register_simhash(self, simhash: str) -> dict[str, Any]:
        """Register this workspace's SimHash on the server for index reuse."""
        resp = await self._client.post(
            f"{self.server_url}/api/register-simhash",
            json={
                "workspace_id": self.workspace_id,
                "simhash": simhash,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def find_similar(self) -> dict[str, Any]:
        """Check if a similar index exists on the server.

        Returns match info including workspace_id and similarity score,
        or None if no match above threshold.
        """
        resp = await self._client.post(
            f"{self.server_url}/api/find-similar",
            json={"workspace_id": self.workspace_id},
        )
        resp.raise_for_status()
        return resp.json()
