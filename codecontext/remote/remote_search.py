"""
Remote Search Proxy — used by MCP server when INDEX_SERVER_URL is set.

Instead of querying local FAISS, forwards search requests to the remote
index server. This lets developers run a thin MCP server locally while
the heavy indexing happens on a shared server.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from codecontext.core.types import SemanticSearchResult

logger = logging.getLogger("codecontext.client")


class RemoteSearchProxy:
    """Proxy search requests to a remote CodeContext index server."""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 10,
    ) -> list[SemanticSearchResult]:
        """Search via remote index server."""
        resp = await self._client.post(
            f"{self.server_url}/api/search",
            json={
                "workspace_id": workspace_id,
                "query": query,
                "limit": top_k,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        return [
            SemanticSearchResult(
                content=r["content"],
                relative_path=r["relative_path"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                language=r["language"],
                score=r.get("score", 0.0),
            )
            for r in data.get("results", [])
        ]

    async def get_status(self, workspace_id: str) -> dict[str, Any]:
        """Get indexing status from remote server."""
        resp = await self._client.get(
            f"{self.server_url}/api/status/{workspace_id}"
        )
        if resp.status_code == 404:
            return {"status": "not_found"}
        resp.raise_for_status()
        return resp.json()

    async def trigger_index(
        self, workspace_id: str, force: bool = False
    ) -> dict[str, Any]:
        """Trigger indexing on remote server."""
        resp = await self._client.post(
            f"{self.server_url}/api/index",
            json={"workspace_id": workspace_id, "force": force},
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()
