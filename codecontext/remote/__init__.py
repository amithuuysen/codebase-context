"""
codecontext.remote — Remote index server and proxy client.

Contains:
  - index_server: HTTP API for centralized indexing with SimHash-based
    index reuse across team members (Architecture §4)
  - sync_client: Uploads local files to the remote server
  - remote_search: Search proxy for forwarding queries
  - server: MCP proxy server (when INDEX_SERVER_URL is set)
"""

from .index_server import create_app, run_server
from .sync_client import SyncClient

__all__ = ["create_app", "run_server", "SyncClient"]
