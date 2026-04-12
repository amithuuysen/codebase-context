"""
codecontext.local — Local MCP server for semantic code search.

Contains the MCP tools, snapshot manager, background sync, and handlers
for standalone local indexing (no remote server required).
"""

from .server import main

__all__ = ["main"]
