import os as _os

# Prevent "OMP: Error #15" on macOS where faiss-cpu and PyTorch each ship
# their own copy of libomp.  Must be set before either library is imported.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

"""
codecontext — Semantic code search via MCP protocol.

Pipeline: Codebase → Tree-sitter (AST) → LlamaIndex TextNode → FAISS

A Python reimplementation of zilliztech/claude-context that stores all
vectors locally via FAISS instead of requiring Milvus/Zilliz Cloud.
Exposes the same MCP tools so any AI agent can use it.

Package layout:
  codecontext.core    — Core library (context, types, embedding, splitter, sync, vectordb)
  codecontext.local   — Local MCP server (handlers, snapshot, sync, server entry point)
  codecontext.remote  — Remote index server + proxy client (index_server, sync_client, proxy)
"""
