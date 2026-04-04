"""
codecontext — Semantic code search via MCP protocol.

Pipeline: Codebase → Tree-sitter (AST) → LlamaIndex TextNode → FAISS

A Python reimplementation of zilliztech/claude-context that stores all
vectors locally via FAISS instead of requiring Milvus/Zilliz Cloud.
Exposes the same MCP tools so any AI agent can use it.

Package layout mirrors the TypeScript original:
  codecontext.core  — Core library (context, types, embedding, splitter, sync, vectordb)
  codecontext.mcp   — MCP server (handlers, snapshot, sync, utils, server entry point)
"""
