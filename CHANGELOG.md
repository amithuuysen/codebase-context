# Changelog

All notable changes to CodeContext are documented in this file.

## [0.3.0] — 2026-04-07

### Performance — Pipelined Indexing Engine

This release rewrites the indexing pipeline for significantly faster codebase indexing, especially on Apple Silicon (M4 Pro) and large codebases (20K+ files).

#### Added

- **Producer/consumer pipeline** — File splitting (CPU) and embedding (GPU/API) now run concurrently via `asyncio.Queue`. While batch N is embedding, batch N+1's files are being read and AST-split in parallel. Backpressure via `maxsize=4` keeps memory bounded.
- **Concurrent embedding sub-batches** — Each embedding batch is split into ~4 sub-batches and sent to the embedding provider in parallel threads. Set `OLLAMA_NUM_PARALLEL=4` on the Ollama server for maximum GPU utilization.
- **Deferred FAISS persistence** — FAISS index is written to disk once at the end of indexing instead of after every batch flush, eliminating hundreds of redundant disk writes on large codebases.
- **Embedding cache** — SHA-256 content hash → embedding vector cache. On re-index, unchanged chunks skip the embedding API entirely. Cache saves periodically (every 10 batches) to prevent data loss on long runs.
- **Path obfuscation** — HMAC-SHA256 per path segment with auto-generated 32-byte key. Encrypted paths stored in metadata alongside plain paths for privacy.
- **Pipeline timing instrumentation** — Logs wall time, split time, embed time, overlap (pipeline gain), and throughput (chunks/s) after every indexing run. MCP response includes timing fields for programmatic access.
- **Adaptive thread pool** — Worker count scales to CPU core count (up to 14 for M4 Pro), up from the previous cap of 8.

#### Changed

- `embedding_batch_size` default: `10` → `100` (10× fewer API round-trips)
- `chunk_size` default: `1500` → `3000` chars (better context per chunk)
- `chunk_overlap` default: `200` → `300` chars
- Ollama, HuggingFace, and sentence-transformers moved from optional to mandatory dependencies

#### Removed

- Unused `_cache_new_embeddings()` method (caching is now inline in `_flush_buffer`)

### Pipeline Architecture

```
Producer (thread pool)              Consumer (async)
┌──────────────────────┐           ┌──────────────────────┐
│ Read files           │           │ Check embedding cache │
│ AST split (tree-     │  Queue    │ Embed uncached chunks │
│   sitter, parallel)  │ ───────▶ │ Insert into FAISS     │
│ Push chunk batches   │ maxsize=4 │ Add to BM25 index     │
└──────────────────────┘           └──────────────────────┘
                                            │
                                   FAISS persist (once at end)
```

---

## [0.2.0] — 2026-04-07

### Initial Release — Hybrid Semantic Code Search

First public release of CodeContext, a Python MCP server implementing Cursor IDE's hybrid search architecture.

#### Core Features

- **Hybrid retrieval** — FAISS dense vector search + BM25 sparse keyword search, fused via Reciprocal Rank Fusion (RRF, k=60). 12.5% accuracy improvement over pure semantic search.
- **AST-aware chunking** — Tree-sitter parses code into functions, classes, and methods for 9 languages (Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, and more). Oversized chunks use line-by-line splitting with configurable overlap.
- **Merkle tree sync** — Directory-aware change detection using SHA-256 hash trees. Only walks branches where hashes diverge — O(changes) instead of O(files).
- **Cross-encoder reranker** — Optional Stage 2 precision refinement on top-N candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **MCP server** — 4 tools (`index_codebase`, `search_code`, `clear_index`, `get_indexing_status`) over stdio or streamable-http. Compatible with VS Code Copilot, Claude Desktop, and any MCP client.
- **Background sync** — Automatic 5-minute periodic file-change sync via `SyncManager`.
- **Multi-provider embeddings** — OpenAI, Ollama (local), or HuggingFace sentence-transformers.
- **Parallel file splitting** — `ThreadPoolExecutor` for concurrent file reading and AST splitting.

#### MCP Tools

| Tool | Description |
|---|---|
| `index_codebase` | Index a directory — scan → AST split → embed → FAISS + BM25 |
| `search_code` | Hybrid search — FAISS + BM25 → RRF → optional rerank |
| `clear_index` | Drop FAISS + BM25 indices and sync state for a codebase |
| `get_indexing_status` | Check indexing progress, file/chunk counts, error state |

#### Data Storage

All data persists locally under `~/.context/` — FAISS indices, BM25 inverted indices, Merkle tree snapshots, and indexing state.

#### Supported Languages

Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, C#, PHP, Ruby, Swift, Kotlin, Scala, Objective-C, Markdown.
