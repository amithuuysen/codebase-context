# Changelog

All notable changes to CodeContext are documented in this file.

## [0.6.0] — 2026-04-13

### Indexing Speed Optimizations + RAG Evaluation

Major indexing speed improvements — **5.7x faster** (9 files/s → 51 files/s on 21K-file codebase).

#### Added — Indexing Speed Optimizations

- **Removed Ollama sub-batching** — sends all chunks in a single HTTP call per batch instead of splitting into 4 sub-batches. Saves ~400ms per batch × 2,100 batches = ~14 min on large codebases.
- **Queue backpressure: 8 → 32** — more producer-consumer overlap, less idle time on either side.
- **Thread pool: 14 → 24 workers** — file reading + AST parsing is I/O-bound; more workers keeps the pipeline fed.
- **Cache save interval: 10 → 50 batches** — 5x fewer SQLite transactions during indexing.
- **Periodic FAISS persist** — persists every 50 batches instead of only at the end, enabling crash recovery and smoother I/O.
- **Configurable `embed_batch_size`** — Ollama embedding batch size now reads `EMBEDDING_BATCH_SIZE` env var (default 100).

#### Added — RAG Evaluation

- **`eval_rag.py`** — RAG accuracy evaluation script with 20 domain-specific queries (10 semantic, 5 keyword, 5 hybrid). Metrics: Hit Rate @K, MRR @K, Recall @K. Supports `--index-only`, `--skip-index`, `--top-k N` flags.
- **Live progress** — shows files/s rate and ETA during indexing.

#### Changed

- **Chunk size: 1500 → 2500** — fewer chunks per file, fewer embedding API calls. Well within `nomic-embed-text` 8K token context.
- **Chunk overlap: 200 → 250** — proportional increase with chunk size.

## [0.5.0] — 2026-04-12

### Package Restructure + Performance Optimizations + Index Reuse

This release splits the package into `local/` and `remote/` modules, implements SimHash-based index reuse for teams, and adds major indexing performance optimizations.

#### Package Structure

Reorganized from `mcp/`, `server/`, `client/` into two clear packages:

```
codecontext/
├── core/          # Shared search library (unchanged)
├── local/         # Local MCP server (standalone single-developer mode)
│   ├── server.py      # MCP tools — local indexing only
│   ├── handlers.py    # background_indexing
│   ├── snapshot.py    # SnapshotManager (V2 format, is_searchable at 80%)
│   ├── sync.py        # SyncManager (5-min background re-index)
│   └── utils.py       # CLI helpers
└── remote/        # Remote server + proxy client (team mode)
    ├── index_server.py   # HTTP API + SimHash registry + index reuse
    ├── server.py         # MCP proxy (when INDEX_SERVER_URL is set)
    ├── sync_client.py    # Upload client with SimHash registration
    └── remote_search.py  # Search proxy
```

- `local/server.py` handles all local indexing — zero remote/SimHash logic
- When `INDEX_SERVER_URL` is set, `local/server.py:main()` delegates to `remote/server.py`
- Entry points updated: `codecontext` → `codecontext.local.server:main`, `codecontext-server` → `codecontext.remote.index_server:run_server`

#### Added — SimHash Index Reuse (Architecture §4)

- **SimHash computation** (`core/simhash.py`) — 128-bit locality-sensitive hash from file content hashes. Teammates on the same branch share ~92% similarity.
- **SimHash registry** on the index server — persisted to `simhash_registry.json`. New endpoints: `POST /api/register-simhash`, `POST /api/find-similar`.
- **Automatic index reuse** — when a new workspace is indexed, the server finds the most similar existing index (≥85% threshold), copies FAISS + BM25 files, then runs incremental indexing only on differing files. Cursor's data: median 7.87s → 525ms.
- **SyncClient SimHash registration** — after uploading files, the client registers its SimHash so future teammates benefit from the index.
- **`copy_collection()`** on `FaissVectorDB` — copies FAISS files + metadata on disk for team reuse.

#### Added — Performance Optimizations

- **Embedding cache: JSON → SQLite** — WAL mode, 3-tier lookup (pending → in-memory read-through → SQLite), `get_batch()` bulk queries (batches of 900 for bind param limit). Startup: instant vs 30+ seconds for 1.2 GB JSON. Auto-migrates from legacy JSON.
- **BM25: JSON → pickle** — 5-10x faster serialization for 900 MB+ indices. Auto-migrates from legacy JSON.
- **Parallel Merkle tree hashing** — `ThreadPoolExecutor` (up to 16 workers) for repos with 100+ files. Serial for small repos to avoid pool overhead.
- **Pre-compiled ignore patterns** — fnmatch patterns compiled into a single regex for fast matching during file walks.
- **Early search at 80%** — `SnapshotManager.is_searchable()` returns true when indexing is ≥80% complete.
- **Queue backpressure tuning** — `asyncio.Queue` maxsize increased 4 → 8 to keep the pipeline fed.
- **Adaptive embedding sub-batching** — reads `OLLAMA_NUM_PARALLEL` env var to match sub-batch count to server capacity.

#### Changed

- **Entry points** — `codecontext` → `codecontext.local.server:main`, `codecontext-server` → `codecontext.remote.index_server:run_server`
- **`_flush_buffer()` refactored** — Phase 1 builds all TextNodes and collects hashes, Phase 2 does bulk cache lookup via `get_batch()`, then splits into cached vs needs-embedding. Eliminates per-chunk SQL overhead.

---

## [0.4.0] — 2026-04-08

### Remote Indexing

This release adds a standalone HTTP index server, enabling remote indexing and search. Upload files from your local machine to a server, which handles embedding, indexing, and search — no local GPU required on the client.

#### Added

- **HTTP index server** (`codecontext-server`) — Starlette ASGI app with 7 REST endpoints: upload files (JSON batch), trigger indexing, search, check status, list collections, clear index. Run via `uv run codecontext-server` on `0.0.0.0:8878`.
- **Sync client** (`SyncClient`) — Collects local files, uploads in batches of 100 via JSON, triggers remote indexing, and supports remote search. Skips upload if already indexed on server. Uses `httpx.AsyncClient` with 300s timeout.
- **Remote search proxy** (`RemoteSearchProxy`) — Bridge between MCP server and remote index server. When `INDEX_SERVER_URL` env var is set, the MCP `search_code` tool forwards queries to the remote server instead of local FAISS.
- **Auto-index on first search** — In proxy mode, `search_code` automatically uploads and indexes the codebase on first use if not already indexed on the remote server.
- **`codecontext-server` entry point** — New CLI command to start the index server standalone.

#### Changed

- **Default embedding provider** — Changed from `openai` to `ollama` with `nomic-embed-text`. No env vars needed for a working setup — just ensure Ollama is running.
- **New dependencies** — Added `starlette>=0.27.0`, `uvicorn>=0.20.0`, `httpx>=0.24.0` as mandatory dependencies.

#### Architecture

```
Local Machine                    Index Server (remote :8878)
┌──────────────────┐   files    ┌──────────────────────┐
│ VS Code + MCP    │───────────▶│ FAISS + BM25         │
│ codecontext      │   JSON     │ Ollama embedding     │
│ (proxy mode)     │◀───search──│ Merkle tree sync     │
│                  │   results  │                      │
└──────────────────┘            └──────────────────────┘
```

---

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
