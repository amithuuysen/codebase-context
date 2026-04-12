# CodeContext — Hybrid Semantic Code Search (Python MCP Server)

## Problem Statement

Modern AI coding agents (Claude, Copilot, Cursor) need to understand entire codebases — not just the file currently open. The fundamental challenge is **retrieval**: given a natural-language query like *"where do we handle authentication?"*, find the most relevant code across thousands of files in milliseconds.

Existing approaches have clear limitations:

| Approach | Limitation |
|---|---|
| **Pure grep / regex** | Cannot match by *meaning* — fails when the query uses different words than the code (e.g. "authentication" vs `session.ts`) |
| **Pure semantic search** (embeddings) | Misses exact keyword matches — a query for `PaymentService` may return conceptually similar but wrong results |
| **Flat file-hash sync** | Re-scans every file on every sync pass — O(total files) even when only 3 files changed in a 50K-file repo |
| **Single-stage retrieval** | Bi-encoder embeddings are fast but imprecise; no mechanism to refine the initial ranking |
| **GitHub Copilot (local index)** | VS Code's local workspace index is limited to **~2,500 indexable files**. Larger codebases fall back to non-semantic tools (grep, file search) unless a remote index is available |
| **GitHub Copilot (remote index)** | Remote index only works for repos on **GitHub.com** or **GitHub Enterprise Cloud** — not supported for GitHub Enterprise Server, self-hosted Git, or non-GitHub repos. Index builds can silently fail or stall on very large repositories, and "External Ingest" for non-GitHub/Azure DevOps code requires a **paid Copilot subscription** |
| **Cursor IDE** | Hybrid search architecture is excellent (12.5% accuracy improvement), but requires a **paid subscription** ($20/mo Pro, $40/mo Business) — the underlying search technology is proprietary and not available as a standalone tool |

### Why existing IDE indexing falls short

**GitHub Copilot's indexing limitations** are well-documented in community discussions ([GitHub Discussion #152490](https://github.com/orgs/community/discussions/152490), [VS Code docs](https://code.visualstudio.com/docs/copilot/workspace-context)):

- **Local index cap (~2,500 files):** For workspaces exceeding ~2,500 indexable files, Copilot cannot build a local semantic index. It falls back to text search, grep, and file search — losing the ability to find code by *meaning*. Enterprise codebases with 10K–100K+ files are left without semantic search entirely unless they use the remote index.
- **Remote index requires GitHub.com hosting:** The remote index is built from the repository's default branch on GitHub.com or GitHub Enterprise Cloud. Repos hosted on GitHub Enterprise Server, GitLab, Bitbucket, or local Git servers **cannot use remote indexing at all**. For non-GitHub/Azure DevOps repos, an "External Ingest" feature exists but requires a paid Copilot subscription and is still gradually rolling out.
- **Remote index build reliability:** Initial indexing can take up to 60 seconds for large repositories. Community reports indicate index builds can silently fail, stall, or produce incomplete results — with limited diagnostic visibility (the status bar shows "indexed" even when indexing is partial).
- **No transparency on index contents:** The local index is stored in VS Code's workspace storage (`~/Library/Application Support/Code/User/workspaceStorage/` on macOS) under `GitHub.copilot-chat`, but file names and format are undocumented and can change between versions. Users have reported unexpected large uploads (~500MB) of workspace content to `api.github.com` during indexing.

**Cursor IDE** solves many of these problems with its hybrid semantic + keyword search architecture, but locks the solution behind a **paid subscription** ($20/mo minimum). The search technology is proprietary and tightly coupled to the Cursor editor — it cannot be used with other editors or as a standalone service.

Cursor IDE's research demonstrates that **combining semantic + keyword search improves accuracy by 12.5%** on average, and up to 23.5% on large codebases (1000+ files). Their architecture uses Merkle trees for O(changes) sync, hybrid retrieval with fusion, and a two-stage pipeline with reranking.

## Proposed Solution

A Python MCP server that replicates the key accuracy techniques from Cursor's architecture:

1. **Hybrid Retrieval (Semantic + Keyword)** — FAISS dense vector search (finds code by meaning) combined with BM25 sparse keyword search (finds exact matches), fused via Reciprocal Rank Fusion (RRF).

2. **Merkle Tree Sync** — Directory-aware change detection using cryptographic hash trees. Only walks branches where hashes diverge — unchanged subtrees are skipped entirely.

3. **Two-Stage Pipeline** — Stage 1: fast recall via FAISS + BM25 + RRF to get top-N candidates. Stage 2: optional cross-encoder reranker for precision refinement on top-N → final top-K.

4. **AST-Aware Chunking** — Tree-sitter parses code into an AST, splitting at logical boundaries (functions, classes) instead of arbitrary character offsets. Oversized chunks use line-by-line splitting with overlap.

5. **MCP Protocol** — Exposes 4 tools (`index_codebase`, `search_code`, `clear_index`, `get_indexing_status`) over stdio, compatible with Claude Desktop, VS Code Copilot, and any MCP client.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Codebase (on disk)                       │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────┐    ┌──────────────────────────────────────┐
│  Merkle Tree Sync   │───▶│  Detect added / modified / removed   │
│  (O(changes) diff)  │    │  Skip unchanged subtrees entirely    │
└─────────────────────┘    └──────────────────┬───────────────────┘
                                              │
                                              ▼
                           ┌──────────────────────────────────────┐
                           │  Tree-sitter AST Splitter            │
                           │  → functions, classes, interfaces    │
                           │  → line-by-line fallback for large   │
                           └──────────────────┬───────────────────┘
                                              │
                                              ▼
                           ┌──────────────────────────────────────┐
                           │  LlamaIndex TextNode (id, metadata)  │
                           └────────┬─────────────────┬───────────┘
                                    │                 │
                              ┌─────▼─────┐    ┌─────▼──────┐
                              │   FAISS    │    │   BM25     │
                              │ IndexFlatIP│    │ Inverted   │
                              │  (dense)   │    │  Index     │
                              │  cosine    │    │ (sparse)   │
                              └─────┬──────┘    └─────┬──────┘
                                    │                 │
                                    ▼                 ▼
                           ┌──────────────────────────────────────┐
                           │  RRF (Reciprocal Rank Fusion)        │
                           │  score = Σ 1/(k + rank_in_list)      │
                           │  Merges dense + sparse ranked lists  │
                           └──────────────────┬───────────────────┘
                                              │
                                              ▼
                           ┌──────────────────────────────────────┐
                           │  Cross-Encoder Reranker (optional)   │
                           │  Stage 2: precision on top-N → top-K │
                           └──────────────────┬───────────────────┘
                                              │
                                              ▼
                           ┌──────────────────────────────────────┐
                           │  MCP Server (stdio)                  │
                           │  4 tools: index, search, clear,      │
                           │           get_status                  │
                           └──────────────────────────────────────┘
```

### Why Hybrid > Pure Semantic

| Query Type | FAISS (dense) | BM25 (sparse) | Hybrid (RRF) |
|---|---|---|---|
| *"where do we handle authentication?"* | ✅ Finds `session.ts` by meaning | ❌ Word "authentication" absent | ✅ Semantic match |
| *"find all imports of PaymentService"* | ⚠️ May return similar but wrong | ✅ Exact keyword match | ✅ Keyword match |
| *"how does the tax calculation work?"* | ✅ Good conceptual match | ✅ Matches "tax" + "calculation" | ✅ Both signals fused → best ranking |

## Project Structure

```
codecontext/
├── __init__.py
├── core/                          # Core search library
│   ├── __init__.py                # Re-exports all public types/classes
│   ├── types.py                   # Config, CodeChunk, SemanticSearchResult, constants
│   ├── context.py                 # Orchestrator: index → hybrid search → rerank
│   ├── embedding.py               # Embedding factory (OpenAI / Ollama / local)
│   ├── embedding_cache.py         # SHA-256 content hash → embedding vector cache
│   ├── vectordb.py                # FAISS IndexFlatIP (dense vectors, cosine similarity)
│   ├── bm25.py                    # BM25 sparse keyword index (inverted index + IDF)
│   ├── hybrid_search.py           # RRF fusion (FAISS + BM25 → merged ranking)
│   ├── merkle.py                  # Merkle tree sync (directory-aware O(changes) diff)
│   ├── path_obfuscation.py        # HMAC-SHA256 path segment encryption for privacy
│   ├── reranker.py                # Cross-encoder reranker (optional Stage 2)
│   ├── simhash.py                 # SimHash locality-sensitive hashing
│   ├── sync.py                    # Flat file-hash sync (SHA-256 fallback)
│   └── splitter/
│       ├── __init__.py
│       ├── ast_splitter.py        # Tree-sitter AST splitter (functions, classes)
│       └── text_splitter.py       # Character-based fallback splitter
├── mcp/                           # MCP server layer
│   ├── __init__.py
│   ├── server.py                  # FastMCP + @mcp.tool() decorators + main()
│   ├── handlers.py                # background_indexing() async helper
│   ├── snapshot.py                # SnapshotManager (V2 format, indexing state)
│   ├── sync.py                    # SyncManager (background 5-min file-change sync)
│   └── utils.py                   # ensure_absolute, truncate, log_config, shutdown
├── server/                        # Remote index server (HTTP API)
│   ├── __init__.py
│   └── index_server.py            # Starlette ASGI app — upload, index, search
├── client/                        # Client for remote index server
│   ├── __init__.py
│   ├── sync_client.py             # SyncClient — upload files, trigger indexing, search
│   └── remote_search.py           # RemoteSearchProxy — MCP ↔ index server bridge
```

## Key Design Decisions

### 1. FAISS IndexFlatIP for Cosine Similarity
Vectors are L2-normalized before insertion. Inner product on normalized vectors equals cosine similarity, producing scores in [0, 1] where 1.0 = perfect match. Threshold default is 0.5 (matching the TypeScript/Milvus original).

### 2. BM25 with Code-Aware Tokenizer
The BM25 tokenizer splits on non-alphanumeric characters AND camelCase/snake_case boundaries (`processUserInput` → `["process", "user", "input"]`). Stop words are filtered. This produces much better keyword matching for source code than generic NLP tokenizers.

### 3. Reciprocal Rank Fusion (k=60)
RRF merges ranked lists without needing score normalization (FAISS cosine scores and BM25 IDF scores are on completely different scales). Each item's fused score is `Σ 1/(60 + rank)` across all lists. Items appearing in both lists are naturally boosted.

### 4. Merkle Tree vs Flat Scan
For a 50K-file repo where 3 files changed:
- **Flat scan**: hash all 50K files → compare → O(50K)
- **Merkle tree**: compare root hash → walk only divergent branches → O(log N + changes)

### 5. AST Splitting Matches TypeScript Original
Node types per language match the original TypeScript implementation exactly. Control flow statements (`if`, `for`, `while`, `try`) stay inside their parent function/class — they are NOT split into separate chunks. Gap text (imports, comments between functions) is discarded, matching the TS behavior.

## MCP Tools

| Tool | Description |
|---|---|
| `index_codebase` | Index a directory — scan → AST split → embed → FAISS + BM25 |
| `search_code` | Hybrid search — FAISS + BM25 → RRF → optional rerank |
| `clear_index` | Drop FAISS + BM25 indices and sync state for a codebase |
| `get_indexing_status` | Check indexing progress, file/chunk counts, error state |

## Data Storage

All data persists locally under `~/.context/`:

```
~/.context/
├── faiss_store/                       # Dense vector indices
│   └── code_chunks_<hash>/
│       ├── default__vector_store.faiss
│       ├── docstore.json
│       ├── index_store.json
│       └── collection_meta.json
├── bm25_store/                        # Sparse keyword indices
│   └── code_chunks_<hash>.pkl         # Binary pickle (5-10x faster than JSON)
├── embedding_cache/                   # Embedding vector cache (SQLite)
│   └── ollama_nomic-embed-text.db     # WAL mode, O(1) lookups, incremental writes
├── merkle/                            # Merkle tree snapshots
│   └── merkle_<hash>.json             # Directory-aware tree
├── simhash_registry.json              # SimHash fingerprints (team index reuse)eam index reuse)
├── path_obfuscation_key               # HMAC key (chmod 600)
└── mcp-codebase-snapshot.json         # Indexing state (V2 format)
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | `streamable-http` | `streamable-http`, `stdio`, or `sse` |
| `EMBEDDING_PROVIDER` | `ollama` | `openai`, `ollama`, or `local` |
| `EMBEDDING_MODEL` | auto | Model name (auto-selected per provider) |
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `OPENAI_BASE_URL` | — | Custom API endpoint |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `nomic-embed-text` | Ollama model |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | `3000` | Max characters per code chunk |
| `CHUNK_OVERLAP` | `300` | Overlap between chunks |
| `EMBEDDING_BATCH_SIZE` | `100` | Chunks per embedding API call |
| `CODECONTEXT_DATA_DIR` | `~/.context` | Data storage directory |
| `RERANKER_PROVIDER` | `none` | `none` or `local` (enables cross-encoder) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `INDEX_SERVER_URL` | — | Remote index server URL (enables proxy mode) |
| `PORT` | `8878` | Index server listen port |

## How to Run

### 1. Install

```bash
cd codebase-context
uv sync
```

### 2. Configure Embeddings

```bash
# Default: Ollama with nomic-embed-text (local, free, no API key needed)
# Just ensure Ollama is running: ollama serve

# Option A: OpenAI (requires API key)
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# Option B: Sentence-transformers (fully offline, no API key)
export EMBEDDING_PROVIDER=local
```

### 3. Enable Reranker (Optional)

```bash
export RERANKER_PROVIDER=local
export RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

### Mode A: Local Indexing (Single Developer)

Everything runs on your machine — embedding, FAISS, BM25, search. No network calls.

```
┌──────────────────────────────────────────────────┐
│                 Your Machine                      │
│                                                   │
│  VS Code / Claude Desktop                         │
│       │ MCP (stdio)                               │
│       ▼                                           │
│  codecontext (MCP server)                         │
│       │                                           │
│       ├── Ollama (nomic-embed-text)               │
│       ├── FAISS + BM25 index (~/.context/)        │
│       └── Merkle tree sync                        │
└──────────────────────────────────────────────────┘
```

**Run the MCP server:**
```bash
# Default: streamable-http on http://127.0.0.1:8877/mcp
uv run codecontext

# Or use stdio transport (for clients that spawn the process)
MCP_TRANSPORT=stdio uv run codecontext
```

**VS Code** (`.vscode/mcp.json`) — streamable-http:
```json
{
  "servers": {
    "codecontext": {
      "type": "http",
      "url": "http://127.0.0.1:8877/mcp"
    }
  }
}
```

**VS Code** (`.vscode/mcp.json`) — stdio:
```json
{
  "servers": {
    "codecontext": {
      "command": "uv",
      "args": ["run", "codecontext"],
      "env": {
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "codecontext": {
      "command": "uv",
      "args": ["run", "codecontext"],
      "env": {
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

> No `EMBEDDING_PROVIDER` or `OLLAMA_MODEL` needed — defaults to Ollama + nomic-embed-text. Just ensure `ollama serve` is running.

---

### Mode B: Remote Index Server

Index server runs on a separate machine. User sends files to it and searches remotely — no local GPU needed on the client.

```
Local Machine                    Index Server (remote :8878)
┌──────────────────┐   files    ┌──────────────────────┐
│ VS Code + MCP    │───────────▶│ FAISS + BM25         │
│ codecontext      │   JSON     │ Ollama embedding     │
│ (proxy mode)     │◀───search──│ Merkle tree sync     │
│                  │   results  │                      │
└──────────────────┘            └──────────────────────┘
```

#### Step 1: Start the index server (on your shared/remote machine)

```bash
# Install and run on the server
cd codebase-context
uv sync

# Start with Ollama (ensure ollama serve is running on the server)
uv run codecontext-server
# Listens on 0.0.0.0:8878

# Or with a custom port
PORT=9000 uv run codecontext-server

# Or with OpenAI embeddings on the server side
EMBEDDING_PROVIDER=openai OPENAI_API_KEY=sk-... uv run codecontext-server
```

#### Step 2: Upload and index from a client (programmatic)

```python
from codecontext.client import SyncClient

client = SyncClient("http://index-server:8878", "/path/to/codebase")
await client.sync()                              # Upload + index
results = await client.search("auth handler")    # Search remote
```

#### Step 3: Connect VS Code / Claude Desktop to the remote index

When `INDEX_SERVER_URL` is set, the MCP server becomes a thin proxy — it forwards `search_code` queries to the remote index server instead of running local FAISS.

**VS Code** (`.vscode/mcp.json`):
```json
{
  "servers": {
    "codecontext": {
      "command": "uv",
      "args": ["run", "codecontext"],
      "env": {
        "MCP_TRANSPORT": "stdio",
        "INDEX_SERVER_URL": "http://your-server:8878"
      }
    }
  }
}
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "codecontext": {
      "command": "uv",
      "args": ["run", "codecontext"],
      "env": {
        "MCP_TRANSPORT": "stdio",
        "INDEX_SERVER_URL": "http://your-server:8878"
      }
    }
  }
}
```

> **No Ollama needed on client machines.** The index server handles all embedding and search. Clients only need `uv` and the `codecontext` package installed.

#### Index Server API

| Endpoint | Method | Description |
|---|---|---|
| `/api/upload` | POST | Multipart file upload with `workspace_id` |
| `/api/upload-json` | POST | Upload files as JSON (batch of 100, path traversal protected) |
| `/api/index` | POST | Trigger background indexing for a workspace (`force` option) |
| `/api/search` | POST | Semantic search (`workspace_id`, `query`, `limit` max 50) |
| `/api/status/{workspace_id}` | GET | Check indexing progress |
| `/api/collections` | GET | List all indexed workspaces |
| `/api/clear/{workspace_id}` | DELETE | Delete a workspace's index + SimHash entry |
| `/api/register-simhash` | POST | Register/update workspace SimHash fingerprint |
| `/api/find-similar` | POST | Find similar existing index for reuse |

The index server also exposes MCP tools at `/mcp` — same 4 tools as local mode (`index_codebase`, `search_code`, `clear_index`, `get_indexing_status`).

---

## Architecture Components — What Each Piece Does

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INDEXING PIPELINE                              │
│                                                                         │
│  Codebase → Merkle Tree → File Discovery → AST Splitter → Chunks      │
│                                               │                         │
│                              ┌─────────────────┼──────────────────┐     │
│                              ▼                 ▼                  ▼     │
│                     Embedding Cache    FAISS (dense)    BM25 (sparse)  │
│                     (SQLite, SHA-256   (IndexFlatIP,    (inverted      │
│                      → vector)         cosine sim)      index, IDF)    │
│                                                                         │
│                              └─────────────────┼──────────────────┘     │
│                                                ▼                        │
│                                    RRF (Reciprocal Rank Fusion)         │
│                                    score = Σ 1/(60 + rank)              │
│                                                │                        │
│                                                ▼                        │
│                                    Cross-Encoder Reranker               │
│                                    (optional Stage 2)                   │
│                                                │                        │
│                                                ▼                        │
│                                         Search Results                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1. Merkle Tree Sync (`core/merkle.py`)

**What it does:** Detects which files changed since the last index — without re-scanning every file.

**How it works:**
- Leaf nodes = SHA-256 hash of each file's content
- Internal nodes = hash of children's hashes
- On re-index: compare root hashes. If they match → nothing changed. If they differ → walk only divergent branches

**Performance:**
- 50K-file repo with 3 changed files: O(log N + 3) vs O(50K) for flat scan
- Files are hashed in parallel via ThreadPool (up to 16 workers) for repos with 100+ files
- Ignore patterns are pre-compiled into a single regex for fast matching

**Problem it solves:** Without Merkle trees, every 5-minute sync pass would need to hash all 50K files just to find 3 changes. With Merkle trees, it compares a single root hash first.

### 2. AST Splitter (`core/splitter/ast_splitter.py`)

**What it does:** Splits code into logical chunks (functions, classes, methods) instead of arbitrary character offsets.

**How it works:**
- Tree-sitter parses code into an AST
- Extracts top-level declarations (functions, classes, interfaces, etc.)
- Supports 8 languages: Python, JavaScript, TypeScript, Java, Go, Rust, C, C++
- Falls back to character-based TextSplitter for unsupported languages
- Oversized chunks are split at line boundaries with overlap

**Problem it solves:** Character-based splitting breaks functions in half — the embedding for half a function is meaningless. AST splitting ensures each chunk is a complete logical unit.

### 3. Embedding Cache (`core/embedding_cache.py`)

**What it does:** Caches embedding vectors keyed by chunk content hash. If a chunk's text hasn't changed, skip the embedding API call entirely.

**How it works:**
- SQLite database with WAL mode for fast concurrent reads
- Three-tier lookup: pending buffer → in-memory read-through cache → SQLite
- Bulk `get_batch()` fetches hundreds of hashes in a single SQL query
- Vectors stored as compact binary (4 bytes per float32 via `struct.pack`)
- Auto-migrates from legacy JSON on first run (renames old file to `.json.bak`)

**Performance:**
- Startup: instant (no full cache load) vs 30+ seconds for 1.2 GB JSON
- Lookups: ~50ns (in-memory dict hit) vs ~50-100μs (SQLite miss)
- Saves: incremental (only dirty entries) vs full rewrite

**Problem it solves:** Embedding is the most expensive step. For re-indexing a 20K-file codebase where 50 files changed, the cache skips ~99.75% of embedding API calls.

### 4. FAISS Vector Database (`core/vectordb.py`)

**What it does:** Stores and searches dense embedding vectors via cosine similarity.

**How it works:**
- `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity)
- Per-collection indices: each codebase gets its own FAISS index
- Soft-delete: FAISS doesn't support deletion, so deleted doc IDs are tracked and filtered at query time
- `copy_collection()`: copies FAISS files on disk for team index reuse

**Problem it solves:** Finds code by *meaning* — a query for "authentication" matches `session.ts` even though the word "authentication" never appears in the file.

### 5. BM25 Keyword Index (`core/bm25.py`)

**What it does:** Classic keyword search using an inverted index with BM25 scoring.

**How it works:**
- Code-aware tokenizer: splits camelCase and snake_case (`processUserInput` → `["process", "user", "input"]`)
- BM25 with k1=1.5, b=0.75 (standard parameters)
- Persisted as binary pickle (5-10x faster than JSON for 900 MB+ indices)
- Auto-migrates from legacy JSON

**Problem it solves:** Semantic search misses exact keyword matches. A query for `PaymentService` should find all exact imports — BM25 handles this perfectly.

### 6. Hybrid Search + RRF (`core/hybrid_search.py`)

**What it does:** Merges FAISS (dense) and BM25 (sparse) results into a single ranked list.

**How it works:**
- Both FAISS and BM25 over-fetch (3x top_k candidates)
- RRF formula: `score(doc) = Σ 1/(60 + rank_in_list)` across both lists
- Items appearing in both lists are naturally boosted
- No score normalization needed (FAISS cosine and BM25 IDF are on different scales)

**Problem it solves:** Neither dense nor sparse search alone is sufficient. Cursor's research shows hybrid search is **12.5% more accurate** than either alone.

### 7. Cross-Encoder Reranker (`core/reranker.py`)

**What it does:** Refines the top-N candidates from Stage 1 with full cross-attention scoring.

**How it works:**
- Stage 1: FAISS + BM25 → RRF (fast, bi-encoder, over-fetches 3x)
- Stage 2: Cross-encoder scores each (query, chunk) pair with full attention
- Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers)
- Graceful fallback: if model can't load, reverts to pass-through (no reranking)

**Problem it solves:** Bi-encoder similarity (FAISS) is fast but imprecise for edge cases. Cross-attention is ~100x slower but much more accurate — affordable on top-N candidates only.

### 8. Path Obfuscation (`core/path_obfuscation.py`)

**What it does:** Encrypts file path segments with HMAC-SHA256 so the vector index stores no plaintext file names.

**How it works:**
- Path split by `/` and `.` into segments
- Each segment encrypted with a local secret key (stored at `~/.context/path_obfuscation_key`)
- Deterministic: same input always produces same output (enables deduplication)
- Decryption only possible with the local key

**Problem it solves:** If the FAISS index is compromised, attackers see `[a3f2c1, 9b4e7d]` instead of `src/auth/middleware.ts`. Matches Cursor's Architecture §5.

### 9. SimHash Index Reuse (`core/simhash.py` + `server/index_server.py`)

**What it does:** When a new team member joins, copies a similar teammate's index instead of re-indexing from scratch.

**How it works:**
- SimHash: 128-bit locality-sensitive hash computed from all file content hashes
- Server maintains a registry of `{workspace_id → simhash}` fingerprints
- New workspace computes SimHash → server finds best match above 85% threshold
- If found: copies FAISS + BM25 from the donor workspace, then runs incremental indexing to reconcile differences

**Performance impact (from Cursor's data):**
- Median repo: 7.87s → 525ms
- 99th percentile: 4.03 hours → 21 seconds

**Problem it solves:** Teammates on the same repo share ~92% of the same files. Without reuse, every developer pays the full indexing cost independently.

### 10. Background Sync (`mcp/sync.py`)

**What it does:** Automatically re-indexes changed files every 5 minutes after the initial index.

**How it works:**
- `SyncManager.start_background_sync()` — starts after first successful index
- Phase 1: Initial sync after 5 seconds
- Phase 2: Periodic sync every 300 seconds (configurable via `SYNC_INTERVAL_SECONDS`)
- Uses `context.reindex_by_change()` which leverages Merkle tree diffs — only re-processes actually changed files

**Problem it solves:** Without background sync, search results become stale as developers edit code. The 5-minute interval matches Cursor's architecture.

---

## Client-Server Model

CodeContext supports two deployment modes: **local** (single developer) and **client-server** (team).

### Local Mode (Default)

Everything runs on one machine — embedding, indexing, search, and the MCP server.

```
┌──────────────────────────────────────────────────┐
│                 Your Machine                      │
│                                                   │
│  VS Code / Claude Desktop                         │
│       │ MCP (stdio or streamable-http)            │
│       ▼                                           │
│  codecontext (MCP server, port 8877)              │
│       │                                           │
│       ├── Ollama (nomic-embed-text, port 11434)   │
│       ├── FAISS + BM25 index (~/.context/)        │
│       ├── Embedding cache (SQLite)                │
│       ├── Merkle tree sync (every 5 min)          │
│       └── AST splitter (Tree-sitter)              │
└──────────────────────────────────────────────────┘
```

**Prerequisites:** `uv`, Ollama running with `nomic-embed-text` pulled.

### Client-Server Mode (Team)

The index server centralizes embedding and search. Clients upload files and query remotely — no local GPU or Ollama needed on client machines.

```
┌─ Client Machines ────────┐            ┌─ Index Server (:8878) ─────────┐
│                          │            │                                │
│  VS Code + MCP           │   files    │  Ollama (embedding)            │
│  codecontext             │───────────▶│  FAISS + BM25 (per workspace)  │
│  (proxy mode)            │   search   │  Embedding cache (SQLite)      │
│                          │◀───results─│  SimHash registry (team reuse) │
│  No Ollama needed        │            │  Background sync (5 min)       │
│  No GPU needed           │            │  Merkle tree sync              │
└──────────────────────────┘            └────────────────────────────────┘
```

**How proxy mode works:** When `INDEX_SERVER_URL` is set, the MCP server becomes a thin proxy:
1. `index_codebase` → uploads files to the index server, triggers remote indexing
2. `search_code` → forwards query to the remote index server
3. `clear_index` → clears remote index
4. `get_indexing_status` → polls remote status

The client never runs an embedding model — all heavy computation happens on the server.

**SimHash team reuse:** When a new developer connects, the server computes their codebase SimHash and checks if a similar index already exists (≥85% similarity). If found, it copies the existing index and only re-indexes the differences — reducing median indexing from ~8 seconds to ~500ms.

---

## Startup Guide

### Quick Start (Local Mode — Single Developer)

```bash
# 1. Clone and install
git clone <repo-url> codebase-context
cd codebase-context
uv sync

# 2. Start Ollama (if not already running)
ollama serve &
ollama pull nomic-embed-text

# 3. Start the MCP server
uv run codecontext
# → Listening on http://127.0.0.1:8877/mcp (streamable-http)

# 4. Configure your editor (see Mode A section above)
```

That's it. Open your editor, use `index_codebase` to index a project, then `search_code` to search.

### Quick Start (Client-Server Mode — Team)

**On the server machine:**
```bash
# 1. Install
cd codebase-context && uv sync

# 2. Start Ollama
ollama serve &
ollama pull nomic-embed-text

# 3. Start the index server
uv run codecontext-server
# → Listening on 0.0.0.0:8878
```

**On each developer machine:**
```bash
# 1. Install (no Ollama needed)
cd codebase-context && uv sync

# 2. Start MCP server in proxy mode
INDEX_SERVER_URL=http://your-server:8878 uv run codecontext
```

Or configure it in `.vscode/mcp.json`:
```json
{
  "servers": {
    "codecontext": {
      "command": "uv",
      "args": ["run", "codecontext"],
      "env": {
        "MCP_TRANSPORT": "stdio",
        "INDEX_SERVER_URL": "http://your-server:8878"
      }
    }
  }
}
```

### Verifying It Works

```bash
# Check server health
curl http://localhost:8878/api/collections

# Index programmatically
python -c "
import asyncio
from codecontext.client import SyncClient
async def go():
    c = SyncClient('http://localhost:8878', '/path/to/codebase')
    await c.sync()
    results = await c.search('authentication handler')
    for r in results:
        print(f\"  {r['relative_path']}:{r['start_line']} (score={r['score']:.3f})\")
    await c.close()
asyncio.run(go())
"
```

---

## Problem-Solving Techniques

### Problem: "Indexing is slow on first run"

**Root cause:** Embedding is the bottleneck — each chunk requires an API call to Ollama/OpenAI.

**Mitigations already in place:**
1. **Embedding cache (SQLite):** After the first run, 99%+ of chunks are served from cache — re-indexing a 20K-file codebase with 50 changed files skips ~19,950 files worth of embeddings
2. **Pipelined producer-consumer:** File splitting (thread pool, up to 14 workers) runs concurrently with embedding — while batch N is being embedded, batch N+1 is being split
3. **Parallel embedding sub-batches:** Embedding requests are split into parallel sub-batches matching `OLLAMA_NUM_PARALLEL` (default 4)
4. **SimHash index reuse:** In team mode, new members copy an existing index instead of building from scratch

**What you can tune:**
```bash
# Increase Ollama parallelism (on Ollama server)
OLLAMA_NUM_PARALLEL=8 ollama serve

# Adjust batch size
EMBEDDING_BATCH_SIZE=200 uv run codecontext

# Use OpenAI for faster cloud embeddings
EMBEDDING_PROVIDER=openai OPENAI_API_KEY=sk-... uv run codecontext
```

### Problem: "Search doesn't find what I expected"

**Debugging steps:**
1. **Check indexing status:** Use `get_indexing_status` — if it shows "indexing" at <80%, results may be incomplete
2. **Try both query styles:** Semantic search finds code by meaning, but exact symbol names work better with keywords. The hybrid search (FAISS + BM25 + RRF) handles both, but very short queries (1-2 words) may benefit from being more specific
3. **Check file extensions:** Only supported extensions are indexed (23 by default). Use `customExtensions` in `index_codebase` to add more
4. **Force re-index:** If files were changed outside the 5-min sync window, use `index_codebase` with `force=true`

### Problem: "Memory usage is high"

**What consumes memory:**
- FAISS index: ~4 bytes × dimensions × num_chunks (768-dim × 100K chunks ≈ 300 MB)
- BM25 index: in-memory inverted index + document store
- Embedding cache read-through: grows as chunks are looked up

**Mitigations:**
- The embedding cache on-disk is SQLite — only looked-up entries are loaded into memory
- BM25 uses pickle persistence (5-10x less I/O than JSON)
- FAISS `IndexFlatIP` is the simplest index — for very large repos, consider IVFPQ (not yet implemented)

### Problem: "Background sync isn't picking up changes"

**How it works:** `SyncManager` starts after the first successful index, waits 5 seconds, then syncs every 5 minutes. It uses Merkle tree diffs — only branches where hashes diverge are walked.

**Debugging:**
- Check logs for `SyncManager` messages
- The sync interval is configurable: `SYNC_INTERVAL_SECONDS=60` for faster updates
- Manual re-index: call `index_codebase` again (without `force`) — it will skip unchanged files automatically

---

### 4. Run Tests

```bash
# Core pipeline tests
uv run python test_components.py

# Accuracy validation (AST splitting, cosine similarity, etc.)
uv run python test_accuracy.py

# Hybrid architecture tests (BM25, RRF, Merkle tree, reranker)
uv run python test_hybrid.py
```

## Pricing Comparison (2026)

CodeContext is free and open-source. Here's how it compares to paid alternatives:

### Individual Plans

| Tool | Plan | Price | Codebase Search |
|---|---|---|---|
| **GitHub Copilot** | Free | $0/mo | Local index capped at ~2,500 files |
| **GitHub Copilot** | Pro | $10/mo | + remote index (GitHub.com repos only) |
| **GitHub Copilot** | Pro+ | $39/mo | Same index limits, more model requests |
| **Cursor** | Hobby | $0/mo | Limited hybrid search |
| **Cursor** | Pro | $20/mo | Full hybrid search + Merkle sync |
| **Cursor** | Pro+ | $60/mo | 3× model usage |
| **Cursor** | Ultra | $200/mo | 20× model usage |
| **CodeContext** | — | $0 | Hybrid search (FAISS + BM25 + RRF), no file limit |

### Team Cost Comparison (10 developers)

| Setup | Monthly | Annual |
|---|---|---|
| Copilot Pro | $100 | $1,200 |
| Cursor Pro | $200 | $2,400 |
| Cursor Teams | $400 | $4,800 |
| **Copilot Pro + CodeContext** | **$100** | **$1,200** |

> **Note:** Cursor and Copilot are full AI IDE products — not just search. They include code completion, AI chat, agent mode, and more. CodeContext only provides codebase search. See [What CodeContext Does NOT Replace](#what-codecontext-does-not-replace) below.

## Where CodeContext Fits

CodeContext doesn't replace Cursor or Copilot. It **fills a specific gap**: bringing Cursor-quality codebase search to tools that don't have it.

| Scenario | Recommendation |
|---|---|
| **Large codebase + Copilot** | Add CodeContext as an MCP server. Copilot gets hybrid search across all files — no plan change needed. |
| **Evaluating Cursor vs Copilot** | Stay in VS Code with Copilot + CodeContext. Save $10–30/user/month. |
| **Enterprise team on a budget** | Copilot Pro ($10/user) + CodeContext (free) vs Cursor Teams ($40/user) = **$3,600/year saved** for 10 devs. |
| **Privacy-sensitive projects** | CodeContext + Ollama = 100% local. No code leaves your machine. Copilot's remote index uploads code to GitHub's cloud; Cursor processes code on their servers. |
| **Want the best AI IDE experience** | Use Cursor Pro ($20/mo). It's the most polished end-to-end product. |

## What CodeContext Does NOT Replace

| Feature | Cursor / Copilot | CodeContext |
|---|---|---|
| Code completion (tab) | ✅ Real-time, context-aware | ❌ Not a completion tool |
| AI chat | ✅ Multi-model, streaming | ❌ Not a chat interface |
| Agent mode | ✅ Multi-file editing | ❌ Feeds agents via MCP only |
| Cloud agents | ✅ (Cursor Pro, Copilot Pro) | ❌ Local only |
| PR code review | ✅ (Cursor Bugbot, Copilot) | ❌ Not in scope |
| IDE integration | ✅ Deep, native | ⚠️ Via MCP protocol |
| **Codebase search** | ✅ Proprietary | **✅ Open-source, comparable architecture** |
| Custom embedding model | ✅ (Cursor trains their own) | ⚠️ Uses open models (nomic-embed-text) |

## Known Limitations

1. **First-index time** — Indexing 20K files with Ollama takes minutes, not seconds. Cursor has optimized proprietary infrastructure. Pipelining and caching mitigate this after the first run.
2. **Embedding quality** — Cursor trains a custom embedding model on real coding sessions. CodeContext uses `nomic-embed-text` (137M params). Good, but not fine-tuned for code search.
3. **MCP overhead** — Communication via MCP protocol adds ~50–100ms per query compared to native in-process search.
4. **Mac-first optimization** — The pipelined engine is optimized for Apple Silicon (M4 Pro). Works on Linux/Windows but not tuned.

## Privacy Considerations

AI coding tools handle your source code differently:

| Tool | Where Code Is Processed | Privacy Risk |
|---|---|---|
| **GitHub Copilot (local index)** | On your machine | Low — but capped at ~2,500 files |
| **GitHub Copilot (remote index)** | Uploaded to GitHub's cloud (`api.github.com`) | **High** — your entire codebase is sent to GitHub's servers for indexing. Community reports indicate ~500MB uploads during indexing. Only works with GitHub.com repos. |
| **Cursor** | Code sent to Cursor's servers for embedding + indexing | **High** — proprietary infrastructure, code leaves your machine |
| **CodeContext + Ollama** | 100% on your machine | **None** — embedding runs locally via Ollama, FAISS index stored at `~/.context/`, no network calls |

For teams working on proprietary code, regulated industries (healthcare, finance, defense), or codebases under NDA, remote indexing is a non-starter. Copilot's remote index sends your source code to GitHub's cloud infrastructure for processing — even if your repo is private. Cursor similarly processes code on their servers.

CodeContext with Ollama keeps everything local: the embedding model runs on your machine's GPU/CPU, the FAISS index is stored on your local disk, and zero bytes of code are transmitted over the network.
