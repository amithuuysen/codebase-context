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
│   ├── simhash.py                 # SimHash locality-sensitive hashing (team index sharing)
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
│   └── index_server.py            # Starlette ASGI app — upload, index, search, SimHash
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
│   └── code_chunks_<hash>.json
├── embedding_cache/                   # Embedding vector cache
│   └── ollama_nomic-embed-text.json
├── merkle/                            # Merkle tree snapshots
│   ├── merkle_<hash>.json             # Directory-aware tree
│   └── <hash>.json                    # Flat file-hash fallback
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

### Mode B: Remote Index Server (Team Sharing)

Index server runs on a shared machine. Developers send files to it and search remotely — no local GPU needed on client machines.

```
Developer A (local)              Index Server (remote :8878)       Developer B (local)
┌──────────────────┐   files    ┌──────────────────────┐  files   ┌──────────────────┐
│ VS Code + MCP    │───────────▶│ FAISS + BM25         │◀─────────│ VS Code + MCP    │
│ codecontext      │   JSON     │ Ollama embedding     │  JSON    │ codecontext      │
│ (proxy mode)     │◀───search──│ SimHash registry     │──search─▶│ (proxy mode)     │
│                  │   results  │ Shared team indexes  │  results │                  │
└──────────────────┘            └──────────────────────┘          └──────────────────┘
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

#### SimHash Team Sharing

When a developer uploads their codebase, the server computes a SimHash fingerprint and checks for similar existing indexes. If a teammate's index is 90%+ similar (e.g., same repo, different branch), the server reuses it — avoiding redundant re-indexing.

#### Index Server API

| Endpoint | Method | Description |
|---|---|---|
| `/api/upload-json` | POST | Upload files as JSON (batch of 100) |
| `/api/index` | POST | Trigger indexing for a workspace |
| `/api/search` | POST | Search indexed codebase |
| `/api/status/{workspace_id}` | GET | Check indexing progress |
| `/api/collections` | GET | List all indexed workspaces |
| `/api/clear/{workspace_id}` | DELETE | Delete a workspace's index |
| `/api/simhash` | POST | Find similar indexes by SimHash |
| `/api/simhash/register` | POST | Register a workspace's SimHash fingerprint |

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
