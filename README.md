# codecontext — Python MCP Server for Semantic Code Search

A complete Python reimplementation of [zilliztech/claude-context](https://github.com/zilliztech/claude-context) that stores vectors **locally** (in-memory + disk persistence via numpy) instead of requiring Milvus/Zilliz Cloud.

## Architecture

```
codecontext/
├── __init__.py        # Package init
├── config.py          # Types, constants, Config (reads env vars)
├── splitter.py        # AstSplitter (tree-sitter) + TextSplitter (fallback)
├── embedding.py       # OpenAI / Ollama / local (sentence-transformers)
├── vectordb.py        # InMemoryVectorDatabase (numpy cosine similarity + disk persistence)
├── sync.py            # FileSynchronizer (SHA-256 change detection)
├── snapshot.py        # SnapshotManager (indexing state persistence)
├── context.py         # Context orchestrator (index → search → reindex)
└── server.py          # MCP server (4 tools over stdio)
```

## Quick Start

### 1. Install

```bash
cd python-port
pip install -e .
```

For fully offline embeddings (no API key needed):
```bash
pip install -e ".[sentence-transformers]"
```

### 2. Configure

Set environment variables:

```bash
# Option A: OpenAI embeddings (default)
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# Option B: Ollama (local, free)
export EMBEDDING_PROVIDER=ollama
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=nomic-embed-text

# Option C: Sentence-transformers (fully offline)
export EMBEDDING_PROVIDER=local
export LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3. Run as MCP server

```bash
codecontext
# or
python -m codecontext.server
```

### 4. Connect to Claude / Copilot

Add to your MCP configuration (e.g. `~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "codecontext": {
      "command": "python",
      "args": ["-m", "codecontext.server"],
      "env": {
        "EMBEDDING_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `index_codebase` | Index a directory (scan → split → embed → store) |
| `search_codebase` | Semantic search over indexed code |
| `get_codebase_summary` | List all indexed codebases + status |
| `clear_codebase_index` | Drop the index for a codebase |

## How It Works

### Indexing
```
Source files → ignore filter → AST splitter (tree-sitter) → CodeChunks
  → Embedding API (batch) → numpy vectors + original text → disk (.npy + .json)
```

### Search
```
Query string → Embedding API → query vector
  → cosine similarity against stored vectors (numpy)
  → top-K results with original code text, file path, line numbers
```

### Incremental Sync
Every 5 minutes, background sync detects changed files via SHA-256 hashing
and only re-indexes the diff.

## Data Storage

All data is stored locally under `~/.context/`:

```
~/.context/
├── vectordb/                    # Vector collections (numpy .npy + metadata .json)
│   └── code_chunks_<hash>/
│       ├── vectors.npy          # Embedding vectors
│       └── metadata.json        # Document text + metadata
├── merkle/                      # File hash snapshots for change detection
│   └── <hash>.json
└── mcp-codebase-snapshot.json   # Indexing state
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `openai` | `openai`, `ollama`, or `local` |
| `EMBEDDING_MODEL` | auto | Model name (auto-selected per provider) |
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `OPENAI_BASE_URL` | — | Custom API endpoint |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `nomic-embed-text` | Ollama model |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `CHUNK_SIZE` | `2500` | Max characters per code chunk |
| `CHUNK_OVERLAP` | `300` | Overlap between chunks |
| `EMBEDDING_BATCH_SIZE` | `100` | Chunks per embedding API call |
| `CODECONTEXT_DATA_DIR` | `~/.context` | Data storage directory |
