# TODO — Cursor Parity Features

## Completed

- [x] Merkle tree sync (O(changes) instead of O(files))
- [x] AST-aware chunking (Tree-sitter: functions, classes, methods)
- [x] Hybrid search (FAISS dense + BM25 sparse + RRF fusion)
- [x] Cross-encoder reranker (optional Stage 2 precision)
- [x] Parallel file reading/splitting (ThreadPoolExecutor)
- [x] Ollama embed_batch_size=100 (10x fewer API round-trips)
- [x] Embedding cache by chunk hash (SHA-256 → embedding vector)
- [x] Path obfuscation (HMAC-SHA256 per segment)
- [x] Background 5-min periodic sync
- [x] Producer/consumer pipeline (asyncio.Queue, concurrent splitting + embedding)
- [x] Concurrent embedding sub-batches (~4 parallel threads per flush)
- [x] Deferred FAISS persistence (single write at end of indexing)
- [x] Adaptive thread pool (scales to CPU cores, up to 14 for M4 Pro)
- [x] Periodic embedding cache saves (every 10 batches)
- [x] Pipeline timing instrumentation (wall, split, embed, overlap, throughput)
- [x] Skip unchanged files during initial index (Merkle tree hash comparison)

## Yet To Implement

### 1. Git history indexing (Medium)
- Index commit SHAs, changed files per commit
- Use `gitpython` to walk recent commit history
- Enable queries like "when was authentication last changed?"

### 2. Custom embedding model (Hard)
- Fine-tune embedding model on agent session traces
- Use LLM-ranked relevance from real coding tasks as training signal
- Requires: training data pipeline, GPU infrastructure
- Consider fine-tuning `all-MiniLM-L6-v2` or `nomic-embed-text` as starting point
- Cursor's biggest proprietary advantage

## Performance Notes

- 20K-file codebase (Zoho CRM): ~20,583 files
- Current model: `nomic-embed-text` (137M params, 768-dim, 8K context)
- Chunk size: 3000 chars, overlap: 300
- Batch size: 100 chunks per Ollama API call
