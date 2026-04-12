# Cursor Codebase Indexing Architecture

> Compiled from Cursor's official documentation, security page, and research blog posts (cursor.com).

---

## 1. High-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CURSOR CLIENT (VS Code Fork)                      │
│                                                                             │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────────────────┐ │
│  │ File Watcher  │  │ Merkle Tree      │  │ .gitignore / .cursorignore     │ │
│  │ (every 5 min) │  │ Builder          │  │ Filter                         │ │
│  └──────┬───────┘  └───────┬──────────┘  └──────────────┬─────────────────┘ │
│         │                  │                             │                   │
│         ▼                  ▼                             ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                  Merkle Tree Sync Engine                                │ │
│  │  - Computes SHA-256 hash per file                                      │ │
│  │  - Builds tree: folder hash = f(children hashes)                       │ │
│  │  - Compares local tree vs server tree                                  │ │
│  │  - Uploads only changed files (delta sync)                             │ │
│  │  - Encrypts file paths with client-side secret key                     │ │
│  └──────────────────────────────┬──────────────────────────────────────────┘ │
│                                 │                                            │
│  ┌──────────────────────────────┴──────────────────────────────────────────┐ │
│  │              Local Decryption & Chunk Reader                            │ │
│  │  - Receives obfuscated file path + line range from server              │ │
│  │  - Decrypts path using client-side secret key                          │ │
│  │  - Reads actual code chunks from local filesystem                      │ │
│  │  - Sends chunks to server ONLY at inference time                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                           HTTPS (HTTP/2)
                        repo42.cursor.sh
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                       CURSOR SERVER (AWS)                                    │
│                                                                              │
│  ┌─────────────────────┐   ┌────────────────────┐   ┌────────────────────┐  │
│  │  Merkle Tree Store  │   │  File Chunker      │   │  Embedding Cache   │  │
│  │  (per workspace)    │   │  (syntactic chunks) │   │  (keyed by chunk   │  │
│  │                     │   │                     │   │   content hash)    │  │
│  └─────────┬───────────┘   └─────────┬──────────┘   └────────┬───────────┘  │
│            │                         │                        │              │
│            │                         ▼                        │              │
│            │               ┌─────────────────────┐            │              │
│            │               │  Custom Embedding    │◄───────────┘              │
│            │               │  Model               │  (cache hit = skip)      │
│            │               │  (trained on agent   │                           │
│            │               │   session traces)    │                           │
│            │               └─────────┬────────────┘                           │
│            │                         │                                        │
│            │                         ▼                                        │
│            │               ┌─────────────────────┐                            │
│            │               │  Vector Storage      │                           │
│            │               │  (Turbopuffer on GCP)│                           │
│            │               │                      │                           │
│            │               │  Stores per chunk:   │                           │
│            │               │  - Embedding vector  │                           │
│            │               │  - Obfuscated path   │                           │
│            │               │  - Line range         │                          │
│            │               └─────────┬────────────┘                           │
│            │                         │                                        │
│  ┌─────────▼─────────────────────────▼────────────────────────────────────┐  │
│  │                    Inference / Query Engine                             │  │
│  │  1. Receive user query                                                 │  │
│  │  2. Embed query with same custom model                                 │  │
│  │  3. Nearest-neighbor search in Turbopuffer                             │  │
│  │  4. Return obfuscated paths + line ranges to client                    │  │
│  │  5. Client reads local chunks, sends them back                         │  │
│  │  6. LLM uses chunks to answer user's question                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    Index Reuse / Team Sharing                          │  │
│  │  - SimHash computed from Merkle tree                                   │  │
│  │  - SimHash used as vector to find similar indexes in same team         │  │
│  │  - If similarity > threshold, copy existing index                      │  │
│  │  - Content proofs (Merkle hashes) ensure no file leaks across users    │  │
│  │  - Background sync reconciles remaining differences                    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Indexing Pipeline — Step by Step

### 2.1 File Discovery (Client-Side)

1. Cursor scans the opened workspace folder.
2. Files/directories matched by `.gitignore` or `.cursorignore` are excluded.
3. A **Merkle tree** is built:
   - **Leaf nodes**: SHA-256 hash of each file's content.
   - **Internal nodes**: Hash derived from children's hashes.
4. The Merkle tree is synced to the server.

### 2.2 Delta Sync (Client → Server)

| Scenario | Action |
|---|---|
| File modified | Only the changed file is re-uploaded |
| File added | New file uploaded, tree updated |
| File deleted | Removed from server tree |
| File unchanged | Skipped (hash matches) |

- **Sync frequency**: Every **5–10 minutes** the client checks for hash mismatches.
- For a 50,000-file workspace, file names + SHA-256 hashes ≈ **3.2 MB**. With Merkle tree diffing, only divergent branches are transferred.

### 2.3 Chunking (Server-Side)

- Files are split into **syntactic chunks** (functions, classes, logical blocks).
- Chunking is language-aware — respects code structure boundaries.

### 2.4 Embedding Generation (Server-Side)

- Each chunk is converted into a **vector embedding** using Cursor's **custom embedding model**.
- The custom model is trained on **agent session traces** — it learns what code should have been retrieved based on how agents actually navigate codebases.
- **Embedding cache**: Keyed by chunk content hash (stored in AWS). If a chunk's content hasn't changed, the cached embedding is reused.
- This is the most expensive step and runs **asynchronously in the background**.

### 2.5 Vector Storage (Turbopuffer on GCP)

Each embedding record stores:

| Field | Description |
|---|---|
| **Embedding vector** | The semantic representation of the code chunk |
| **Obfuscated file path** | Path split by `/` and `.`, each segment encrypted with a client-side secret key and deterministic 6-byte nonce |
| **Line range** | Start and end lines of the chunk in the original file |

- Stored in **Turbopuffer** (vector database running on Google Cloud Platform, US servers).
- No plaintext code or real file names are stored in Turbopuffer.

---

## 3. Query / Retrieval Flow

```
User Query: "Where do we handle authentication?"
    │
    ▼
┌──────────────────────────────────────────┐
│ 1. QUERY EMBEDDING (Server)              │
│    Same custom embedding model converts   │
│    the natural language query into a      │
│    vector                                 │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│ 2. NEAREST-NEIGHBOR SEARCH (Turbopuffer) │
│    Query vector matched against stored    │
│    code embeddings                        │
│    Returns: obfuscated paths + line       │
│    ranges of top-k matching chunks        │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│ 3. PATH DECRYPTION (Client)              │
│    Client decrypts obfuscated paths       │
│    using its local secret key             │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│ 4. LOCAL FILE READ (Client)              │
│    Client reads the actual code chunks    │
│    from local filesystem at the given     │
│    line ranges                            │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│ 5. CHUNKS SENT TO LLM (Server)          │
│    Retrieved code chunks sent back to     │
│    server and fed to the LLM along with   │
│    the user's question                    │
└──────────────────────────────────────────┘
```

**Key insight**: For Privacy Mode users, **no plaintext code is stored** on Cursor's servers or in Turbopuffer. Code only transits the server at inference time and is not persisted.

---

## 4. Index Reuse Across Team Members

```
New User Joins Team
    │
    ▼
┌─────────────────────────────────────────┐
│ Compute Merkle tree for local codebase  │
│ Derive SimHash (similarity hash) from   │
│ the tree — a single value summarizing   │
│ all file content hashes                 │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Upload SimHash to server                │
│ Server does vector search across all    │
│ team/user SimHashes                     │
└──────────────────┬──────────────────────┘
                   │
            Match found?
           ┌───────┴───────┐
           │               │
         Yes              No
           │               │
           ▼               ▼
┌──────────────────┐  ┌────────────────────┐
│ Copy existing    │  │ Full indexing from  │
│ index            │  │ scratch             │
│ (background)     │  │                     │
└────────┬─────────┘  └────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ Content Proofs (Merkle tree hashes)      │
│ ensure user only sees results for files  │
│ they actually have locally               │
│                                          │
│ Background sync reconciles remaining     │
│ differences with the copied index        │
└──────────────────────────────────────────┘
```

### Performance Impact (from Cursor's data)

| Percentile | Without Reuse | With Reuse |
|---|---|---|
| Median repo | 7.87 seconds | 525 milliseconds |
| 90th percentile | 2.82 minutes | 1.87 seconds |
| 99th percentile (large repos) | 4.03 hours | 21 seconds |

Average codebase similarity within a team: **92%**.

---

## 5. File Path Encryption

```
Original path:    src/auth/middleware/session.ts
                   │
Split by / and .   │
                   ▼
Segments:         [src, auth, middleware, session, ts]
                   │
Each segment       │   secret key (stored on client only)
encrypted with ────┤   deterministic 6-byte nonce
                   │
                   ▼
Obfuscated:       [a3f2c1, 9b4e7d, c8a1f3, 2d6e9a, 7f3b2c]
```

**What this leaks**: Directory hierarchy structure (depth, number of segments). Does NOT leak actual names.  
**Nonce**: 6-byte deterministic — some collisions possible, but considered acceptable.  
**Key derivation for teams**: Secret key derived from hashes of recent commit contents, enabling shared data structures within the same team/repo.

---

## 6. Custom Embedding Model

Cursor uses a **proprietary embedding model** (not OpenAI's off-the-shelf model) trained specifically for code retrieval:

### Training Process

```
Agent Session Traces (training data)
    │
    │  Agent searches, opens files, finds code
    │  during real task execution
    │
    ▼
┌──────────────────────────────────────────┐
│ LLM Ranking                              │
│ An LLM analyzes traces and ranks what    │
│ content SHOULD have been retrieved        │
│ earlier in the conversation               │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│ Embedding Model Training                 │
│ Model trained to align similarity        │
│ scores with LLM-generated rankings       │
│                                          │
│ Creates feedback loop:                   │
│ model learns from how agents actually    │
│ work through coding tasks                │
└──────────────────────────────────────────┘
```

### Results

- **+12.5% average accuracy** improvement over grep-only (6.5%–23.5% depending on model).
- **+0.3% code retention** (code written stays in codebase) — rises to **+2.6%** on repos with 1,000+ files.
- **-2.2% dissatisfied follow-up requests** when semantic search is available.

---

## 7. Infrastructure & Subprocessors

### Where Things Run

| Component | Provider | Location |
|---|---|---|
| Primary servers, API, embedding cache | **AWS** | US |
| Reverse proxy | **Cloudflare** | Global |
| Secondary infrastructure | **Azure**, **GCP** | US |
| Vector database (embeddings) | **Turbopuffer** (on GCP) | US |
| Custom model inference | **Fireworks**, **Baseten**, **Together** | US, Canada, Asia, Europe |
| LLM providers | **OpenAI**, **Anthropic**, **Google Vertex**, **xAI** | Various |
| Web search | **Exa**, **SerpApi** | — |

### Key Endpoints

| Domain | Purpose |
|---|---|
| `repo42.cursor.sh` | Codebase indexing (HTTP/2 only) |
| `api2.cursor.sh` | Most API requests |
| `api3.cursor.sh` | Cursor Tab requests |
| `api5.cursor.sh` | Agent requests |

---

## 8. Privacy Mode Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Request Flow                                │
│                                                                  │
│  Client ──► Proxy ──┬──► Privacy Mode Replica (logs disabled)   │
│                     │                                            │
│         x-ghost-mode│                                            │
│         header      ├──► Non-Privacy Replica (logs enabled)     │
│                     │                                            │
│  Missing header ────┘    Default: treat as Privacy Mode         │
│                                                                  │
│  Background tasks ──► Parallel queues & workers per mode        │
│                                                                  │
│  Team enforcement:                                               │
│  - Client pings server every 5 min                              │
│  - Server also checks in hot path (cache TTL: 5 min)           │
│  - Cache miss → assume Privacy Mode                             │
│                                                                  │
│  Zero Data Retention agreements with:                           │
│  Fireworks, Baseten, Together, OpenAI, Anthropic, Google Vertex │
└─────────────────────────────────────────────────────────────────┘
```

### What Privacy Mode guarantees:
- Code data is **never stored** by model providers.
- Code is **never used for training**.
- No plaintext code stored on Cursor servers or Turbopuffer.
- Separate infrastructure replicas for privacy/non-privacy requests.
- Log functions are **no-ops** on privacy replicas.

---

## 9. Complete Data Flow — End to End

```
┌─ CLIENT ────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. Open workspace                                                      │
│  2. Scan files (respect .gitignore / .cursorignore)                     │
│  3. Build Merkle tree (SHA-256 per file, hash-of-hashes per folder)     │
│  4. Encrypt file paths with client-side secret key                      │
│  5. Sync Merkle tree to server (delta only)                             │
│  6. Upload changed files to server                                      │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                        repo42.cursor.sh
                           (HTTP/2)
                                 │
┌─ SERVER ───────────────────────▼────────────────────────────────────────┐
│                                                                         │
│  7. Receive files                                                       │
│  8. Split into syntactic chunks (language-aware)                        │
│  9. Hash each chunk                                                     │
│  10. Check embedding cache (AWS, keyed by chunk hash)                   │
│      ├── Cache HIT  → reuse existing embedding                         │
│      └── Cache MISS → generate embedding with custom model              │
│  11. Store embedding + obfuscated path + line range in Turbopuffer      │
│  12. Discard plaintext code from memory                                 │
│                                                                         │
│  ─── INDEXING COMPLETE (available at 80% completion) ───                │
│                                                                         │
│  13. User sends query                                                   │
│  14. Embed query with same custom model                                 │
│  15. Nearest-neighbor search in Turbopuffer                             │
│  16. Return obfuscated paths + line ranges to client                    │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌─ CLIENT ───────────────────────▼────────────────────────────────────────┐
│                                                                         │
│  17. Decrypt file paths                                                 │
│  18. Read code chunks from local filesystem                             │
│  19. Send chunks back to server for LLM context                         │
│                                                                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌─ SERVER ───────────────────────▼────────────────────────────────────────┐
│                                                                         │
│  20. LLM receives: user query + retrieved code chunks + conversation    │
│  21. LLM generates response                                            │
│  22. Response sent to client                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Git History Indexing

When codebase indexing is enabled in a Git repo, Cursor also indexes Git history:

| Indexed | NOT Indexed |
|---|---|
| Commit SHAs | Commit messages |
| Parent commit information | File contents / diffs |
| Obfuscated file names (same encryption) | — |

For team sharing, the obfuscation key is **derived from hashes of recent commit contents**, enabling shared data structures across teammates on the same repo.

---

## 11. Search Strategy — How Agent Uses the Index

| User Intent | Tool Selected | Example |
|---|---|---|
| Specific symbol or string | **Instant Grep** (custom engine, faster than ripgrep) | "Find all files that import PaymentService" |
| Concept or behavior | **Semantic search** → then grep for details | "How does our app handle authentication?" |
| Complex exploration | **Multiple searches** + file reads + reference following | "Map the data flow from checkout to confirmation email" |
| Broad codebase scan | **Explore subagent** (parallel searches in separate context) | "Find all places we validate user input" |

The agent decides which tool(s) to use automatically. Semantic search + grep combined yields **12.5% higher accuracy** than grep alone.

---

## 12. Security Considerations

| Concern | Mitigation |
|---|---|
| Embedding reversal attacks | Academic attacks require model access + short strings; Cursor considers risk low but acknowledges possibility |
| File path leakage | Encrypted per-segment; leaks hierarchy structure only |
| Nonce collisions | 6-byte deterministic nonce; some collisions expected but acceptable |
| Data at rest | Turbopuffer stores only embeddings + obfuscated metadata; no plaintext code |
| Data in transit | HTTPS/HTTP2; code transits server only at inference time |
| Model provider retention | Zero data retention agreements with all providers for Privacy Mode users |
| SOC 2 | Type II certified |
| Penetration testing | Annual third-party testing |

---

## 13. Summary: Where Does Embedding Happen?

**Embedding conversion happens entirely on Cursor's remote servers — NOT locally.**

```
LOCAL (Client)                    REMOTE (Server)
─────────────                     ───────────────
✓ File scanning                   ✓ Chunking
✓ Merkle tree building            ✓ Embedding generation
✓ File hashing (SHA-256)          ✓ Embedding caching
✓ Path encryption                 ✓ Vector storage (Turbopuffer)
✓ Delta sync                      ✓ Nearest-neighbor search
✓ Path decryption                 ✓ LLM inference
✓ Local chunk reading
```

Files are uploaded to Cursor's servers (AWS), chunked and embedded there, and the vectors are stored in Turbopuffer (GCP). The client never runs an embedding model.

---

*Sources: [cursor.com/security](https://cursor.com/security), [cursor.com/blog/secure-codebase-indexing](https://cursor.com/blog/secure-codebase-indexing), [cursor.com/blog/semsearch](https://cursor.com/blog/semsearch), [cursor.com/docs](https://cursor.com/docs/agent/tools/search)*
