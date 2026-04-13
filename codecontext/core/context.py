"""
Context — the main orchestrator.

Pipeline (Cursor-inspired hybrid architecture):
  Codebase → Merkle-tree sync → Tree-sitter (AST) → LlamaIndex TextNode
  → FAISS (dense) + BM25 (sparse) → RRF fusion → optional cross-encoder rerank

Key accuracy techniques (from Cursor research):
  1. Hybrid search: semantic (FAISS) + keyword (BM25) fused via RRF → 12.5% better
  2. Merkle tree sync: O(changes) not O(files) — skip unchanged subtrees
  3. Cross-encoder reranker: precision refinement on the top-N candidates
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from .types import (
    CONTEXT_DIR,
    DEFAULT_IGNORE_PATTERNS,
    DEFAULT_SUPPORTED_EXTENSIONS,
    EXTENSION_TO_LANGUAGE,
    CodeChunk,
    Config,
    SemanticSearchResult,
)
from .bm25 import BM25Index
from .embedding_cache import EmbeddingCache
from .hybrid_search import HybridSearcher
from .merkle import MerkleSynchronizer
from .path_obfuscation import PathObfuscator
from .reranker import Reranker
from .splitter import AstSplitter, Splitter, TextSplitter
from .sync import FileSynchronizer
from .vectordb import FaissVectorDB

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int, int], None]  # phase, current, total, pct


class Context:
    """
    Top-level orchestrator (Cursor-inspired hybrid architecture).

    Responsibilities:
      1. index_codebase()  – scan → tree-sitter split → TextNode → FAISS + BM25
      2. semantic_search() – hybrid: FAISS (dense) + BM25 (sparse) → RRF → rerank
      3. reindex_by_change() – Merkle tree diff → update only changed files
    """

    def __init__(
        self,
        vector_db: FaissVectorDB,
        splitter: Splitter | None = None,
        supported_extensions: set[str] | None = None,
        ignore_patterns: list[str] | None = None,
        config: Config | None = None,
        reranker: Reranker | None = None,
    ):
        self.vector_db = vector_db
        self.splitter = splitter or AstSplitter()
        self.supported_extensions = set(supported_extensions or DEFAULT_SUPPORTED_EXTENSIONS)
        self.ignore_patterns = list(ignore_patterns or DEFAULT_IGNORE_PATTERNS)

        self._cfg = config or Config.from_env()
        self._synchronizers: dict[str, FileSynchronizer] = {}
        self._merkle_synchronizers: dict[str, MerkleSynchronizer] = {}

        # BM25 sparse indices (one per collection / codebase)
        self._bm25_indices: dict[str, BM25Index] = {}

        # Hybrid searcher (FAISS + BM25 + RRF)
        self._hybrid = HybridSearcher(vector_db, self._bm25_indices)

        # Optional cross-encoder reranker (Stage 2 precision)
        self._reranker = reranker or Reranker()

        # Embedding cache — skip re-embedding unchanged chunks (Cursor-style)
        self._embedding_cache = EmbeddingCache(
            self._cfg.data_dir,
            self._cfg.embedding_provider,
            self._cfg.embedding_model,
        )

        # Path obfuscation — encrypt path segments for privacy (Cursor-style)
        self._path_obfuscator = PathObfuscator(self._cfg.data_dir)

        logger.info(
            "Context initialized — %d extensions, %d ignore patterns, "
            "reranker=%s, embedding_cache=%d entries",
            len(self.supported_extensions),
            len(self.ignore_patterns),
            self._reranker.provider,
            len(self._embedding_cache),
        )

    # ------------------------------------------------------------------
    # Public helpers (match TS Context getters/setters)
    # ------------------------------------------------------------------

    def get_vector_db(self) -> FaissVectorDB:
        return self.vector_db

    def get_splitter(self) -> Splitter:
        return self.splitter

    def get_supported_extensions(self) -> set[str]:
        return self.supported_extensions

    def get_ignore_patterns(self) -> list[str]:
        return list(self.ignore_patterns)

    def get_synchronizers(self) -> dict[str, FileSynchronizer]:
        return self._synchronizers

    def get_merkle_synchronizers(self) -> dict[str, MerkleSynchronizer]:
        return self._merkle_synchronizers

    def get_bm25_indices(self) -> dict[str, BM25Index]:
        return self._bm25_indices

    def get_reranker(self) -> Reranker:
        return self._reranker

    def set_synchronizer(self, collection_name: str, sync: FileSynchronizer) -> None:
        self._synchronizers[collection_name] = sync

    def get_collection_name(self, codebase_path: str) -> str:
        normalized = os.path.abspath(codebase_path)
        h = hashlib.md5(normalized.encode()).hexdigest()
        return f"code_chunks_{h[:8]}"

    # -- mutation helpers (mirror TS) --

    def add_custom_extensions(self, extensions: list[str]) -> None:
        self.supported_extensions.update(extensions)

    def add_custom_ignore_patterns(self, patterns: list[str]) -> None:
        for p in patterns:
            if p not in self.ignore_patterns:
                self.ignore_patterns.append(p)

    def update_ignore_patterns(self, patterns: list[str]) -> None:
        self.ignore_patterns = list(patterns)

    def reset_ignore_patterns_to_defaults(self) -> None:
        self.ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)

    def update_splitter(self, splitter: Splitter) -> None:
        self.splitter = splitter

    def get_splitter_info(self) -> dict[str, Any]:
        is_ast = isinstance(self.splitter, AstSplitter)
        return {
            "type": "ast" if is_ast else "text",
            "has_builtin_fallback": is_ast,
        }

    # -- public wrappers matching TS (used by handlers) --

    async def get_loaded_ignore_patterns(self, codebase_path: str) -> None:
        """Public wrapper — load .gitignore, .contextignore, global ignore."""
        await self._load_ignore_patterns(codebase_path)

    async def get_prepared_collection(self, codebase_path: str) -> None:
        """Public wrapper — ensure collection exists."""
        self._prepare_collection(codebase_path)

    # ------------------------------------------------------------------
    # Index a codebase
    # ------------------------------------------------------------------

    async def index_codebase(
        self,
        codebase_path: str,
        progress: ProgressCallback | None = None,
        force_reindex: bool = False,
        max_files: int = 0,
    ) -> dict[str, Any]:
        """
        Full pipeline:
          1. Load ignore patterns (.gitignore etc.)
          2. Create FAISS collection (if needed)
          3. Scan code files
          4. Tree-sitter split → LlamaIndex TextNode → insert into FAISS
        """
        codebase_path = os.path.abspath(codebase_path)
        logger.info("Starting indexing: %s (force=%s)", codebase_path, force_reindex)

        await self._load_ignore_patterns(codebase_path)

        _report(progress, "Preparing collection…", 0, 100, 0)
        self._prepare_collection(codebase_path, force_reindex)

        _report(progress, "Scanning files…", 5, 100, 5)
        files = self._get_code_files(codebase_path)
        logger.info("Found %d code files", len(files))
        if max_files > 0 and len(files) > max_files:
            files = files[:max_files]
            logger.info("Capped to %d files (--max-files)", max_files)
        if not files:
            _report(progress, "No files to index", 100, 100, 100)
            return {"indexed_files": 0, "total_chunks": 0, "status": "completed"}

        # Skip unchanged files (Architecture §2.2): when not force-reindexing
        # and a previous Merkle tree exists, only process files whose content
        # hash has changed since last index.
        skipped_unchanged = 0
        col = self.get_collection_name(codebase_path)
        if not force_reindex and self.vector_db.has_collection(col):
            merkle = self._merkle_synchronizers.get(col)
            if merkle is None:
                merkle = MerkleSynchronizer(codebase_path, self.ignore_patterns, merkle_dir=Path(self._cfg.data_dir) / "merkle")
                await merkle.initialize()
                self._merkle_synchronizers[col] = merkle
            if merkle._saved_tree is not None:
                files, skipped_unchanged = self._skip_unchanged_files(
                    files, codebase_path, merkle
                )
                if skipped_unchanged:
                    logger.info(
                        "Skipped %d unchanged files, %d files to process",
                        skipped_unchanged, len(files),
                    )

        if not files and skipped_unchanged > 0:
            _report(progress, "All files unchanged — nothing to index", 100, 100, 100)
            return {
                "indexed_files": 0,
                "total_chunks": 0,
                "skipped_unchanged": skipped_unchanged,
                "status": "completed",
            }

        result = await self._process_file_list(files, codebase_path, progress)

        # Persist sync snapshot (Merkle tree + flat hash fallback)
        col = self.get_collection_name(codebase_path)
        # Legacy flat sync
        sync = self._synchronizers.get(col)
        if sync is None:
            sync = FileSynchronizer(codebase_path, self.ignore_patterns)
            await sync.initialize()
            self._synchronizers[col] = sync
        sync.save_current_state()

        # Merkle tree sync (Cursor-style directory-aware change detection)
        merkle = self._merkle_synchronizers.get(col)
        if merkle is None:
            merkle = MerkleSynchronizer(codebase_path, self.ignore_patterns, merkle_dir=Path(self._cfg.data_dir) / "merkle")
            await merkle.initialize()
            self._merkle_synchronizers[col] = merkle
        merkle.save_current_state()

        # Persist BM25 index
        bm25 = self._bm25_indices.get(col)
        if bm25:
            bm25_path = Path(self._cfg.data_dir) / "bm25_store" / f"{col}.json"
            bm25.save(bm25_path)

        # Persist embedding cache
        self._embedding_cache.save()

        logger.info(
            "Indexing complete: %d/%d files processed, %d skipped, %d chunks",
            result["processed_files"], result["total_files"],
            result["skipped_files"], result["total_chunks"],
        )
        _report(progress, "Indexing complete!", result["processed_files"], len(files), 100)
        return {
            "indexed_files": result["processed_files"],
            "total_files": result["total_files"],
            "skipped_files": result["skipped_files"],
            "total_chunks": result["total_chunks"],
            "status": result["status"],
        }

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    async def semantic_search(
        self,
        codebase_path: str,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
        filter_expr: str | None = None,
    ) -> list[SemanticSearchResult]:
        """
        Hybrid search: FAISS (dense) + BM25 (sparse) → RRF → rerank.

        Stage 1: Retrieve top-N candidates via FAISS + BM25, fuse with RRF.
        Stage 2: Rerank top-N with cross-encoder (if available) → final top-K.
        """
        codebase_path = os.path.abspath(codebase_path)
        col_name = self.get_collection_name(codebase_path)

        if not self.vector_db.has_collection(col_name):
            logger.warning("Collection %s does not exist", col_name)
            return []

        # Load BM25 index from disk if not in memory
        if col_name not in self._bm25_indices:
            bm25_path = Path(self._cfg.data_dir) / "bm25_store" / f"{col_name}.json"
            if bm25_path.exists():
                self._bm25_indices[col_name] = BM25Index.load(bm25_path)

        # Over-fetch for reranking (Stage 1 → 3x candidates)
        rerank_k = top_k * 3 if self._reranker.is_active else top_k

        # Hybrid search: FAISS + BM25 → RRF
        results = self._hybrid.search(
            col_name, query,
            top_k=rerank_k,
            threshold=threshold,
        )

        # Stage 2: Cross-encoder rerank
        if self._reranker.is_active and results:
            results = self._reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        return results

    # ------------------------------------------------------------------
    # Incremental re-index
    # ------------------------------------------------------------------

    async def reindex_by_change(
        self,
        codebase_path: str,
        progress: ProgressCallback | None = None,
    ) -> dict[str, int]:
        """
        Incremental re-index using Merkle-tree diff (Cursor-style).

        Falls back to flat file-hash sync if Merkle tree not available.
        """
        codebase_path = os.path.abspath(codebase_path)
        col_name = self.get_collection_name(codebase_path)

        # Prefer Merkle tree sync (O(changes) instead of O(files))
        merkle = self._merkle_synchronizers.get(col_name)
        if merkle is not None:
            changes = await merkle.check_for_changes()
        else:
            # Fallback to flat sync
            sync = self._synchronizers.get(col_name)
            if sync is None:
                await self._load_ignore_patterns(codebase_path)
                sync = FileSynchronizer(codebase_path, self.ignore_patterns)
                await sync.initialize()
                self._synchronizers[col_name] = sync
            changes = await sync.check_for_changes()

        added, modified, removed = changes["added"], changes["modified"], changes["removed"]

        if not (added or modified or removed):
            logger.info("No file changes detected")
            return {"added": 0, "removed": 0, "modified": 0}

        logger.info("Changes: +%d ~%d -%d", len(added), len(modified), len(removed))

        # Delete chunks for removed / modified files via ref_doc_id
        bm25 = self._bm25_indices.get(col_name)
        for rel in removed + modified:
            self.vector_db.delete_ref_doc(col_name, rel)
            if bm25:
                bm25.delete_ref_doc(rel)

        # Re-index added + modified
        to_index = [os.path.join(codebase_path, r) for r in added + modified]
        if to_index:
            await self._process_file_list(to_index, codebase_path, progress)

        # Persist updated BM25 index
        if bm25:
            bm25_path = Path(self._cfg.data_dir) / "bm25_store" / f"{col_name}.json"
            bm25.save(bm25_path)

        # Persist embedding cache
        self._embedding_cache.save()

        return {"added": len(added), "removed": len(removed), "modified": len(modified)}

    # ------------------------------------------------------------------
    # Has index / clear index
    # ------------------------------------------------------------------

    async def has_index(self, codebase_path: str) -> bool:
        return self.vector_db.has_collection(
            self.get_collection_name(os.path.abspath(codebase_path))
        )

    async def clear_index(self, codebase_path: str) -> None:
        codebase_path = os.path.abspath(codebase_path)
        col = self.get_collection_name(codebase_path)
        if self.vector_db.has_collection(col):
            self.vector_db.drop_collection(col)
        # Clear BM25 index
        bm25 = self._bm25_indices.pop(col, None)
        if bm25:
            bm25.drop()
        bm25_path = Path(self._cfg.data_dir) / "bm25_store" / f"{col}.json"
        if bm25_path.exists():
            bm25_path.unlink()
        # Clear sync snapshots
        await FileSynchronizer.delete_snapshot(codebase_path)
        await MerkleSynchronizer.delete_snapshot(codebase_path, merkle_dir=Path(self._cfg.data_dir) / "merkle")
        self._merkle_synchronizers.pop(col, None)
        self._synchronizers.pop(col, None)
        logger.info("Cleared index for %s", codebase_path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _skip_unchanged_files(
        self,
        file_paths: list[str],
        codebase_path: str,
        merkle: MerkleSynchronizer,
    ) -> tuple[list[str], int]:
        """Filter out files whose content hash hasn't changed since last index.

        Compares each file's current SHA-256 against the saved Merkle tree.
        Returns (files_to_process, count_skipped).
        """
        saved_hashes = self._collect_saved_hashes(merkle._saved_tree)
        changed: list[str] = []
        skipped = 0

        for fpath in file_paths:
            rel = os.path.relpath(fpath, codebase_path)
            current_hash = MerkleSynchronizer._hash_file(fpath)
            saved_hash = saved_hashes.get(rel)

            if saved_hash and saved_hash == current_hash:
                # File unchanged — check if its chunks exist in FAISS
                col_name = self.get_collection_name(codebase_path)
                if self.vector_db.has_collection(col_name):
                    skipped += 1
                    continue

            changed.append(fpath)

        return changed, skipped

    @staticmethod
    def _collect_saved_hashes(node, prefix: str = "") -> dict[str, str]:
        """Flatten a Merkle tree into {relative_path: file_hash}."""
        if node is None:
            return {}
        result: dict[str, str] = {}
        if not node.is_dir:
            result[node.path] = node.hash
        else:
            for child in node.children.values():
                result.update(Context._collect_saved_hashes(child))
        return result

    def _prepare_collection(self, codebase_path: str, force: bool = False) -> None:
        col_name = self.get_collection_name(codebase_path)
        exists = self.vector_db.has_collection(col_name)

        if exists and not force:
            logger.info("Collection %s already exists, skipping", col_name)
            return
        if exists and force:
            logger.info("Force reindex — dropping %s", col_name)
            self.vector_db.drop_collection(col_name)
            # Also drop BM25 on force reindex
            bm25 = self._bm25_indices.pop(col_name, None)
            if bm25:
                bm25.drop()

        # Detect embedding dimension from the embed_model stored in vector_db
        dim = len(self.vector_db._embed_model.get_text_embedding("test"))
        self.vector_db.create_collection(
            col_name, dim, description=f"codebasePath:{codebase_path}"
        )

        # Initialize BM25 sparse index for this collection
        self._bm25_indices[col_name] = BM25Index()

        logger.info("Created collection %s (dim=%d) + BM25 index", col_name, dim)

    def _get_code_files(self, codebase_path: str) -> list[str]:
        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(codebase_path):
            dirnames[:] = [
                d for d in dirnames
                if not self._matches_ignore(os.path.join(dirpath, d), codebase_path)
            ]
            for f in filenames:
                full = os.path.join(dirpath, f)
                if self._matches_ignore(full, codebase_path):
                    continue
                ext = os.path.splitext(f)[1]
                if ext in self.supported_extensions:
                    files.append(full)
        return files

    def _matches_ignore(self, path: str, base: str) -> bool:
        rel = os.path.relpath(path, base)
        name = os.path.basename(path)
        if name.startswith("."):
            return True
        # Use compiled regex for fast matching (built lazily once)
        compiled = self._get_compiled_ignore()
        if compiled is not None and compiled.search(rel):
            return True
        if compiled is not None and compiled.search(name):
            return True
        # Fallback for suffix patterns
        for pat in self.ignore_patterns:
            if pat == name:
                return True
            if pat.endswith("/**") and rel.startswith(pat[:-3]):
                return True
        return False

    def _get_compiled_ignore(self):
        """Lazily compile ignore patterns into a single regex for speed."""
        if not hasattr(self, "_compiled_ignore_re"):
            import re
            from fnmatch import translate
            parts = []
            for pat in self.ignore_patterns:
                if not pat.endswith("/**"):
                    parts.append(translate(pat))
            if parts:
                combined = "|".join(parts)
                self._compiled_ignore_re = re.compile(combined)
            else:
                self._compiled_ignore_re = None
        return self._compiled_ignore_re

    def _read_and_split_file(self, fpath: str) -> tuple[str, list[CodeChunk] | None]:
        """Read a file and split it into chunks (runs in thread pool)."""
        try:
            content = Path(fpath).read_text(errors="replace")
        except Exception as exc:
            logger.warning("Skipping %s: %s", fpath, exc)
            return fpath, None
        ext = os.path.splitext(fpath)[1]
        language = EXTENSION_TO_LANGUAGE.get(ext, "text")
        chunks = self.splitter.split(content, language, fpath)
        return fpath, chunks

    async def _process_file_list(
        self,
        file_paths: list[str],
        codebase_path: str,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Process files with concurrent splitting + embedding pipeline.

        Architecture (Cursor-inspired):
          - Producer: thread pool reads + AST-splits files in parallel
          - Consumer: embeds + inserts chunks into FAISS/BM25
          - Pipeline: producer fills an asyncio.Queue, consumer drains it
            concurrently — embedding batch N while splitting batch N+1
        """
        batch_size = self._cfg.embedding_batch_size
        chunk_limit = self._cfg.chunk_limit
        col_name = self.get_collection_name(codebase_path)

        total_files = len(file_paths)
        logger.info("Starting indexing: %d files to process", total_files)

        loop = asyncio.get_event_loop()

        # --- Thread pool for parallel file reading + AST splitting ---
        from concurrent.futures import ThreadPoolExecutor
        max_workers = min(os.cpu_count() or 4, 24)  # I/O-bound: more workers = faster splitting
        executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info("Using %d workers for parallel file splitting (pipelined)", max_workers)

        # --- Async queue: decouples splitting (producer) from embedding (consumer) ---
        # Larger queue (8) allows more producer-consumer overlap so the
        # producer never stalls waiting for the consumer to embed.
        queue: asyncio.Queue[list[CodeChunk] | None] = asyncio.Queue(maxsize=32)

        import time as _time

        # Shared mutable state + timing
        state = {
            "processed": 0,
            "skipped": 0,
            "total_chunks": 0,
            "limit_reached": False,
            "cache_save_counter": 0,
            "split_time": 0.0,
            "embed_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        wall_start = _time.monotonic()

        # --- Producer: read + split files, push chunk batches to queue ---
        async def producer():
            buffer: list[CodeChunk] = []
            parallel_batch = max_workers * 4

            for batch_start in range(0, total_files, parallel_batch):
                if state["limit_reached"]:
                    break

                batch_end = min(batch_start + parallel_batch, total_files)
                batch_files = file_paths[batch_start:batch_end]

                # Split files in parallel via thread pool
                t0 = _time.monotonic()
                futures = [
                    loop.run_in_executor(executor, self._read_and_split_file, fpath)
                    for fpath in batch_files
                ]
                results = await asyncio.gather(*futures)
                state["split_time"] += _time.monotonic() - t0

                for i_in_batch, (fpath, chunks) in enumerate(results):
                    if state["limit_reached"]:
                        break

                    idx = batch_start + i_in_batch
                    rel = os.path.relpath(fpath, codebase_path)

                    if chunks is None:
                        state["skipped"] += 1
                        continue

                    logger.info("Processing file [%d/%d]: %s (%d chunks)",
                                idx + 1, total_files, rel, len(chunks))

                    for chunk in chunks:
                        buffer.append(chunk)
                        state["total_chunks"] += 1

                        if len(buffer) >= batch_size:
                            await queue.put(buffer)
                            buffer = []

                        if state["total_chunks"] >= chunk_limit:
                            logger.warning("Chunk limit (%d) reached", chunk_limit)
                            state["limit_reached"] = True
                            break

                    state["processed"] += 1
                    pct = 10 + int((idx + 1) / total_files * 90)
                    _report(progress, f"Processing ({idx+1}/{total_files})…",
                            idx + 1, total_files, pct)

            # Flush remaining buffer
            if buffer:
                await queue.put(buffer)

            # Signal consumer to stop
            await queue.put(None)

        # --- Consumer: embed + insert chunk batches from queue ---
        async def consumer():
            while True:
                batch = await queue.get()
                if batch is None:
                    break  # producer is done

                t0 = _time.monotonic()
                try:
                    await loop.run_in_executor(
                        None, self._flush_buffer, batch, codebase_path, col_name
                    )
                except Exception as exc:
                    logger.warning("Batch flush failed, skipping %d chunks: %s",
                                   len(batch), exc)
                state["embed_time"] += _time.monotonic() - t0

                # Periodically save embedding cache (every 50 batches)
                state["cache_save_counter"] += 1
                if state["cache_save_counter"] % 50 == 0:
                    self._embedding_cache.save()
                    # Periodic FAISS persist — spread I/O, enable crash recovery
                    self.vector_db.persist(col_name)

                queue.task_done()

        # --- Run producer and consumer concurrently ---
        await asyncio.gather(producer(), consumer())

        executor.shutdown(wait=False)

        # Final persist (all inserts used defer_persist=True)
        self.vector_db.persist(col_name)

        # Final cache save
        self._embedding_cache.save()

        wall_elapsed = _time.monotonic() - wall_start
        chunks = state["total_chunks"]
        throughput = chunks / wall_elapsed if wall_elapsed > 0 else 0
        overlap = max(0, state["split_time"] + state["embed_time"] - wall_elapsed)
        logger.info(
            "Indexing complete: %d/%d files processed, %d skipped, %d chunks, status=%s",
            state["processed"], total_files, state["skipped"], chunks,
            "limit_reached" if state["limit_reached"] else "completed",
        )
        logger.info(
            "⏱ Timing: wall=%.1fs, split=%.1fs, embed=%.1fs, overlap=%.1fs, throughput=%.1f chunks/s",
            wall_elapsed, state["split_time"], state["embed_time"], overlap, throughput,
        )

        return {
            "processed_files": state["processed"],
            "total_files": total_files,
            "skipped_files": state["skipped"],
            "total_chunks": state["total_chunks"],
            "status": "limit_reached" if state["limit_reached"] else "completed",
            "wall_time_s": round(wall_elapsed, 1),
            "split_time_s": round(state["split_time"], 1),
            "embed_time_s": round(state["embed_time"], 1),
            "overlap_s": round(overlap, 1),
            "throughput_chunks_per_s": round(throughput, 1),
        }

    def _flush_buffer(
        self, chunks: list[CodeChunk], codebase_path: str, col_name: str
    ) -> None:
        """Convert CodeChunks → LlamaIndex TextNodes → insert into FAISS + BM25.

        Uses embedding cache to skip re-embedding unchanged chunks (Cursor-style).
        Adds obfuscated paths to metadata for privacy.
        """
        nodes: list[TextNode] = []
        cached_nodes: list[tuple[TextNode, list[float]]] = []  # nodes with cached embeddings
        bm25 = self._bm25_indices.get(col_name)
        cache = self._embedding_cache

        # Truncate chunk text to stay within embedding model context limits.
        # nomic-embed-text has a 2048 token context; ~4 chars/token for code
        # gives a safe limit of ~8000 chars.  We cap at chunk_size as the max.
        max_embed_chars = self._cfg.chunk_size

        # Phase 1: Build all TextNodes and collect text hashes
        node_data: list[tuple[TextNode, str]] = []  # (node, text_hash)
        for chunk in chunks:
            text = chunk.content
            if len(text) > max_embed_chars:
                text = text[:max_embed_chars]
                logger.debug(
                    "Truncated chunk from %s (%d→%d chars)",
                    chunk.file_path, len(chunk.content), max_embed_chars,
                )

            rel = os.path.relpath(chunk.file_path, codebase_path) if chunk.file_path else ""
            ext = os.path.splitext(chunk.file_path)[1] if chunk.file_path else ""
            node_id = _generate_id(rel, chunk.start_line, chunk.end_line, text)

            # Path obfuscation — store encrypted path alongside plain path
            obfuscated_path = self._path_obfuscator.obfuscate(rel)

            node = TextNode(
                text=text,
                id_=node_id,
                metadata={
                    "relative_path": rel,
                    "obfuscated_path": obfuscated_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "file_extension": ext,
                    "language": chunk.language,
                    "codebase_path": codebase_path,
                },
            )
            # Set ref_doc_id to the relative path so delete_ref_doc can
            # remove all chunks belonging to a single file at once.
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=rel
            )
            # Don't embed metadata fields — only the code text.
            node.excluded_embed_metadata_keys = list(node.metadata.keys())
            node.excluded_llm_metadata_keys = list(node.metadata.keys())

            text_hash = cache.content_hash(text)
            node_data.append((node, text_hash))

        # Phase 2: Bulk cache lookup — single SQL query instead of N queries
        all_hashes = [h for _, h in node_data]
        cached_map = cache.get_batch(all_hashes)
        cache_hits = 0

        for node, text_hash in node_data:
            cached_embedding = cached_map.get(text_hash)
            if cached_embedding is not None:
                cached_nodes.append((node, cached_embedding))
                cache_hits += 1
            else:
                nodes.append(node)

            # Also add to BM25 sparse index (parallel indexing)
            if bm25:
                meta = node.metadata
                bm25.add_document(
                    doc_id=node.id_,
                    content=node.text,
                    relative_path=meta.get("relative_path", ""),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    language=meta.get("language", "text"),
                )

        if cache_hits:
            logger.info(
                "Embedding cache: %d hits, %d misses (skipping %d API calls)",
                cache_hits, len(nodes), cache_hits,
            )

        # Insert cached nodes directly with pre-computed embeddings (deferred persist)
        if cached_nodes:
            self._insert_cached_nodes(col_name, cached_nodes, defer_persist=True)

        # Embed and insert uncached nodes (calls embedding API)
        if nodes:
            # Pre-compute embeddings so we can cache them AND pass to FAISS.
            # Use concurrent sub-batches to exploit OLLAMA_NUM_PARALLEL slots.
            embed_model = self.vector_db._embed_model
            try:
                texts = [n.text for n in nodes]
                num_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "1"))
                if num_parallel > 1 and len(texts) > num_parallel:
                    sub_batch_size = max(10, len(texts) // num_parallel)
                    sub_batches = [
                        texts[i:i + sub_batch_size]
                        for i in range(0, len(texts), sub_batch_size)
                    ]
                    from concurrent.futures import ThreadPoolExecutor as _TPE
                    with _TPE(max_workers=len(sub_batches)) as emb_pool:
                        emb_results = list(emb_pool.map(
                            embed_model.get_text_embedding_batch, sub_batches
                        ))
                    embeddings = [e for batch_embs in emb_results for e in batch_embs]
                else:
                    embeddings = embed_model.get_text_embedding_batch(texts)

                for node, emb in zip(nodes, embeddings):
                    node.embedding = emb
                    cache.put(cache.content_hash(node.text), emb)
            except Exception as exc:
                logger.warning("Pre-embedding failed, falling back to insert_nodes: %s", exc)

            self._insert_nodes_safe(col_name, nodes, defer_persist=True)

    def _insert_nodes_safe(
        self, col_name: str, nodes: list[TextNode], *, defer_persist: bool = False
    ) -> None:
        """Insert nodes into FAISS, retrying individual nodes on context-length errors."""
        try:
            self.vector_db.insert_nodes(col_name, nodes, defer_persist=defer_persist)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "context length" in exc_str or "too long" in exc_str or "exceeds" in exc_str:
                logger.warning(
                    "Batch embedding failed (context length). "
                    "Retrying %d nodes individually with truncation…", len(nodes),
                )
                for node in nodes:
                    self._insert_single_node_safe(col_name, node)
            else:
                raise

    def _insert_single_node_safe(self, col_name: str, node: TextNode) -> None:
        """Insert a single node, progressively truncating on context-length errors."""
        text = node.text
        for attempt in range(4):  # try full, 75%, 50%, 25%
            try:
                self.vector_db.insert_nodes(col_name, [node])
                return
            except Exception as exc:
                exc_str = str(exc).lower()
                if "context length" in exc_str or "too long" in exc_str or "exceeds" in exc_str:
                    # Truncate to smaller fraction
                    fraction = [1.0, 0.75, 0.5, 0.25][min(attempt + 1, 3)]
                    new_len = int(len(text) * fraction)
                    node.text = text[:new_len]
                    logger.debug(
                        "Truncating node %s to %d chars (attempt %d)",
                        node.metadata.get("relative_path", "?"), new_len, attempt + 1,
                    )
                else:
                    logger.warning(
                        "Skipping node %s: %s",
                        node.metadata.get("relative_path", "?"), exc,
                    )
                    return
        logger.warning(
            "Skipping node %s after 4 truncation attempts",
            node.metadata.get("relative_path", "?"),
        )

    def _insert_cached_nodes(
        self, col_name: str, cached_nodes: list[tuple[TextNode, list[float]]],
        *, defer_persist: bool = False,
    ) -> None:
        """Insert nodes with pre-computed embeddings directly into FAISS (no API call)."""
        index = self.vector_db._indices.get(col_name)
        if index is None:
            return

        for node, embedding in cached_nodes:
            # Set the embedding directly on the node so LlamaIndex skips the API call
            node.embedding = embedding

        nodes = [n for n, _ in cached_nodes]

        # Clear soft-deletions for re-inserted ref_doc_ids
        deleted = self.vector_db._deleted_refs.get(col_name)
        if deleted:
            for node in nodes:
                ref = node.ref_doc_id
                if ref and ref in deleted:
                    deleted.discard(ref)

        index.insert_nodes(nodes)
        if not defer_persist:
            self.vector_db._persist(col_name)

    async def _load_ignore_patterns(self, codebase_path: str) -> None:
        # 1. Local ignore files (.gitignore, .contextignore)
        for name in (".gitignore", ".contextignore"):
            ignore_file = os.path.join(codebase_path, name)
            if os.path.isfile(ignore_file):
                try:
                    lines = Path(ignore_file).read_text().splitlines()
                    patterns = [
                        l.strip() for l in lines
                        if l.strip() and not l.strip().startswith("#")
                    ]
                    for p in patterns:
                        if p not in self.ignore_patterns:
                            self.ignore_patterns.append(p)
                    logger.info("Loaded %d patterns from %s", len(patterns), name)
                except Exception as exc:
                    logger.warning("Could not read %s: %s", ignore_file, exc)

        # 2. Global ignore file (~/.context/.contextignore) — mirrors TS loadGlobalIgnoreFile
        global_ignore = CONTEXT_DIR / ".contextignore"
        if global_ignore.is_file():
            try:
                lines = global_ignore.read_text().splitlines()
                patterns = [
                    l.strip() for l in lines
                    if l.strip() and not l.strip().startswith("#")
                ]
                for p in patterns:
                    if p not in self.ignore_patterns:
                        self.ignore_patterns.append(p)
                logger.info("Loaded %d global ignore patterns from %s", len(patterns), global_ignore)
            except Exception as exc:
                logger.warning("Could not read global ignore file: %s", exc)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _generate_id(rel_path: str, start: int, end: int, content: str) -> str:
    raw = f"{rel_path}:{start}:{end}:{content}"
    h = hashlib.sha256(raw.encode()).hexdigest()
    return f"chunk_{h[:16]}"


def _report(cb: ProgressCallback | None, phase: str, current: int, total: int, pct: int) -> None:
    if cb:
        cb(phase, current, total, pct)
