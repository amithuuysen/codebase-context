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
from .hybrid_search import HybridSearcher
from .merkle import MerkleSynchronizer
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

        logger.info(
            "Context initialized — %d extensions, %d ignore patterns, "
            "reranker=%s",
            len(self.supported_extensions),
            len(self.ignore_patterns),
            self._reranker.provider,
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
        if not files:
            _report(progress, "No files to index", 100, 100, 100)
            return {"indexed_files": 0, "total_chunks": 0, "status": "completed"}

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
            merkle = MerkleSynchronizer(codebase_path, self.ignore_patterns)
            await merkle.initialize()
            self._merkle_synchronizers[col] = merkle
        merkle.save_current_state()

        # Persist BM25 index
        bm25 = self._bm25_indices.get(col)
        if bm25:
            bm25_path = Path(self._cfg.data_dir) / "bm25_store" / f"{col}.json"
            bm25.save(bm25_path)

        logger.info(
            "Indexing complete: %d files, %d chunks",
            result["processed_files"], result["total_chunks"],
        )
        _report(progress, "Indexing complete!", result["processed_files"], len(files), 100)
        return {
            "indexed_files": result["processed_files"],
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
        await MerkleSynchronizer.delete_snapshot(codebase_path)
        self._merkle_synchronizers.pop(col, None)
        self._synchronizers.pop(col, None)
        logger.info("Cleared index for %s", codebase_path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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
        from fnmatch import fnmatch
        rel = os.path.relpath(path, base)
        name = os.path.basename(path)
        if name.startswith("."):
            return True
        for pat in self.ignore_patterns:
            if pat == name:
                return True
            if fnmatch(rel, pat) or fnmatch(name, pat):
                return True
            if pat.endswith("/**") and rel.startswith(pat[:-3]):
                return True
        return False

    async def _process_file_list(
        self,
        file_paths: list[str],
        codebase_path: str,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        batch_size = self._cfg.embedding_batch_size
        chunk_limit = self._cfg.chunk_limit
        buffer: list[CodeChunk] = []
        processed = 0
        total_chunks = 0
        limit_reached = False
        col_name = self.get_collection_name(codebase_path)

        for i, fpath in enumerate(file_paths):
            try:
                content = Path(fpath).read_text(errors="replace")
            except Exception as exc:
                logger.warning("Skipping %s: %s", fpath, exc)
                continue

            ext = os.path.splitext(fpath)[1]
            language = EXTENSION_TO_LANGUAGE.get(ext, "text")
            chunks = self.splitter.split(content, language, fpath)

            for chunk in chunks:
                buffer.append(chunk)
                total_chunks += 1

                if len(buffer) >= batch_size:
                    self._flush_buffer(buffer, codebase_path, col_name)
                    buffer = []

                if total_chunks >= chunk_limit:
                    logger.warning("Chunk limit (%d) reached", chunk_limit)
                    limit_reached = True
                    break

            processed += 1
            pct = 10 + int((i + 1) / len(file_paths) * 90)
            _report(progress, f"Processing ({i+1}/{len(file_paths)})…", i + 1, len(file_paths), pct)
            if limit_reached:
                break

        if buffer:
            self._flush_buffer(buffer, codebase_path, col_name)

        return {
            "processed_files": processed,
            "total_chunks": total_chunks,
            "status": "limit_reached" if limit_reached else "completed",
        }

    def _flush_buffer(
        self, chunks: list[CodeChunk], codebase_path: str, col_name: str
    ) -> None:
        """Convert CodeChunks → LlamaIndex TextNodes → insert into FAISS + BM25."""
        nodes: list[TextNode] = []
        bm25 = self._bm25_indices.get(col_name)

        for chunk in chunks:
            rel = os.path.relpath(chunk.file_path, codebase_path) if chunk.file_path else ""
            ext = os.path.splitext(chunk.file_path)[1] if chunk.file_path else ""
            node_id = _generate_id(rel, chunk.start_line, chunk.end_line, chunk.content)

            node = TextNode(
                text=chunk.content,
                id_=node_id,
                metadata={
                    "relative_path": rel,
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
            nodes.append(node)

            # Also add to BM25 sparse index (parallel indexing)
            if bm25:
                bm25.add_document(
                    doc_id=node_id,
                    content=chunk.content,
                    relative_path=rel,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    language=chunk.language,
                )

        self.vector_db.insert_nodes(col_name, nodes)

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
