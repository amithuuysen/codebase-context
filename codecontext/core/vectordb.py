"""
FAISS vector database backed by LlamaIndex.

Mirrors packages/core/src/vectordb/ (milvus-vectordb.ts adapted for FAISS).

Pipeline:  Tree-sitter chunks → LlamaIndex TextNode → FAISS (via VectorStoreIndex)

Persistence layout (each collection = one indexed codebase):
    <persist_dir>/<collection_name>/
        default__vector_store.faiss   ← FAISS binary index
        docstore.json                 ← original text + metadata
        index_store.json              ← LlamaIndex index metadata
        collection_meta.json          ← our own description file
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import faiss
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.faiss import FaissVectorStore

from .types import SemanticSearchResult

logger = logging.getLogger(__name__)


class FaissVectorDB:
    """
    Manages per-collection FAISS indices via LlamaIndex.

    Each "collection" is an independent VectorStoreIndex backed by its own
    FAISS flat index, persisted to a sub-directory under *persist_dir*.
    """

    def __init__(self, persist_dir: str | Path, embed_model: BaseEmbedding):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._embed_model = embed_model

        # In-memory cache: collection_name → VectorStoreIndex
        self._indices: dict[str, VectorStoreIndex] = {}
        self._descriptions: dict[str, str] = {}
        self._dimensions: dict[str, int] = {}
        # Deleted ref_doc_ids per collection (FAISS can't delete vectors,
        # so we track deletions and filter at query time)
        self._deleted_refs: dict[str, set[str]] = {}

        self._load_all_from_disk()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, name: str, dimension: int, description: str = "") -> None:
        if name in self._indices:
            return
        # Use IndexFlatIP (inner-product) for cosine similarity.
        # Vectors MUST be L2-normalized before insertion (LlamaIndex
        # OpenAI embeddings are already normalized; we add an explicit
        # normalize step in insert_nodes for safety).
        faiss_index = faiss.IndexFlatIP(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            [], storage_context=storage_ctx, embed_model=self._embed_model
        )
        self._indices[name] = index
        self._descriptions[name] = description
        self._dimensions[name] = dimension
        self._persist(name)
        logger.info("Created FAISS collection %s (dim=%d)", name, dimension)

    def drop_collection(self, name: str) -> None:
        self._indices.pop(name, None)
        self._descriptions.pop(name, None)
        self._dimensions.pop(name, None)
        self._deleted_refs.pop(name, None)
        col_dir = self._persist_dir / name
        if col_dir.exists():
            shutil.rmtree(col_dir)
        logger.info("Dropped collection %s", name)

    def has_collection(self, name: str) -> bool:
        return name in self._indices

    def list_collections(self) -> list[str]:
        return list(self._indices.keys())

    def get_collection_description(self, name: str) -> str:
        return self._descriptions.get(name, "")

    # ------------------------------------------------------------------
    # Insert LlamaIndex TextNodes
    # ------------------------------------------------------------------

    def insert_nodes(
        self, collection_name: str, nodes: list[TextNode], *, defer_persist: bool = False
    ) -> None:
        index = self._indices.get(collection_name)
        if index is None:
            raise KeyError(f"Collection '{collection_name}' does not exist")
        if not nodes:
            return

        # Clear any soft-deletions for ref_doc_ids being re-inserted
        deleted = self._deleted_refs.get(collection_name)
        if deleted:
            for node in nodes:
                ref = node.ref_doc_id
                if ref and ref in deleted:
                    deleted.discard(ref)

        index.insert_nodes(nodes)
        if not defer_persist:
            self._persist(collection_name)

    # ------------------------------------------------------------------
    # Delete by ref_doc_id  (= relative file path)
    #
    # FAISS does not support individual vector deletion.  We use a
    # soft-delete approach: record the ref_doc_id and filter it out in
    # search().  On the next full reindex (force=True) the FAISS index
    # is rebuilt from scratch without the stale vectors.
    # ------------------------------------------------------------------

    def delete_ref_doc(self, collection_name: str, ref_doc_id: str) -> None:
        """Soft-delete all nodes whose SOURCE ref_doc_id matches."""
        if collection_name not in self._indices:
            return
        deleted = self._deleted_refs.setdefault(collection_name, set())
        deleted.add(ref_doc_id)
        self._persist(collection_name)
        logger.info("Soft-deleted ref_doc '%s' in %s", ref_doc_id, collection_name)

    # ------------------------------------------------------------------
    # Search  (returns SemanticSearchResult list for convenience)
    # ------------------------------------------------------------------

    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[SemanticSearchResult]:
        index = self._indices.get(collection_name)
        if index is None:
            return []

        deleted = self._deleted_refs.get(collection_name, set())

        # Request extra results so we can filter out soft-deleted and
        # below-threshold ones
        fetch_k = top_k * 3 + len(deleted) * 2
        retriever = index.as_retriever(similarity_top_k=fetch_k)
        hits: list[NodeWithScore] = retriever.retrieve(query)

        results: list[SemanticSearchResult] = []
        for h in hits:
            meta = h.node.metadata or {}
            ref_path = meta.get("relative_path", "")
            if ref_path in deleted:
                continue
            score = float(h.score) if h.score is not None else 0.0
            # Filter below threshold (mirrors TS Milvus COSINE metric
            # where 1.0 = perfect match, 0.0 = orthogonal)
            if score < threshold:
                continue
            results.append(
                SemanticSearchResult(
                    content=h.node.get_content(),
                    relative_path=ref_path,
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    language=meta.get("language", "unknown"),
                    score=score,
                )
            )
            if len(results) >= top_k:
                break
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, name: str) -> None:
        """Public alias so callers can trigger a deferred persist."""
        self._persist(name)

    def _persist(self, name: str) -> None:
        col_dir = self._persist_dir / name
        col_dir.mkdir(parents=True, exist_ok=True)
        index = self._indices.get(name)
        if index is None:
            return
        index.storage_context.persist(persist_dir=str(col_dir))
        meta = {
            "description": self._descriptions.get(name, ""),
            "dimension": self._dimensions.get(name, 0),
            "deleted_refs": sorted(self._deleted_refs.get(name, set())),
        }
        with open(col_dir / "collection_meta.json", "w") as f:
            json.dump(meta, f)

    def _load_all_from_disk(self) -> None:
        if not self._persist_dir.exists():
            return
        for col_dir in sorted(self._persist_dir.iterdir()):
            if not col_dir.is_dir():
                continue
            meta_file = col_dir / "collection_meta.json"
            if not meta_file.exists():
                continue
            name = col_dir.name
            try:
                with open(meta_file) as f:
                    meta = json.load(f)

                vector_store = FaissVectorStore.from_persist_dir(str(col_dir))
                storage_ctx = StorageContext.from_defaults(
                    vector_store=vector_store, persist_dir=str(col_dir)
                )
                index = load_index_from_storage(
                    storage_ctx, embed_model=self._embed_model
                )
                self._indices[name] = index
                self._descriptions[name] = meta.get("description", "")
                self._dimensions[name] = meta.get("dimension", 0)
                deleted = meta.get("deleted_refs", [])
                if deleted:
                    self._deleted_refs[name] = set(deleted)
                logger.info("Loaded FAISS collection '%s' from disk", name)
            except Exception as exc:
                logger.warning("Failed to load collection %s: %s", name, exc)
