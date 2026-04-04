"""
Hybrid search — fuses dense (FAISS) and sparse (BM25) retrieval via RRF.

Inspired by Cursor's architecture:
  - Semantic search (FAISS vectors) finds code by *meaning*
  - BM25 keyword search (like Cursor's Instant Grep) finds exact matches
  - RRF (Reciprocal Rank Fusion) merges both ranked lists into one
  - Optional cross-encoder reranker refines the final top-K

Research shows this hybrid approach is 12.5% more accurate than either alone
(Cursor blog: "Improving agent with semantic search", Nov 2025).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .bm25 import BM25Index
from .types import SemanticSearchResult
from .vectordb import FaissVectorDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists using RRF.

    For each item, its fused score is:
        sum over lists of  1 / (k + rank_in_list)

    where k=60 is the standard constant (from the original RRF paper).

    Args:
        ranked_lists: Each list is [(doc_id, score), ...] in descending order.
        k: RRF constant (default 60).

    Returns:
        Fused list of (doc_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}
    for rlist in ranked_lists:
        for rank, (doc_id, _score) in enumerate(rlist, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Hybrid searcher
# ---------------------------------------------------------------------------

class HybridSearcher:
    """
    Combines FAISS (dense) + BM25 (sparse) search with RRF fusion.

    Usage:
        searcher = HybridSearcher(vector_db, bm25_indices)
        results = searcher.search(collection, query, top_k=10)
    """

    def __init__(
        self,
        vector_db: FaissVectorDB,
        bm25_indices: dict[str, BM25Index],
        *,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.vector_db = vector_db
        self.bm25_indices = bm25_indices
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
        dense_only: bool = False,
    ) -> list[SemanticSearchResult]:
        """
        Hybrid search: dense FAISS + sparse BM25, fused with RRF.

        Args:
            collection_name: The FAISS collection to search.
            query: Natural-language query string.
            top_k: Max results to return.
            threshold: Minimum FAISS cosine similarity threshold.
            dense_only: If True, skip BM25 (fallback to pure semantic).
        """
        # --- 1. Dense retrieval (FAISS) ---
        fetch_dense = top_k * 3  # Over-fetch for fusion
        dense_results = self.vector_db.search(
            collection_name, query,
            top_k=fetch_dense,
            threshold=threshold,
        )
        # Convert to (doc_id, score) for RRF
        dense_ranked: list[tuple[str, float]] = []
        dense_map: dict[str, SemanticSearchResult] = {}
        for r in dense_results:
            doc_id = f"{r.relative_path}:{r.start_line}-{r.end_line}"
            dense_ranked.append((doc_id, r.score))
            dense_map[doc_id] = r

        if dense_only:
            return dense_results[:top_k]

        # --- 2. Sparse retrieval (BM25) ---
        bm25 = self.bm25_indices.get(collection_name)
        sparse_ranked: list[tuple[str, float]] = []
        sparse_map: dict[str, SemanticSearchResult] = {}

        if bm25 and bm25.doc_count > 0:
            bm25_hits = bm25.search(query, top_k=fetch_dense)
            for doc_id, score in bm25_hits:
                doc = bm25.get_doc(doc_id)
                if doc is None:
                    continue
                result_key = f"{doc.relative_path}:{doc.start_line}-{doc.end_line}"
                sparse_ranked.append((result_key, score))
                if result_key not in sparse_map:
                    sparse_map[result_key] = SemanticSearchResult(
                        content=doc.content,
                        relative_path=doc.relative_path,
                        start_line=doc.start_line,
                        end_line=doc.end_line,
                        language=doc.language,
                        score=score,
                    )

        # --- 3. RRF fusion ---
        if not sparse_ranked:
            # No BM25 results — fall back to pure dense
            return dense_results[:top_k]

        fused = reciprocal_rank_fusion(
            [dense_ranked, sparse_ranked],
            k=self.rrf_k,
        )

        # --- 4. Build final result list ---
        results: list[SemanticSearchResult] = []
        for doc_id, rrf_score in fused:
            # Prefer the dense result (has actual cosine score) if available
            if doc_id in dense_map:
                r = dense_map[doc_id]
                results.append(SemanticSearchResult(
                    content=r.content,
                    relative_path=r.relative_path,
                    start_line=r.start_line,
                    end_line=r.end_line,
                    language=r.language,
                    score=rrf_score,
                ))
            elif doc_id in sparse_map:
                r = sparse_map[doc_id]
                results.append(SemanticSearchResult(
                    content=r.content,
                    relative_path=r.relative_path,
                    start_line=r.start_line,
                    end_line=r.end_line,
                    language=r.language,
                    score=rrf_score,
                ))
            if len(results) >= top_k:
                break

        return results
