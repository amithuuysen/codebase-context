"""
Cross-encoder reranker — refines hybrid search results.

After BM25 + FAISS fusion via RRF returns a candidate set, a cross-encoder
scores each (query, chunk) pair with full attention.  This is much more
accurate than bi-encoder similarity but too slow for first-stage retrieval.

Architecture (Cursor-inspired two-stage pipeline):
  Stage 1: Fast retrieval   — FAISS (dense) + BM25 (sparse) → RRF top-N
  Stage 2: Precision rerank — cross-encoder scores top-N → final top-K

Supported backends:
  - "local"   → sentence-transformers CrossEncoder (runs on CPU/GPU locally)
  - "none"    → pass-through (no reranking, just return inputs)
"""

from __future__ import annotations

import logging
from typing import Any

from .types import SemanticSearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker.

    Usage:
        reranker = Reranker(provider="local", model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranked = reranker.rerank(query, candidates, top_k=10)
    """

    def __init__(self, provider: str = "none", model: str = ""):
        self.provider = provider
        self._model: Any = None
        self._model_name = model

        if provider == "local" and model:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(model)
                logger.info("Loaded cross-encoder reranker: %s", model)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers. "
                    "Falling back to no reranking."
                )
                self.provider = "none"
            except Exception as exc:
                logger.warning("Failed to load reranker model %s: %s", model, exc)
                self.provider = "none"

    def rerank(
        self,
        query: str,
        candidates: list[SemanticSearchResult],
        top_k: int = 10,
    ) -> list[SemanticSearchResult]:
        """
        Rerank candidates using the cross-encoder.

        Args:
            query: The search query.
            candidates: Pre-ranked results from hybrid search.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked list of SemanticSearchResult, scored by cross-encoder.
        """
        if self.provider == "none" or self._model is None:
            return candidates[:top_k]

        if not candidates:
            return []

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, c.content) for c in candidates]

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            logger.warning("Reranker prediction failed: %s", exc)
            return candidates[:top_k]

        # Attach scores and sort
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        results = []
        for candidate, score in scored[:top_k]:
            results.append(SemanticSearchResult(
                content=candidate.content,
                relative_path=candidate.relative_path,
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                language=candidate.language,
                score=float(score),
            ))
        return results

    @property
    def is_active(self) -> bool:
        return self.provider != "none" and self._model is not None
