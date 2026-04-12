"""
BM25 sparse keyword index — local, in-memory keyword search.

Inspired by Cursor's Instant Grep: provides exact and keyword-based matching
alongside the dense FAISS vector search.  The two are fused via RRF (Reciprocal
Rank Fusion) in hybrid_search().

BM25 (Best Matching 25) is the standard probabilistic relevance function used
by Elasticsearch, Lucene, etc.  We implement a minimal pure-Python version to
avoid external dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Split on non-alphanumeric, keep camelCase / snake_case parts
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")

_STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "by", "from", "as", "into", "about",
    "this", "that", "it", "its", "and", "or", "not", "but", "if", "then",
    "else", "so", "no", "nor", "up", "out",
}


def tokenize(text: str) -> list[str]:
    """Tokenize source code for BM25 indexing."""
    tokens: list[str] = []
    # Split on non-alnum first
    parts = _SPLIT_RE.split(text)
    for part in parts:
        # Split camelCase
        sub_parts = _CAMEL_RE.split(part)
        for t in sub_parts:
            t_lower = t.lower()
            if len(t_lower) >= 2 and t_lower not in _STOP_WORDS:
                tokens.append(t_lower)
    return tokens


# ---------------------------------------------------------------------------
# BM25 Document
# ---------------------------------------------------------------------------

class BM25Doc:
    """Holds metadata for a single indexed document (code chunk)."""
    __slots__ = ("doc_id", "relative_path", "start_line", "end_line",
                 "language", "content", "token_count")

    def __init__(
        self,
        doc_id: str,
        relative_path: str,
        start_line: int,
        end_line: int,
        language: str,
        content: str,
        token_count: int,
    ):
        self.doc_id = doc_id
        self.relative_path = relative_path
        self.start_line = start_line
        self.end_line = end_line
        self.language = language
        self.content = content
        self.token_count = token_count


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    In-memory BM25 index for a single collection (codebase).

    Parameters follow the standard BM25 formulation:
      - k1 controls term-frequency saturation (default 1.5)
      - b  controls length normalization    (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # doc_id → BM25Doc
        self._docs: dict[str, BM25Doc] = {}

        # Inverted index: token → set of doc_ids
        self._inv: dict[str, set[str]] = defaultdict(set)

        # Term frequencies: (doc_id, token) → count
        self._tf: dict[tuple[str, str], int] = {}

        # Total docs and average doc length (for BM25 normalization)
        self._total_docs = 0
        self._avg_dl = 0.0

        # Soft-deleted ref_doc_ids (file-level delete, mirrors FAISS)
        self._deleted_refs: set[str] = set()

    # ------------------------------------------------------------------
    # Insert / delete
    # ------------------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        content: str,
        relative_path: str,
        start_line: int = 0,
        end_line: int = 0,
        language: str = "text",
    ) -> None:
        """Index a single code chunk."""
        tokens = tokenize(content)
        token_count = len(tokens)

        self._docs[doc_id] = BM25Doc(
            doc_id=doc_id,
            relative_path=relative_path,
            start_line=start_line,
            end_line=end_line,
            language=language,
            content=content,
            token_count=token_count,
        )

        # Build inverted index + TF
        freq: dict[str, int] = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        for t, count in freq.items():
            self._inv[t].add(doc_id)
            self._tf[(doc_id, t)] = count

        # Update average doc length
        self._total_docs += 1
        self._avg_dl = (
            (self._avg_dl * (self._total_docs - 1) + token_count)
            / self._total_docs
        )

        # Un-delete if the file was previously soft-deleted
        self._deleted_refs.discard(relative_path)

    def delete_ref_doc(self, ref_doc_id: str) -> None:
        """Soft-delete all docs with a given relative_path."""
        self._deleted_refs.add(ref_doc_id)

    def drop(self) -> None:
        """Clear the entire index."""
        self._docs.clear()
        self._inv.clear()
        self._tf.clear()
        self._deleted_refs.clear()
        self._total_docs = 0
        self._avg_dl = 0.0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        BM25 search.  Returns list of (doc_id, score) sorted descending.
        """
        query_tokens = tokenize(query)
        if not query_tokens or self._total_docs == 0:
            return []

        scores: dict[str, float] = defaultdict(float)

        for token in query_tokens:
            doc_ids = self._inv.get(token)
            if not doc_ids:
                continue
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            df = len(doc_ids)
            idf = math.log((self._total_docs - df + 0.5) / (df + 0.5) + 1.0)

            for did in doc_ids:
                doc = self._docs.get(did)
                if doc is None:
                    continue
                # Skip soft-deleted
                if doc.relative_path in self._deleted_refs:
                    continue

                tf = self._tf.get((did, token), 0)
                dl = doc.token_count
                # BM25 TF normalization
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
                )
                scores[did] += idf * tf_norm

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_doc(self, doc_id: str) -> BM25Doc | None:
        return self._docs.get(doc_id)

    @property
    def doc_count(self) -> int:
        return self._total_docs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize the BM25 index to a binary (pickle) file.

        Falls back to JSON if the path ends with .json (legacy compat).
        Pickle is ~5-10x faster than JSON for large indices (900 MB+).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer binary .pkl format
        pkl_path = path.with_suffix(".pkl") if path.suffix == ".json" else path

        import pickle
        data = {
            "k1": self.k1,
            "b": self.b,
            "total_docs": self._total_docs,
            "avg_dl": self._avg_dl,
            "deleted_refs": self._deleted_refs,
            "docs": self._docs,
            "inv": self._inv,
            "tf": self._tf,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """Deserialize a BM25 index from a pickle or JSON file."""
        path = Path(path)

        # Try binary pickle first (faster)
        pkl_path = path.with_suffix(".pkl") if path.suffix == ".json" else path
        if pkl_path.exists():
            return cls._load_pickle(pkl_path)

        # Fall back to legacy JSON
        if path.exists() and path.suffix == ".json":
            return cls._load_json(path)

        return cls()

    @classmethod
    def _load_pickle(cls, path: Path) -> "BM25Index":
        """Load from binary pickle format."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

        idx = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        idx._total_docs = data.get("total_docs", 0)
        idx._avg_dl = data.get("avg_dl", 0.0)
        idx._deleted_refs = data.get("deleted_refs", set())
        idx._docs = data.get("docs", {})
        idx._inv = data.get("inv", defaultdict(set))
        idx._tf = data.get("tf", {})
        return idx

    @classmethod
    def _load_json(cls, path: Path) -> "BM25Index":
        """Load from legacy JSON format (auto-migrated on next save)."""
        with open(path) as f:
            data = json.load(f)

        idx = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        idx._total_docs = data.get("total_docs", 0)
        idx._avg_dl = data.get("avg_dl", 0.0)
        idx._deleted_refs = set(data.get("deleted_refs", []))

        for did, d in data.get("docs", {}).items():
            idx._docs[did] = BM25Doc(
                doc_id=did,
                relative_path=d["rp"],
                start_line=d["sl"],
                end_line=d["el"],
                language=d["lang"],
                content=d["text"],
                token_count=d["tc"],
            )

        for token, dids in data.get("inv", {}).items():
            idx._inv[token] = set(dids)

        for did, tok, cnt in data.get("tf", []):
            idx._tf[(did, tok)] = cnt

        logger.info("Loaded BM25 from legacy JSON, will save as pickle next time")
        return idx
