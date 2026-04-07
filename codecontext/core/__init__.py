"""
codecontext.core — Core library for semantic code search.

Mirrors packages/core/src/ from the TypeScript project.

Re-exports all public types so callers can do:
    from codecontext.core import Context, FaissVectorDB, AstSplitter, ...
"""

from .types import (
    CONTEXT_DIR,
    DEFAULT_IGNORE_PATTERNS,
    DEFAULT_SUPPORTED_EXTENSIONS,
    EXTENSION_TO_LANGUAGE,
    MERKLE_DIR,
    SNAPSHOT_FILE,
    CodeChunk,
    Config,
    SemanticSearchResult,
)
from .embedding import create_embedding
from .bm25 import BM25Index
from .embedding_cache import EmbeddingCache
from .hybrid_search import HybridSearcher, reciprocal_rank_fusion
from .merkle import MerkleSynchronizer
from .path_obfuscation import PathObfuscator
from .reranker import Reranker
from .splitter import AstSplitter, Splitter, TextSplitter
from .sync import FileSynchronizer
from .vectordb import FaissVectorDB
from .context import Context

__all__ = [
    # Types & constants
    "CONTEXT_DIR",
    "DEFAULT_IGNORE_PATTERNS",
    "DEFAULT_SUPPORTED_EXTENSIONS",
    "EXTENSION_TO_LANGUAGE",
    "MERKLE_DIR",
    "SNAPSHOT_FILE",
    "CodeChunk",
    "Config",
    "SemanticSearchResult",
    # Embedding
    "create_embedding",
    # BM25 sparse search
    "BM25Index",
    # Hybrid search (FAISS + BM25 + RRF)
    "HybridSearcher",
    "reciprocal_rank_fusion",
    # Merkle tree sync
    "MerkleSynchronizer",
    # Reranker
    "Reranker",
    # Splitter
    "AstSplitter",
    "Splitter",
    "TextSplitter",
    # Sync
    "FileSynchronizer",
    # VectorDB
    "FaissVectorDB",
    # Context
    "Context",
]
