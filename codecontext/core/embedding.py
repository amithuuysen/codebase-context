"""
Embedding provider factory — returns LlamaIndex BaseEmbedding instances.

Mirrors packages/core/src/embedding/ (openai-embedding.ts, ollama-embedding.ts, etc.)

Pipeline: Tree-sitter chunks → LlamaIndex TextNode → **embed_model** → FAISS

Supported providers:
  - openai   → llama-index-embeddings-openai
  - ollama   → llama-index-embeddings-ollama  (pip install codecontext[ollama])
  - local    → llama-index-embeddings-huggingface (pip install codecontext[local])
"""

from __future__ import annotations

import logging

from llama_index.core.embeddings import BaseEmbedding

__all__ = ["create_embedding"]

logger = logging.getLogger(__name__)


def create_embedding(provider: str, **kwargs) -> BaseEmbedding:
    """
    Instantiate the right LlamaIndex embedding model from a provider name.

    Keyword args are forwarded to the underlying LlamaIndex class.  Common
    ones: model / model_name, api_key, base_url, embed_batch_size.
    """
    provider = provider.lower().strip()

    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        init: dict = {}
        if kwargs.get("api_key"):
            init["api_key"] = kwargs["api_key"]
        if kwargs.get("model"):
            init["model"] = kwargs["model"]
        if kwargs.get("base_url"):
            init["api_base"] = kwargs["base_url"]

        embed_model = OpenAIEmbedding(**init)
        logger.info("Embedding: OpenAI model=%s", init.get("model", "default"))
        return embed_model

    elif provider == "ollama":
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
        except ImportError:
            raise ImportError(
                "llama-index-embeddings-ollama is required.  "
                "Install with: pip install 'codecontext[ollama]'"
            )

        embed_model = OllamaEmbedding(
            model_name=kwargs.get("model", "nomic-embed-text"),
            base_url=kwargs.get("host", "http://127.0.0.1:11434"),
            embed_batch_size=100,
        )
        logger.info("Embedding: Ollama model=%s", kwargs.get("model"))
        return embed_model

    elif provider == "local":
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError:
            raise ImportError(
                "llama-index-embeddings-huggingface is required.  "
                "Install with: pip install 'codecontext[local]'"
            )

        embed_model = HuggingFaceEmbedding(
            model_name=kwargs.get("model", "all-MiniLM-L6-v2"),
        )
        logger.info("Embedding: HuggingFace model=%s", kwargs.get("model"))
        return embed_model

    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'.  "
            "Choose from: openai, ollama, local"
        )
