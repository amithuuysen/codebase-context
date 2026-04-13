"""
Embedding provider factory — returns LlamaIndex BaseEmbedding instances.

Pipeline: Tree-sitter chunks → LlamaIndex TextNode → **embed_model** → FAISS

Supported providers:
  - openai    → llama-index-embeddings-openai
  - ollama    → llama-index-embeddings-ollama
  - local     → llama-index-embeddings-huggingface (sentence-transformers)
  - fastembed → llama-index-embeddings-fastembed (ONNX-optimized, fastest)
  - llamacpp  → llama-index-embeddings-llama-cpp (GGUF models, Metal GPU)
"""

from __future__ import annotations

import logging
import os
import sys
import types

from llama_index.core.embeddings import BaseEmbedding

# ---------------------------------------------------------------------------
# Compatibility shims for transformers >=5  (needed by Jina v2 custom code)
# ---------------------------------------------------------------------------

# 1) transformers.onnx removed — Jina v2 configuration_bert.py imports OnnxConfig
if "transformers.onnx" not in sys.modules:
    _stub = types.ModuleType("transformers.onnx")
    _stub.OnnxConfig = type("OnnxConfig", (), {})  # type: ignore[attr-defined]
    sys.modules["transformers.onnx"] = _stub

# 2) find_pruneable_heads_and_indices removed from pytorch_utils
try:
    from transformers.pytorch_utils import find_pruneable_heads_and_indices as _  # noqa: F401
except ImportError:
    import torch
    from transformers import pytorch_utils as _pu

    def _find_pruneable(heads, n_heads, head_size, already_pruned):  # type: ignore[override]
        mask = torch.ones(n_heads, head_size)
        for h in already_pruned:
            mask[h] = 0
        for h in heads:
            mask[h] = 0
        idx = torch.arange(n_heads * head_size)[mask.view(-1).bool()]
        return heads, idx

    _pu.find_pruneable_heads_and_indices = _find_pruneable  # type: ignore[attr-defined]

# 3) PretrainedConfig no longer sets is_decoder / add_cross_attention / use_cache
#    by default — Jina v2 modeling_bert.py reads them on the config object.
try:
    from transformers import PretrainedConfig as _PC

    _REMOVED_DEFAULTS = {"is_decoder": False, "add_cross_attention": False, "use_cache": True}

    _orig_pc_init = _PC.__init__

    def _compat_pc_init(self, **kwargs):  # type: ignore[override]
        _orig_pc_init(self, **kwargs)
        for _attr, _default in _REMOVED_DEFAULTS.items():
            if not hasattr(self, _attr):
                setattr(self, _attr, kwargs.get(_attr, _default))

    _PC.__init__ = _compat_pc_init  # type: ignore[method-assign]
except Exception:
    pass

# 4) transformers 5 unconditionally wraps model __init__ in torch.device("meta")
#    which breaks custom modeling code that creates real tensors during init
#    (e.g. Jina v2 ALiBi construction).  Remove the meta context.
try:
    import torch as _torch
    from transformers.modeling_utils import PreTrainedModel as _PTM

    _orig_get_init_context = _PTM.get_init_context.__func__  # type: ignore[attr-defined]

    @classmethod  # type: ignore[misc]
    def _safe_get_init_context(cls, *args, **kwargs):
        contexts = _orig_get_init_context(cls, *args, **kwargs)
        return [c for c in contexts if not (isinstance(c, _torch.device) and c.type == "meta")]

    _PTM.get_init_context = _safe_get_init_context  # type: ignore[method-assign]

    # Also restore get_head_mask (removed in transformers 5) — Jina v2 calls it
    # in forward(). When head_mask is None it just returns [None]*num_layers.
    if not hasattr(_PTM, "get_head_mask"):

        def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        _PTM.get_head_mask = _get_head_mask  # type: ignore[attr-defined]
except Exception:
    pass

__all__ = ["create_embedding"]

logger = logging.getLogger(__name__)

# Default models per provider
DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
    "local": "all-MiniLM-L6-v2",
    "fastembed": "BAAI/bge-small-en-v1.5",
    "llamacpp": "nomic-embed-text-v1.5.f16.gguf",
}


def create_embedding(provider: str, **kwargs) -> BaseEmbedding:
    """
    Instantiate the right LlamaIndex embedding model from a provider name.

    Keyword args are forwarded to the underlying LlamaIndex class.  Common
    ones: model / model_name, api_key, base_url, embed_batch_size.
    """
    provider = provider.lower().strip()
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

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
            embed_batch_size=batch_size,
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

        model_name = kwargs.get("model", "all-MiniLM-L6-v2")
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=batch_size,
            trust_remote_code=True,
            # low_cpu_mem_usage=False avoids meta-device init that breaks
            # custom modeling code (e.g. Jina v2 ALiBi tensor construction).
            model_kwargs={"low_cpu_mem_usage": False},
        )
        logger.info("Embedding: HuggingFace model=%s", model_name)
        return embed_model

    elif provider == "fastembed":
        try:
            from llama_index.embeddings.fastembed import FastEmbedEmbedding
        except ImportError:
            raise ImportError(
                "llama-index-embeddings-fastembed is required.  "
                "Install with: pip install llama-index-embeddings-fastembed fastembed"
            )

        model = kwargs.get("model", "BAAI/bge-small-en-v1.5")
        embed_model = FastEmbedEmbedding(
            model_name=model,
            embed_batch_size=batch_size,
        )
        logger.info("Embedding: FastEmbed (ONNX) model=%s", model)
        return embed_model

    elif provider == "llamacpp":
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required.  "
                "Install with: pip install llama-cpp-python"
            )

        model_path = kwargs.get("model", "")
        if not model_path:
            raise ValueError(
                "LLAMACPP_MODEL_PATH env var required. Point it to a .gguf embedding model file. "
                "Download from: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF"
            )

        # Wrap llama-cpp-python in a LlamaIndex-compatible adapter
        llm = Llama(model_path=model_path, embedding=True, n_ctx=2048, verbose=False)

        class LlamaCppEmbeddingAdapter(BaseEmbedding):
            """Thin adapter: llama-cpp-python → LlamaIndex BaseEmbedding."""

            model_name: str = model_path

            def _get_text_embedding(self, text: str) -> list[float]:
                resp = llm.create_embedding(text)
                return resp["data"][0]["embedding"]

            def _get_query_embedding(self, query: str) -> list[float]:
                return self._get_text_embedding(query)

            async def _aget_query_embedding(self, query: str) -> list[float]:
                return self._get_query_embedding(query)

            def _get_text_embedding_batch(
                self, texts: list[str], **kwargs
            ) -> list[list[float]]:
                resp = llm.create_embedding(texts)
                return [d["embedding"] for d in resp["data"]]

        embed_model = LlamaCppEmbeddingAdapter(embed_batch_size=batch_size)
        logger.info("Embedding: llama.cpp model=%s", model_path)
        return embed_model

    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'.  "
            "Choose from: openai, ollama, local, fastembed, llamacpp"
        )
