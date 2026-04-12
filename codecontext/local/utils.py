"""
MCP utility helpers.

Mirrors packages/mcp/src/utils.ts.
"""

from __future__ import annotations

import logging
import os
import signal
import sys

from codecontext.core.types import Config

logger = logging.getLogger(__name__)


def ensure_absolute(p: str) -> str:
    """Resolve a path to absolute, expanding ~."""
    return os.path.abspath(os.path.expanduser(p))


def truncate(text: str, max_len: int = 5000) -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def log_config(cfg: Config) -> None:
    """Log configuration summary (mirrors TS logConfigurationSummary)."""
    logger.info("=" * 60)
    logger.info("Context MCP Server (Python)")
    logger.info("=" * 60)
    logger.info("Embedding provider: %s", cfg.embedding_provider)
    logger.info("Embedding model:    %s", cfg.embedding_model)
    if cfg.embedding_provider == "openai":
        logger.info("OpenAI API key:     %s", "***set***" if cfg.openai_api_key else "NOT SET")
        if cfg.openai_base_url:
            logger.info("OpenAI base URL:    %s", cfg.openai_base_url)
    elif cfg.embedding_provider == "ollama":
        logger.info("Ollama host:        %s", cfg.ollama_host)
    logger.info("Data directory:     %s", cfg.data_dir)
    logger.info("Chunk size:         %d", cfg.chunk_size)
    logger.info("Chunk overlap:      %d", cfg.chunk_overlap)
    logger.info("Chunk limit:        %d", cfg.chunk_limit)
    logger.info("Sync interval:      %d s", cfg.sync_interval_seconds)
    logger.info("=" * 60)


def show_help() -> None:
    """Print CLI help message to stderr."""
    print(
        """
Context MCP Server (Python) — Semantic code search via MCP protocol.

USAGE
  codecontext                     Start the MCP server (streamable-http on :8080)
  MCP_TRANSPORT=stdio codecontext Start using stdio transport
  codecontext --help              Show this message

ENVIRONMENT VARIABLES
  MCP_TRANSPORT        streamable-http | stdio | sse  (default: streamable-http)
  EMBEDDING_PROVIDER   openai | ollama | local   (default: ollama)
  EMBEDDING_MODEL      Model name (default: text-embedding-3-small)
  OPENAI_API_KEY       Required for OpenAI provider
  OPENAI_BASE_URL      Optional custom base URL
  OLLAMA_HOST          Ollama server (default: http://127.0.0.1:11434)
  OLLAMA_MODEL         Ollama model (default: nomic-embed-text)
  CODECONTEXT_DATA_DIR Persistence directory (default: ~/.context)
  CHUNK_SIZE           Max chars per chunk (default: 1500)
  CHUNK_OVERLAP        Overlap between chunks (default: 200)
  CHUNK_LIMIT          Max total chunks (default: 450000)
  SYNC_INTERVAL_SECONDS  Background sync interval (default: 300)
  INDEX_SERVER_URL     Remote index server URL (enables proxy mode)

MCP TOOLS
  index_codebase       Index a directory for semantic search
  search_code          Search an indexed codebase
  clear_index          Remove a codebase index
  get_indexing_status   Get indexing progress / status
""",
        file=sys.stderr,
    )


def install_shutdown_handlers() -> None:
    """Install SIGINT / SIGTERM handlers for graceful shutdown."""
    def _shutdown(sig, _frame):
        logger.info("Received %s, shutting down gracefully...", signal.Signals(sig).name)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
