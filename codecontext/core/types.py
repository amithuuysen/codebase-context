"""Configuration, constants, and type definitions.

Mirrors packages/core/src/types.ts + constant definitions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SUPPORTED_EXTENSIONS: set[str] = {
    ".ts", ".tsx", ".js", ".jsx", ".py", ".java",
    ".cpp", ".c", ".h", ".hpp", ".cs", ".go", ".rs",
    ".php", ".rb", ".swift", ".kt", ".scala", ".m", ".mm",
    ".md", ".markdown", ".ipynb",
}

DEFAULT_IGNORE_PATTERNS: list[str] = [
    "node_modules/**", "dist/**", "build/**", "out/**", "target/**",
    "coverage/**", ".nyc_output/**", ".vscode/**", ".idea/**",
    "*.swp", "*.swo", "__pycache__/**", ".pytest_cache/**",
    "logs/**", "tmp/**", "temp/**", "*.log",
    ".env", ".env.*", "*.local",
    "*.min.js", "*.min.css", "*.min.map",
    "*.bundle.js", "*.bundle.css", "*.chunk.js",
    "*.vendor.js", "*.polyfills.js", "*.runtime.js", "*.map",
    "node_modules", ".git", ".svn", ".hg", "build", "dist", "out",
    "target", ".vscode", ".idea", "__pycache__", ".pytest_cache",
    "coverage", ".nyc_output", "logs", "tmp", "temp",
]

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript",
    ".py": "python", ".java": "java",
    ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust",
    ".php": "php", ".rb": "ruby", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala",
    ".m": "objective-c", ".mm": "objective-c",
    ".ipynb": "jupyter", ".md": "markdown", ".markdown": "markdown",
}

CONTEXT_DIR = Path.home() / ".context"
SNAPSHOT_FILE = CONTEXT_DIR / "mcp-codebase-snapshot.json"
MERKLE_DIR = CONTEXT_DIR / "merkle"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A chunk of source code produced by the tree-sitter splitter."""
    content: str
    start_line: int
    end_line: int
    language: str = "text"
    file_path: str = ""


@dataclass
class SemanticSearchResult:
    """A single search result returned to the MCP caller."""
    content: str
    relative_path: str
    start_line: int
    end_line: int
    language: str
    score: float


# ---------------------------------------------------------------------------
# Runtime config — reads from env vars
# ---------------------------------------------------------------------------

@dataclass
class Config:
    embedding_provider: str = ""
    embedding_model: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "nomic-embed-text"
    local_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 3000
    chunk_overlap: int = 300
    embedding_batch_size: int = 100
    chunk_limit: int = 450_000
    sync_interval_seconds: int = 300
    data_dir: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls(
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai").lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
            ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "nomic-embed-text"),
            local_model=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            chunk_limit=int(os.getenv("CHUNK_LIMIT", "450000")),
            sync_interval_seconds=int(os.getenv("SYNC_INTERVAL_SECONDS", "300")),
            data_dir=os.getenv("CODECONTEXT_DATA_DIR", str(CONTEXT_DIR)),
        )
        if not cfg.embedding_model:
            if cfg.embedding_provider == "openai":
                cfg.embedding_model = "text-embedding-3-small"
            elif cfg.embedding_provider == "ollama":
                cfg.embedding_model = cfg.ollama_model
            elif cfg.embedding_provider == "local":
                cfg.embedding_model = cfg.local_model
        return cfg
