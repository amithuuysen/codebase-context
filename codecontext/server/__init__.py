"""CodeContext index server — centralized HTTP API for team index sharing."""

from .index_server import create_app, run_server

__all__ = ["create_app", "run_server"]
