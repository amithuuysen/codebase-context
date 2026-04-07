"""
Path obfuscation — encrypt file path segments for privacy.

Cursor splits paths by '/' and '.' and encrypts each segment with a
client-side secret key.  This module replicates that approach using
HMAC-SHA256 (no external crypto dependency needed).

The secret key is auto-generated on first use and stored locally at
``<data_dir>/path_obfuscation_key``.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)


class PathObfuscator:
    """Obfuscate file path segments using HMAC-SHA256 with a local secret key."""

    def __init__(self, data_dir: str | Path):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._key_path = self._data_dir / "path_obfuscation_key"
        self._key = self._load_or_create_key()

    def _load_or_create_key(self) -> bytes:
        if self._key_path.exists():
            key = self._key_path.read_bytes()
            if len(key) >= 32:
                return key
        key = secrets.token_bytes(32)
        self._key_path.write_bytes(key)
        # Restrict permissions to owner only
        os.chmod(self._key_path, 0o600)
        logger.info("Generated new path obfuscation key")
        return key

    def obfuscate(self, relative_path: str) -> str:
        """Obfuscate a relative file path, preserving directory structure.

        Each path segment is replaced with its HMAC-SHA256 hash (first 12 hex chars).
        The structure (number of segments) is preserved but content is hidden.

        Example: ``src/auth/session.py`` → ``a3f1b2/c4d5e6/f7a8b9.c0d1e2``
        """
        if not relative_path:
            return ""

        # Split on path separator
        parts = relative_path.replace("\\", "/").split("/")
        obfuscated_parts = []

        for part in parts:
            # Split filename on '.' to obfuscate name and extension separately
            if "." in part and part != parts[-1] is False:
                pass
            # For the last part (filename), split on '.' to preserve extension structure
            if part == parts[-1] and "." in part:
                name_parts = part.split(".")
                obfuscated_name = ".".join(
                    self._hmac_segment(seg) for seg in name_parts
                )
                obfuscated_parts.append(obfuscated_name)
            else:
                obfuscated_parts.append(self._hmac_segment(part))

        return "/".join(obfuscated_parts)

    def _hmac_segment(self, segment: str) -> str:
        """HMAC-SHA256 a single path segment, return first 12 hex chars."""
        mac = hmac.new(self._key, segment.encode(), hashlib.sha256)
        return mac.hexdigest()[:12]
