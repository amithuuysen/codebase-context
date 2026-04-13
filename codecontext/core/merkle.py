"""
Merkle tree file synchronizer — directory-aware change detection.

Inspired by Cursor's indexing architecture:
  "Cursor builds its first view of a codebase using a Merkle tree, which lets
   it detect exactly which files and directories have changed without
   reprocessing everything."

Instead of flat file-hash comparison, the Merkle tree hashes each directory
from its children's hashes.  On sync, only the branches where hashes diverge
are walked — unchanged subtrees are skipped entirely.

For a 50K-file workspace, this reduces sync IO from scanning every file to
walking only divergent branches.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from .types import MERKLE_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Merkle node
# ---------------------------------------------------------------------------

@dataclass
class MerkleNode:
    """A node in the Merkle tree — either a file or a directory."""
    path: str          # relative path from root
    hash: str          # SHA-256 of content (file) or of children hashes (dir)
    is_dir: bool
    children: dict[str, "MerkleNode"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MerkleSynchronizer
# ---------------------------------------------------------------------------

class MerkleSynchronizer:
    """
    Directory-aware file change detection using Merkle trees.

    Replaces flat file-hash scanning:
      - build_tree() → construct full Merkle tree
      - diff(old, new) → added, modified, removed files
      - Only divergent branches are walked during diff

    Persistence: serialized to ~/.context/merkle/<hash>.json
    """

    def __init__(self, root_dir: str, ignore_patterns: list[str] | None = None, merkle_dir: Path | None = None):
        self.root_dir = os.path.abspath(root_dir)
        self.ignore_patterns = ignore_patterns or []
        self._saved_tree: MerkleNode | None = None
        self._merkle_dir = merkle_dir or MERKLE_DIR
        self._snapshot_path = self._get_snapshot_path(self.root_dir, self._merkle_dir)

    async def initialize(self) -> None:
        """Load the previously saved Merkle tree from disk."""
        self._saved_tree = self._load_snapshot()

    def build_tree(self) -> MerkleNode:
        """Build a fresh Merkle tree of the codebase.

        Uses a thread pool to parallelize file hashing (I/O-bound SHA-256)
        for codebases with 100+ files.  Small repos use the simpler serial
        path to avoid ThreadPool overhead.
        """
        # Phase 1: Collect all file paths (fast, single-threaded walk)
        file_entries: list[tuple[str, str]] = []  # (full_path, rel_path)
        dir_structure: dict[str, list[str]] = {}

        self._walk_tree(self.root_dir, "", file_entries, dir_structure)

        # Small repos: serial hashing is faster (no thread pool overhead)
        if len(file_entries) < 100:
            return self._build_node(self.root_dir, "")

        # Phase 2: Hash all files in parallel via thread pool
        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(os.cpu_count() or 4, 16)
        file_hashes: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            full_paths = [fp for fp, _ in file_entries]
            hashes = list(pool.map(self._hash_file, full_paths))
            for (_, rel_path), h in zip(file_entries, hashes):
                file_hashes[rel_path] = h

        # Phase 3: Build tree bottom-up using pre-computed hashes
        return self._build_node_cached(self.root_dir, "", file_hashes)

    def _walk_tree(
        self,
        full_path: str,
        rel_path: str,
        file_entries: list[tuple[str, str]],
        dir_structure: dict[str, list[str]],
    ) -> None:
        """Walk the directory tree, collecting file paths and directory structure."""
        try:
            entries = sorted(os.listdir(full_path))
        except PermissionError:
            return

        children = []
        for entry in entries:
            child_full = os.path.join(full_path, entry)
            child_rel = os.path.join(rel_path, entry) if rel_path else entry

            if self._should_ignore(child_full, os.path.isdir(child_full)):
                continue

            children.append(entry)
            if os.path.isfile(child_full):
                file_entries.append((child_full, child_rel))
            elif os.path.isdir(child_full):
                self._walk_tree(child_full, child_rel, file_entries, dir_structure)

        dir_structure[rel_path] = children

    def _build_node_cached(
        self,
        full_path: str,
        rel_path: str,
        file_hashes: dict[str, str],
    ) -> MerkleNode:
        """Build Merkle node using pre-computed file hashes (no I/O)."""
        if os.path.isfile(full_path):
            return MerkleNode(
                path=rel_path,
                hash=file_hashes.get(rel_path, ""),
                is_dir=False,
            )

        children: dict[str, MerkleNode] = {}
        try:
            entries = sorted(os.listdir(full_path))
        except PermissionError:
            return MerkleNode(path=rel_path, hash="", is_dir=True)

        for entry in entries:
            child_full = os.path.join(full_path, entry)
            child_rel = os.path.join(rel_path, entry) if rel_path else entry

            if self._should_ignore(child_full, os.path.isdir(child_full)):
                continue

            child_node = self._build_node_cached(child_full, child_rel, file_hashes)
            children[entry] = child_node

        hasher = hashlib.sha256()
        for name in sorted(children.keys()):
            hasher.update(name.encode())
            hasher.update(children[name].hash.encode())
        dir_hash = hasher.hexdigest()

        return MerkleNode(path=rel_path, hash=dir_hash, is_dir=True, children=children)

    async def check_for_changes(self) -> dict[str, list[str]]:
        """
        Compare the saved tree with the current state.

        Returns {"added": [...], "modified": [...], "removed": [...]}.
        All paths are relative to root_dir.
        """
        current_tree = self.build_tree()

        added: list[str] = []
        modified: list[str] = []
        removed: list[str] = []

        self._diff_trees(self._saved_tree, current_tree, added, modified, removed)

        # Update saved tree to current
        self._saved_tree = current_tree
        self._save_snapshot(current_tree)

        return {"added": added, "modified": modified, "removed": removed}

    def save_current_state(self) -> None:
        """Force-save the current Merkle tree as the baseline."""
        self._saved_tree = self.build_tree()
        self._save_snapshot(self._saved_tree)

    def get_root_hash(self) -> str:
        """Return the root hash (useful for simhash comparison)."""
        if self._saved_tree:
            return self._saved_tree.hash
        return ""

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_node(self, full_path: str, rel_path: str) -> MerkleNode:
        """Recursively build a Merkle node for a file or directory."""
        if os.path.isfile(full_path):
            file_hash = self._hash_file(full_path)
            return MerkleNode(path=rel_path, hash=file_hash, is_dir=False)

        # Directory: build children, then hash from their hashes
        children: dict[str, MerkleNode] = {}
        try:
            entries = sorted(os.listdir(full_path))
        except PermissionError:
            return MerkleNode(path=rel_path, hash="", is_dir=True)

        for entry in entries:
            child_full = os.path.join(full_path, entry)
            child_rel = os.path.join(rel_path, entry) if rel_path else entry

            if self._should_ignore(child_full, os.path.isdir(child_full)):
                continue

            child_node = self._build_node(child_full, child_rel)
            children[entry] = child_node

        # Directory hash = SHA-256 of sorted children hashes
        hasher = hashlib.sha256()
        for name in sorted(children.keys()):
            hasher.update(name.encode())
            hasher.update(children[name].hash.encode())
        dir_hash = hasher.hexdigest()

        node = MerkleNode(path=rel_path, hash=dir_hash, is_dir=True, children=children)
        return node

    # ------------------------------------------------------------------
    # Tree diff  (the key optimization — skip identical subtrees)
    # ------------------------------------------------------------------

    def _diff_trees(
        self,
        old: MerkleNode | None,
        new: MerkleNode | None,
        added: list[str],
        modified: list[str],
        removed: list[str],
    ) -> None:
        """
        Recursively diff two Merkle trees.

        Key optimization: if old.hash == new.hash, the entire subtree is
        unchanged — skip it.  This is what makes Merkle trees O(changes)
        instead of O(total_files).
        """
        if old is None and new is None:
            return

        # Entirely new subtree
        if old is None and new is not None:
            self._collect_files(new, added)
            return

        # Entirely removed subtree
        if old is not None and new is None:
            self._collect_files(old, removed)
            return

        assert old is not None and new is not None

        # FAST PATH: hashes match → entire subtree is unchanged
        if old.hash == new.hash:
            return

        # Both are files but content changed
        if not old.is_dir and not new.is_dir:
            modified.append(new.path)
            return

        # Type changed (file ↔ dir): treat as remove + add
        if old.is_dir != new.is_dir:
            self._collect_files(old, removed)
            self._collect_files(new, added)
            return

        # Both are directories: recurse into children
        old_keys = set(old.children.keys())
        new_keys = set(new.children.keys())

        for name in old_keys - new_keys:
            self._collect_files(old.children[name], removed)

        for name in new_keys - old_keys:
            self._collect_files(new.children[name], added)

        for name in old_keys & new_keys:
            self._diff_trees(
                old.children[name], new.children[name],
                added, modified, removed,
            )

    def _collect_files(self, node: MerkleNode, output: list[str]) -> None:
        """Collect all file paths under a node."""
        if not node.is_dir:
            output.append(node.path)
            return
        for child in node.children.values():
            self._collect_files(child, output)

    # ------------------------------------------------------------------
    # File hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(path: str) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
        except (OSError, PermissionError):
            return ""
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Ignore patterns
    # ------------------------------------------------------------------

    def _should_ignore(self, full_path: str, is_dir: bool) -> bool:
        rel = os.path.relpath(full_path, self.root_dir)
        name = os.path.basename(full_path)

        if name.startswith("."):
            return True

        # Use compiled regex for fast matching (built lazily once)
        compiled = self._get_compiled_ignore()
        if compiled is not None and (compiled.search(rel) or compiled.search(name)):
            return True

        for pattern in self.ignore_patterns:
            if is_dir and pattern == name:
                return True
            if pattern.endswith("/**") and rel.startswith(pattern[:-3]):
                return True
        return False

    def _get_compiled_ignore(self):
        """Lazily compile ignore patterns into a single regex."""
        if not hasattr(self, "_compiled_ignore_re"):
            import re
            from fnmatch import translate
            parts = []
            for pat in self.ignore_patterns:
                if not pat.endswith("/**"):
                    parts.append(translate(pat))
            if parts:
                self._compiled_ignore_re = re.compile("|".join(parts))
            else:
                self._compiled_ignore_re = None
        return self._compiled_ignore_re

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _get_snapshot_path(root_dir: str, merkle_dir: Path | None = None) -> str:
        d = merkle_dir or MERKLE_DIR
        h = hashlib.md5(os.path.abspath(root_dir).encode()).hexdigest()
        d.mkdir(parents=True, exist_ok=True)
        return str(d / f"merkle_{h}.json")

    def _save_snapshot(self, tree: MerkleNode) -> None:
        try:
            data = self._serialize_node(tree)
            self._merkle_dir.mkdir(parents=True, exist_ok=True)
            with open(self._snapshot_path, "w") as f:
                json.dump(data, f)
        except Exception as exc:
            logger.warning("Failed to save Merkle snapshot: %s", exc)

    def _load_snapshot(self) -> MerkleNode | None:
        if not os.path.exists(self._snapshot_path):
            return None
        try:
            with open(self._snapshot_path) as f:
                data = json.load(f)
            return self._deserialize_node(data)
        except Exception as exc:
            logger.warning("Failed to load Merkle snapshot: %s", exc)
            return None

    @staticmethod
    async def delete_snapshot(root_dir: str, merkle_dir: Path | None = None) -> None:
        sp = MerkleSynchronizer._get_snapshot_path(root_dir, merkle_dir)
        if os.path.exists(sp):
            os.remove(sp)

    def _serialize_node(self, node: MerkleNode) -> dict:
        d: dict[str, Any] = {
            "p": node.path,
            "h": node.hash,
            "d": node.is_dir,
        }
        if node.is_dir and node.children:
            d["c"] = {
                name: self._serialize_node(child)
                for name, child in node.children.items()
            }
        return d

    def _deserialize_node(self, data: dict) -> MerkleNode:
        children: dict[str, MerkleNode] = {}
        if data.get("d") and "c" in data:
            for name, child_data in data["c"].items():
                children[name] = self._deserialize_node(child_data)
        return MerkleNode(
            path=data["p"],
            hash=data["h"],
            is_dir=data["d"],
            children=children,
        )
