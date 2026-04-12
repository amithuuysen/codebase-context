"""Tests for Architecture completion: skip-unchanged files & index reuse."""

import asyncio
import json
import os
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Skip Unchanged Files
# ---------------------------------------------------------------------------
print("--- 1. Skip Unchanged Files ---")
from codecontext.core.merkle import MerkleSynchronizer, MerkleNode
from codecontext.core.context import Context

with tempfile.TemporaryDirectory() as tmpdir:
    # Create a small codebase
    src_dir = os.path.join(tmpdir, "project")
    os.makedirs(os.path.join(src_dir, "src"))
    Path(os.path.join(src_dir, "src", "main.py")).write_text("def main(): pass\n")
    Path(os.path.join(src_dir, "src", "utils.py")).write_text("def helper(): return 42\n")
    Path(os.path.join(src_dir, "src", "config.py")).write_text("DEBUG = True\n")

    # Build Merkle tree and save
    merkle = MerkleSynchronizer(src_dir, ignore_patterns=[])
    asyncio.run(merkle.initialize())
    merkle.save_current_state()

    # Collect saved hashes
    saved_hashes = Context._collect_saved_hashes(merkle._saved_tree)
    assert len(saved_hashes) == 3, f"Expected 3 file hashes, got {len(saved_hashes)}"
    assert "src/main.py" in saved_hashes
    assert "src/utils.py" in saved_hashes
    print(f"Collected {len(saved_hashes)} file hashes from Merkle tree")

    # Verify hashes match actual files
    for rel, h in saved_hashes.items():
        actual = MerkleSynchronizer._hash_file(os.path.join(src_dir, rel))
        assert h == actual, f"Hash mismatch for {rel}"
    print("Merkle hashes match actual files OK")

    # Modify one file
    Path(os.path.join(src_dir, "src", "main.py")).write_text("def main():\n    print('hello')\n")

    # Check which files changed
    files = [
        os.path.join(src_dir, "src", "main.py"),
        os.path.join(src_dir, "src", "utils.py"),
        os.path.join(src_dir, "src", "config.py"),
    ]

    changed = []
    skipped = 0
    for fpath in files:
        rel = os.path.relpath(fpath, src_dir)
        current_hash = MerkleSynchronizer._hash_file(fpath)
        saved_hash = saved_hashes.get(rel)
        if saved_hash and saved_hash == current_hash:
            skipped += 1
        else:
            changed.append(fpath)

    assert len(changed) == 1, f"Expected 1 changed file, got {len(changed)}"
    assert skipped == 2, f"Expected 2 skipped, got {skipped}"
    assert changed[0].endswith("main.py")
    print(f"Skip unchanged: {skipped} skipped, {len(changed)} changed (main.py)")
    print("Skip unchanged files logic OK")


# ---------------------------------------------------------------------------
# 2. SimHash Index Reuse (Architecture §4)
# ---------------------------------------------------------------------------
print()
print("--- 2. SimHash Index Reuse ---")
from codecontext.core.simhash import (
    compute_simhash,
    compute_simhash_from_directory,
    simhash_similarity,
)

with tempfile.TemporaryDirectory() as tmpdir:
    # Create two nearly-identical projects (simulate teammates)
    proj_a = os.path.join(tmpdir, "project_a")
    proj_b = os.path.join(tmpdir, "project_b")
    shared_files = {
        "src/main.py": "def main(): pass\n",
        "src/utils.py": "def helper(): return 42\n",
        "src/config.py": "DEBUG = True\n",
        "src/models.py": "class User:\n    name: str\n    email: str\n",
        "src/routes.py": "def get_users(): return []\ndef create_user(u): pass\n",
        "src/db.py": "import sqlite3\ndef connect(): pass\n",
        "src/auth.py": "def login(user, pw): pass\ndef logout(): pass\n",
        "src/middleware.py": "def cors(req): pass\ndef logging(req): pass\n",
    }
    for proj in (proj_a, proj_b):
        os.makedirs(os.path.join(proj, "src"))
        for rel, content in shared_files.items():
            Path(os.path.join(proj, rel)).write_text(content)
    # project_b differs slightly in one file
    Path(os.path.join(proj_b, "src", "utils.py")).write_text(
        "def helper(): return 42\n\ndef extra(): pass\n"
    )

    hash_a = compute_simhash_from_directory(proj_a)
    hash_b = compute_simhash_from_directory(proj_b)
    sim = simhash_similarity(hash_a, hash_b)
    print(f"SimHash A: {hash_a[:16]}...  B: {hash_b[:16]}...  similarity: {sim:.4f}")
    assert sim > 0.8, f"Expected high similarity, got {sim}"
    print("Near-identical codebases have high SimHash similarity OK")

    # Create a very different project
    proj_c = os.path.join(tmpdir, "project_c")
    os.makedirs(os.path.join(proj_c, "lib"))
    Path(os.path.join(proj_c, "lib", "server.js")).write_text(
        "const express = require('express');\napp.listen(3000);\n"
    )
    Path(os.path.join(proj_c, "lib", "db.js")).write_text(
        "module.exports = { connect() {} };\n"
    )
    hash_c = compute_simhash_from_directory(proj_c)
    sim_ac = simhash_similarity(hash_a, hash_c)
    print(f"SimHash A vs C (unrelated): {sim_ac:.4f}")
    assert sim_ac < sim, "Unrelated project should be less similar"
    print("Unrelated codebases have lower SimHash similarity OK")


# ---------------------------------------------------------------------------
# 3. SimHash Registry (server helpers)
# ---------------------------------------------------------------------------
print()
print("--- 3. SimHash Registry ---")
from codecontext.server.index_server import (
    _load_simhash_registry,
    _save_simhash_registry,
    _find_similar_index,
)
import codecontext.server.index_server as _isrv

with tempfile.TemporaryDirectory() as tmpdir:
    # Patch config for test
    original_cfg = _isrv._cfg

    class _FakeCfg:
        data_dir = tmpdir

    _isrv._cfg = _FakeCfg()

    # Empty registry initially
    _isrv._simhash_registry.clear()
    _save_simhash_registry()
    _isrv._simhash_registry.clear()
    _load_simhash_registry()
    assert len(_isrv._simhash_registry) == 0, "Registry should be empty"
    print("Empty registry load OK")

    # Register entries
    _isrv._simhash_registry["ws1"] = hash_a
    _isrv._simhash_registry["ws2"] = hash_c
    _save_simhash_registry()

    # Simulate indexed status for ws1 and ws2
    _isrv._index_status["ws1"] = {"status": "indexed"}
    _isrv._index_status["ws2"] = {"status": "indexed"}

    # Reload and verify
    _isrv._simhash_registry.clear()
    _load_simhash_registry()
    assert _isrv._simhash_registry["ws1"] == hash_a
    assert _isrv._simhash_registry["ws2"] == hash_c
    print("Registry save/load round-trip OK")

    # Find similar
    match = _find_similar_index(hash_b, exclude_workspace="ws_new")
    assert match is not None, "Should find similar index"
    assert match[0] == "ws1", f"Expected ws1, got {match[0]}"
    assert match[1] > 0.8, f"Expected high similarity, got {match[1]}"
    print(f"Found similar index: {match[0]} (similarity={match[1]:.4f})")

    # Exclude yourself
    match_self = _find_similar_index(hash_a, exclude_workspace="ws1")
    assert match_self is None or match_self[0] != "ws1", "Should not match self"
    print("Self-exclusion OK")

    # Restore config
    _isrv._cfg = original_cfg
    _isrv._simhash_registry.clear()
    _isrv._index_status.pop("ws1", None)
    _isrv._index_status.pop("ws2", None)


print()
print("=" * 60)
print("All Architecture completion tests passed!")
print("=" * 60)
