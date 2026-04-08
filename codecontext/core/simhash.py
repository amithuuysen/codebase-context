"""
SimHash — locality-sensitive hashing for codebase similarity.

Used for team index sharing: when two developers' codebases have similar
SimHash values, their indexes overlap significantly (~92% for same-branch
teammates per Cursor's research).

This enables:
  1. Developer joins team → compute SimHash of their codebase
  2. Find the most similar existing index on the server
  3. Copy that index as a starting point → diff only divergent files
  4. Time-to-first-query drops from minutes to seconds
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _hash_to_bits(data: bytes, n_bits: int = 128) -> list[int]:
    """Convert SHA-256 hash to a list of +1/-1 bits."""
    h = hashlib.sha256(data).digest()
    bits = []
    for byte in h:
        for i in range(8):
            bits.append(1 if (byte >> (7 - i)) & 1 else -1)
            if len(bits) >= n_bits:
                return bits
    return bits


def compute_simhash(file_hashes: dict[str, str], n_bits: int = 128) -> str:
    """Compute a SimHash fingerprint from a dict of {relative_path: content_hash}.

    Each file contributes a weighted vote to each bit position.
    The final fingerprint is the majority vote across all files.

    Args:
        file_hashes: mapping of relative paths to their SHA-256 content hashes.
        n_bits:      number of bits in the fingerprint (default 128).

    Returns:
        Hex string of the SimHash fingerprint.
    """
    if not file_hashes:
        return "0" * (n_bits // 4)

    # Accumulator for each bit position
    v = [0] * n_bits

    for rel_path, content_hash in file_hashes.items():
        # Combine path + content hash so file renames are detected
        combined = f"{rel_path}:{content_hash}".encode()
        bits = _hash_to_bits(combined, n_bits)
        for i, bit in enumerate(bits):
            v[i] += bit

    # Majority vote: positive → 1, else → 0
    result_bits = [1 if x > 0 else 0 for x in v]

    # Convert to hex
    result = 0
    for bit in result_bits:
        result = (result << 1) | bit
    return format(result, f"0{n_bits // 4}x")


def simhash_distance(hash_a: str, hash_b: str) -> int:
    """Compute Hamming distance between two SimHash fingerprints.

    Lower distance = more similar codebases.
    Distance 0 = identical file sets.
    """
    a = int(hash_a, 16)
    b = int(hash_b, 16)
    xor = a ^ b
    return bin(xor).count("1")


def simhash_similarity(hash_a: str, hash_b: str, n_bits: int = 128) -> float:
    """Compute similarity ratio (0.0–1.0) between two SimHash fingerprints.

    Returns 1.0 for identical, 0.0 for completely different.
    """
    dist = simhash_distance(hash_a, hash_b)
    return 1.0 - (dist / n_bits)


def compute_simhash_from_directory(
    root_dir: str,
    supported_extensions: set[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> str:
    """Convenience: walk a directory and compute its SimHash.

    Args:
        root_dir:             absolute path to codebase root.
        supported_extensions: only hash files with these extensions.
        ignore_patterns:      glob patterns to skip.

    Returns:
        Hex SimHash fingerprint.
    """
    from fnmatch import fnmatch

    file_hashes: dict[str, str] = {}
    root = Path(root_dir)
    ignore = ignore_patterns or []
    exts = supported_extensions

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden directories
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        rel_dir = os.path.relpath(dirpath, root)
        # Skip ignored directories
        skip_dir = False
        for pat in ignore:
            if fnmatch(rel_dir, pat) or fnmatch(os.path.basename(dirpath), pat):
                skip_dir = True
                break
        if skip_dir:
            dirnames.clear()
            continue

        for fname in filenames:
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1]
            if exts and ext not in exts:
                continue
            rel_path = os.path.relpath(os.path.join(dirpath, fname), root)

            # Skip ignored files
            skip_file = False
            for pat in ignore:
                if fnmatch(rel_path, pat) or fnmatch(fname, pat):
                    skip_file = True
                    break
            if skip_file:
                continue

            fpath = os.path.join(dirpath, fname)
            try:
                content = Path(fpath).read_bytes()
                file_hashes[rel_path] = hashlib.sha256(content).hexdigest()
            except (OSError, PermissionError):
                continue

    return compute_simhash(file_hashes)
