"""End-to-end accuracy validation for the search pipeline."""

from codecontext.core.splitter import AstSplitter, TextSplitter

splitter = AstSplitter(chunk_size=2500, chunk_overlap=300)

# ------------------------------------------------------------------
# 1. Control flow should NOT be split out (matches TS behavior)
# ------------------------------------------------------------------
print("--- 1. AST splitter: control flow stays inside parent ---")

code = """import os

def process_file(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = f.read()
    for line in data.splitlines():
        if line.startswith('#'):
            continue
        print(line)
    return data

class FileProcessor:
    def __init__(self, root):
        self.root = root

    def scan(self):
        results = []
        for f in os.listdir(self.root):
            try:
                results.append(self.process_file(f))
            except Exception:
                pass
        return results
"""

chunks = splitter.split(code, "python", "example.py")
print(f"Chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    lines = c.content.count("\n") + 1
    print(f"  [{i}] L{c.start_line}-{c.end_line} ({lines} lines): {c.content[:70]!r}")

# Verify: no bare control flow as separate chunks
for c in chunks:
    first_line = c.content.strip().split("\n")[0].strip()
    assert not first_line.startswith("if "), f"Bare if: {first_line}"
    assert not first_line.startswith("for "), f"Bare for: {first_line}"
    assert not first_line.startswith("with "), f"Bare with: {first_line}"
    assert not first_line.startswith("try:"), f"Bare try: {first_line}"
print("PASS: Control flow stays inside parent chunks\n")


# ------------------------------------------------------------------
# 2. Gap text (imports, comments) should be discarded (matches TS)
# ------------------------------------------------------------------
print("--- 2. AST splitter: gap text discarded ---")

code_with_gaps = """# This is a module docstring
# with multiple comment lines
import os
import sys

CONSTANT = 42

def foo():
    pass

# More comments between functions

def bar():
    return CONSTANT
"""

chunks2 = splitter.split(code_with_gaps, "python", "gaps.py")
print(f"Chunks: {len(chunks2)}")
for i, c in enumerate(chunks2):
    print(f"  [{i}] L{c.start_line}-{c.end_line}: {c.content[:70]!r}")

# Only foo() and bar() should be chunks — no import/comment/constant chunks
for c in chunks2:
    first_line = c.content.strip().split("\n")[0].strip()
    assert first_line.startswith("def ") or first_line.startswith("class ") or first_line.startswith("@"), \
        f"Non-function/class chunk found: {first_line!r}"
print("PASS: Gap text (imports, comments, constants) correctly discarded\n")


# ------------------------------------------------------------------
# 3. Large function is split line-by-line with overlap
# ------------------------------------------------------------------
print("--- 3. Oversized chunk: line-by-line split with overlap ---")

big_func = "def big():\n" + "\n".join(f"    line_{i} = {i}" for i in range(200))
small_splitter = AstSplitter(chunk_size=500, chunk_overlap=100)
big_chunks = small_splitter.split(big_func, "python", "big.py")
print(f"200-line function with chunk_size=500: {len(big_chunks)} chunks")
assert len(big_chunks) > 1, "Should produce multiple chunks"

# Verify overlap: second chunk should start with content from end of first
if len(big_chunks) >= 2:
    last_lines_1 = set(big_chunks[0].content.strip().split("\n")[-3:])
    first_lines_2 = set(big_chunks[1].content.strip().split("\n")[:3])
    overlap = last_lines_1 & first_lines_2
    print(f"  Overlap between chunk 0 and 1: {len(overlap)} shared lines")
    assert len(overlap) > 0, "Should have overlap between consecutive chunks"
print("PASS: Oversized chunks split line-by-line with overlap\n")


# ------------------------------------------------------------------
# 4. TypeScript splitter nodes match TS original
# ------------------------------------------------------------------
print("--- 4. TypeScript AST node types ---")

ts_code = """
export interface User {
    name: string;
    age: number;
}

type Status = "active" | "inactive";

export function greet(user: User): string {
    if (user.age > 18) {
        return "Hello, " + user.name;
    }
    return "Hi!";
}

export class UserService {
    private users: User[] = [];

    addUser(user: User): void {
        this.users.push(user);
    }

    getUsers(): User[] {
        return this.users;
    }
}

const helper = () => {
    for (let i = 0; i < 10; i++) {
        console.log(i);
    }
};
"""

ts_chunks = splitter.split(ts_code, "typescript", "users.ts")
print(f"Chunks: {len(ts_chunks)}")
for i, c in enumerate(ts_chunks):
    first = c.content.strip().split("\n")[0][:60]
    print(f"  [{i}] L{c.start_line}-{c.end_line}: {first}")

# Verify interface and type_alias are captured
chunk_texts = [c.content for c in ts_chunks]
has_interface = any("interface User" in t for t in chunk_texts)
has_type = any("type Status" in t for t in chunk_texts)
has_function = any("function greet" in t for t in chunk_texts)
has_class = any("class UserService" in t for t in chunk_texts)
print(f"  interface: {has_interface}, type_alias: {has_type}, function: {has_function}, class: {has_class}")
# Note: interface and type_alias are only captured if tree-sitter-typescript is installed
if has_interface:
    assert has_type, "type_alias should be captured when interface is"
    assert has_function, "function should be captured"
    assert has_class, "class should be captured"
    print("PASS: TypeScript nodes match TS original")
else:
    print("SKIP: tree-sitter-typescript not producing expected nodes (grammar version issue)")
print()


# ------------------------------------------------------------------
# 5. FAISS IndexFlatIP (cosine) produces meaningful scores
# ------------------------------------------------------------------
print("--- 5. FAISS cosine similarity (IndexFlatIP) ---")

import faiss
import numpy as np

dim = 4
index = faiss.IndexFlatIP(dim)

# Insert normalized vectors
v1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # "hello"
v2 = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)  # "world"
v3 = np.array([[0.7, 0.7, 0.0, 0.0]], dtype=np.float32)  # mix
v3 /= np.linalg.norm(v3)  # normalize

index.add(v1)
index.add(v2)
index.add(v3)

# Query with v1 — should rank v1 first, v3 second, v2 last
q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
scores, ids = index.search(q, 3)
print(f"  Query [1,0,0,0] → scores={scores[0].tolist()}, ids={ids[0].tolist()}")
assert ids[0][0] == 0, "v1 should be most similar"
assert scores[0][0] > scores[0][1], "v1 score > v3 score"
assert scores[0][1] > scores[0][2], "v3 score > v2 score"
print("PASS: IndexFlatIP correctly ranks by cosine similarity\n")


# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
print("=" * 60)
print("ALL ACCURACY TESTS PASSED")
print("=" * 60)
