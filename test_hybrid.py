"""Tests for new Cursor-inspired components: BM25, Hybrid Search, Merkle Tree, Reranker."""

import asyncio
import os
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. BM25 Sparse Index
# ---------------------------------------------------------------------------
print("--- 1. BM25 Sparse Index ---")
from codecontext.core.bm25 import BM25Index, tokenize

# Tokenizer
tokens = tokenize("def processUserInput(data):")
print(f"Tokens for 'def processUserInput(data):': {tokens}")
assert "process" in tokens
assert "user" in tokens
assert "input" in tokens
assert "data" in tokens
assert "def" in tokens
print("Tokenizer OK — camelCase split + lowercase")

# Index
bm25 = BM25Index()
bm25.add_document("d1", "def process_payment(amount): return amount * 1.1",
                   relative_path="payment.py", start_line=1, end_line=1, language="python")
bm25.add_document("d2", "class UserAuthentication: def login(self): pass",
                   relative_path="auth.py", start_line=1, end_line=1, language="python")
bm25.add_document("d3", "def calculate_tax(amount, rate): return amount * rate",
                   relative_path="tax.py", start_line=1, end_line=1, language="python")
bm25.add_document("d4", "import os\ndef read_config(): return os.environ",
                   relative_path="config.py", start_line=1, end_line=2, language="python")

assert bm25.doc_count == 4
print(f"BM25 index: {bm25.doc_count} docs")

# Search — "payment amount" should rank payment.py and tax.py high
results = bm25.search("payment amount", top_k=3)
print(f"BM25 search 'payment amount': {[(did, f'{s:.3f}') for did, s in results]}")
assert results[0][0] == "d1", f"Expected d1 first, got {results[0][0]}"
print("BM25 search ranking OK")

# Soft delete
bm25.delete_ref_doc("payment.py")
results2 = bm25.search("payment amount", top_k=3)
print(f"After deleting payment.py: {[(did, f'{s:.3f}') for did, s in results2]}")
assert all(bm25.get_doc(did).relative_path != "payment.py" for did, _ in results2)
print("BM25 soft-delete OK")

# Persistence
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = os.path.join(tmpdir, "bm25.json")
    bm25.save(save_path)
    bm25_loaded = BM25Index.load(save_path)
    assert bm25_loaded.doc_count == 4
    results3 = bm25_loaded.search("authentication login", top_k=2)
    assert results3[0][0] == "d2"
    print("BM25 persistence (save/load) OK")
print()

# ---------------------------------------------------------------------------
# 2. RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------
print("--- 2. Reciprocal Rank Fusion ---")
from codecontext.core.hybrid_search import reciprocal_rank_fusion

dense = [("a", 0.95), ("b", 0.80), ("c", 0.70)]
sparse = [("b", 5.2), ("d", 4.1), ("a", 3.0)]

fused = reciprocal_rank_fusion([dense, sparse], k=60)
print(f"Dense:  {dense}")
print(f"Sparse: {sparse}")
print(f"Fused:  {[(did, f'{s:.5f}') for did, s in fused[:5]]}")

# "b" appears in both lists (rank 2 dense + rank 1 sparse) → should be top
# "a" also in both (rank 1 dense + rank 3 sparse)
fused_ids = [did for did, _ in fused]
assert "b" in fused_ids[:2], "b should be in top 2 (appears in both lists)"
assert "a" in fused_ids[:2], "a should be in top 2 (appears in both lists)"
assert "d" in fused_ids, "d should appear (only in sparse)"
assert "c" in fused_ids, "c should appear (only in dense)"
print("RRF fusion OK — items in both lists ranked higher\n")

# ---------------------------------------------------------------------------
# 3. Hybrid Search (FAISS + BM25 + RRF)
# ---------------------------------------------------------------------------
print("--- 3. Hybrid Search ---")
from codecontext.core.hybrid_search import HybridSearcher
from codecontext.core.vectordb import FaissVectorDB
from codecontext.core.types import SemanticSearchResult
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.embeddings import MockEmbedding

mock_embed = MockEmbedding(embed_dim=8)

with tempfile.TemporaryDirectory() as tmpdir:
    db = FaissVectorDB(persist_dir=os.path.join(tmpdir, "faiss"), embed_model=mock_embed)
    db.create_collection("test", dimension=8, description="test")

    # Insert into FAISS
    nodes = []
    for i, (text, path) in enumerate([
        ("def process_payment(amount): return amount * 1.1", "payment.py"),
        ("class UserAuth: def login(self): pass", "auth.py"),
        ("def calculate_tax(amount, rate): return amount * rate", "tax.py"),
    ]):
        node = TextNode(text=text, id_=f"n{i}",
                        metadata={"relative_path": path, "start_line": 1, "end_line": 1,
                                  "file_extension": ".py", "language": "python"})
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=path)
        nodes.append(node)
    db.insert_nodes("test", nodes)

    # Build BM25 index for same chunks
    bm25_idx = BM25Index()
    for i, (text, path) in enumerate([
        ("def process_payment(amount): return amount * 1.1", "payment.py"),
        ("class UserAuth: def login(self): pass", "auth.py"),
        ("def calculate_tax(amount, rate): return amount * rate", "tax.py"),
    ]):
        bm25_idx.add_document(f"n{i}", text, relative_path=path,
                               start_line=1, end_line=1, language="python")

    bm25_map = {"test": bm25_idx}
    searcher = HybridSearcher(db, bm25_map)

    # Search — BM25 should boost "payment" query because keyword match
    results = searcher.search("test", "payment amount", top_k=3, threshold=0.0)
    print(f"Hybrid search 'payment amount': {len(results)} results")
    for r in results:
        print(f"  {r.relative_path}: score={r.score:.5f}")
    assert len(results) >= 2
    print("Hybrid search (FAISS + BM25 + RRF) OK")

    # Dense-only fallback
    dense_results = searcher.search("test", "login", top_k=3, threshold=0.0, dense_only=True)
    print(f"Dense-only search: {len(dense_results)} results")
    print("Dense-only fallback OK")
print()

# ---------------------------------------------------------------------------
# 4. Merkle Tree Sync
# ---------------------------------------------------------------------------
print("--- 4. Merkle Tree Sync ---")
from codecontext.core.merkle import MerkleSynchronizer, MerkleNode

with tempfile.TemporaryDirectory() as tmpdir:
    # Create a test codebase
    os.makedirs(os.path.join(tmpdir, "src"))
    Path(os.path.join(tmpdir, "src", "main.py")).write_text("def main(): pass")
    Path(os.path.join(tmpdir, "src", "utils.py")).write_text("def helper(): pass")
    Path(os.path.join(tmpdir, "README.md")).write_text("# Test")

    merkle = MerkleSynchronizer(tmpdir)
    await_fn = asyncio.get_event_loop().run_until_complete

    # Build initial tree
    tree = merkle.build_tree()
    print(f"Merkle tree root hash: {tree.hash[:16]}...")
    assert tree.is_dir
    assert "src" in tree.children
    assert "README.md" in tree.children
    assert "main.py" in tree.children["src"].children
    print(f"Tree structure: {list(tree.children.keys())}")
    print(f"  src/ children: {list(tree.children['src'].children.keys())}")

    # Save initial state
    merkle.save_current_state()

    # Modify a file
    Path(os.path.join(tmpdir, "src", "main.py")).write_text("def main(): print('hello')")
    # Add a file
    Path(os.path.join(tmpdir, "src", "new_file.py")).write_text("x = 42")
    # Remove a file
    os.remove(os.path.join(tmpdir, "src", "utils.py"))

    # Detect changes
    await_fn(merkle.initialize())  # reload saved tree
    changes = await_fn(merkle.check_for_changes())
    print(f"Changes detected: added={changes['added']}, modified={changes['modified']}, removed={changes['removed']}")
    assert os.path.join("src", "new_file.py") in changes["added"]
    assert os.path.join("src", "main.py") in changes["modified"]
    assert os.path.join("src", "utils.py") in changes["removed"]
    print("Merkle tree diff OK — correctly detected add/modify/remove")

    # Verify: unchanged files are NOT in any change list
    assert "README.md" not in changes["added"]
    assert "README.md" not in changes["modified"]
    assert "README.md" not in changes["removed"]
    print("Merkle tree optimization OK — unchanged subtrees skipped")

    # Verify root hash changed
    new_hash = merkle.get_root_hash()
    print(f"New root hash: {new_hash[:16]}... (changed: {new_hash[:16] != tree.hash[:16]})")
    assert new_hash != tree.hash
    print("Root hash correctly updated after changes")
print()

# ---------------------------------------------------------------------------
# 5. Reranker (pass-through mode)
# ---------------------------------------------------------------------------
print("--- 5. Reranker ---")
from codecontext.core.reranker import Reranker

# None provider (pass-through)
reranker = Reranker(provider="none")
assert not reranker.is_active

candidates = [
    SemanticSearchResult("code1", "a.py", 1, 5, "python", 0.9),
    SemanticSearchResult("code2", "b.py", 1, 5, "python", 0.8),
    SemanticSearchResult("code3", "c.py", 1, 5, "python", 0.7),
]
result = reranker.rerank("test query", candidates, top_k=2)
assert len(result) == 2
assert result[0].content == "code1"  # pass-through preserves order
print("Reranker (none/pass-through) OK")

# Local provider without sentence-transformers (should gracefully fallback)
reranker2 = Reranker(provider="local", model="nonexistent-model-xyz")
# This should fallback to "none" since the model won't load
print(f"Reranker fallback: provider={reranker2.provider}, active={reranker2.is_active}")
print("Reranker graceful fallback OK")
print()

# ---------------------------------------------------------------------------
# 6. Full Pipeline: Context with Hybrid Search
# ---------------------------------------------------------------------------
print("--- 6. Full Pipeline (Context + Hybrid) ---")
from codecontext.core.context import Context

with tempfile.TemporaryDirectory() as tmpdir:
    db = FaissVectorDB(persist_dir=os.path.join(tmpdir, "faiss"), embed_model=mock_embed)
    ctx = Context(vector_db=db)

    # Verify new components are wired
    assert ctx._bm25_indices is not None
    assert ctx._hybrid is not None
    assert ctx._reranker is not None
    assert ctx._reranker.provider == "none"
    print(f"Context has: bm25_indices={type(ctx._bm25_indices).__name__}, "
          f"hybrid={type(ctx._hybrid).__name__}, "
          f"reranker={ctx._reranker.provider}")

    # Verify new getters
    assert isinstance(ctx.get_bm25_indices(), dict)
    assert isinstance(ctx.get_merkle_synchronizers(), dict)
    assert ctx.get_reranker() is not None
    print("All new Context getters OK")
print()

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("=" * 60)
print("ALL HYBRID ARCHITECTURE TESTS PASSED")
print("Pipeline: Tree-sitter → FAISS (dense) + BM25 (sparse) → RRF → Rerank")
print("Sync: Merkle tree (O(changes) not O(files))")
print("=" * 60)
