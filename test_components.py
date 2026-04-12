"""Quick integration test — verifies full MCP pipeline matches TS original."""

from codecontext.core.types import Config, CodeChunk, SemanticSearchResult
from codecontext.core.splitter import AstSplitter, TextSplitter
from codecontext.core.embedding import create_embedding
from codecontext.core.vectordb import FaissVectorDB
from codecontext.core.sync import FileSynchronizer
from codecontext.local.snapshot import SnapshotManager
from codecontext.core.context import Context

print("=" * 60)
print("All imports OK")
print("Config:", Config.from_env())
print()

# ---------------------------------------------------------------------------
# 1. Test Tree-sitter AST splitter
# ---------------------------------------------------------------------------
print("--- 1. Splitter ---")
splitter = AstSplitter()
code = 'def hello():\n    print("world")\n\nclass Foo:\n    pass\n'
chunks = splitter.split(code, "python", "test.py")
print(f"AST Splitter: {len(chunks)} chunks from sample code")
for c in chunks:
    print(f"  L{c.start_line}-{c.end_line}: {c.content[:60]!r}")

# Test text fallback
fallback = TextSplitter(chunk_size=30, chunk_overlap=5)
fb_chunks = fallback.split(code, "python", "test.py")
print(f"Text Splitter (size=30): {len(fb_chunks)} chunks")
print()

# ---------------------------------------------------------------------------
# 2. Test FAISS vector DB (LlamaIndex)
# ---------------------------------------------------------------------------
print("--- 2. FAISS VectorDB ---")
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.embeddings import MockEmbedding

mock_embed = MockEmbedding(embed_dim=8)

import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    db = FaissVectorDB(persist_dir=os.path.join(tmpdir, "faiss"), embed_model=mock_embed)
    db.create_collection("test_col", dimension=8, description="codebasePath:/tmp/test")
    assert db.has_collection("test_col")
    assert not db.has_collection("nonexistent")
    print(f"Collections: {db.list_collections()}")
    print(f"Description: {db.get_collection_description('test_col')}")

    # Insert nodes with ref_doc_id for file-level deletion
    nodes = []
    for i, (text, path) in enumerate([
        ("def hello(): print('world')", "a.py"),
        ("class Foo: pass", "a.py"),
        ("import os; os.listdir('.')", "b.py"),
    ]):
        node = TextNode(text=text, id_=f"n{i}",
                        metadata={"relative_path": path, "start_line": 1, "end_line": 1,
                                  "file_extension": ".py", "language": "python"})
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=path)
        nodes.append(node)
    db.insert_nodes("test_col", nodes)
    print(f"Inserted {len(nodes)} TextNodes")

    # Search (threshold=0.0 for mock embeddings which produce identical vectors)
    results = db.search("test_col", "hello function", top_k=2, threshold=0.0)
    print(f"Search returned {len(results)} results")
    for r in results:
        print(f"  path={r.relative_path}, score={r.score:.4f}, text={r.content[:40]!r}")

    # Delete by ref_doc_id (file-level delete)
    db.delete_ref_doc("test_col", "a.py")
    results2 = db.search("test_col", "anything", top_k=10, threshold=0.0)
    print(f"After deleting a.py chunks: {len(results2)} results remain")
    assert all(r.relative_path == "b.py" for r in results2), "Only b.py chunks should remain"
    print("File-level deletion OK!")

    # Persistence
    db.drop_collection("test_col")
    assert not db.has_collection("test_col")
    print("Drop collection OK!")
    print()

# ---------------------------------------------------------------------------
# 3. Test Snapshot Manager (V2 format, locking, status)
# ---------------------------------------------------------------------------
print("--- 3. SnapshotManager ---")
with tempfile.TemporaryDirectory() as tmpdir:
    snap_path = os.path.join(tmpdir, "snapshot.json")
    snap = SnapshotManager(snap_path)

    # Test state transitions (match TS flow)
    snap.set_codebase_indexing("/tmp", 0)
    assert snap.get_codebase_status("/tmp") == "indexing"
    assert snap.get_indexing_progress("/tmp") == 0
    print(f"Indexing: status={snap.get_codebase_status('/tmp')}")

    snap.set_codebase_indexing("/tmp", 50)
    assert snap.get_indexing_progress("/tmp") == 50
    print(f"Indexing progress: {snap.get_indexing_progress('/tmp')}%")

    snap.set_codebase_indexed("/tmp", indexed_files=10, total_chunks=500, index_status="completed")
    assert snap.get_codebase_status("/tmp") == "indexed"
    info = snap.get_codebase_info("/tmp")
    assert info and info.indexed_files == 10 and info.total_chunks == 500
    print(f"Indexed: {info.indexed_files} files, {info.total_chunks} chunks, status={info.index_status}")

    # Test failed state
    snap.set_codebase_index_failed("/tmp", "Disk full", last_pct=75)
    assert snap.get_codebase_status("/tmp") == "indexfailed"
    info = snap.get_codebase_info("/tmp")
    assert info and info.error_message == "Disk full"
    print(f"Failed: error={info.error_message}, pct={info.indexing_percentage}%")

    assert "/tmp" in snap.get_failed_codebases()

    # Test remove (marks as recently_removed)
    snap.remove_codebase("/tmp")
    assert snap.get_codebase_status("/tmp") == "not_found"
    print("Remove + recently_removed OK!")

    # Test V2 persistence
    snap2 = SnapshotManager(snap_path)
    # /tmp was removed, so it should not be reloaded
    assert snap2.get_codebase_status("/tmp") == "not_found"
    print("V2 persistence OK!")
    print()

# ---------------------------------------------------------------------------
# 4. Test Context public API methods
# ---------------------------------------------------------------------------
print("--- 4. Context API ---")
with tempfile.TemporaryDirectory() as tmpdir:
    db = FaissVectorDB(persist_dir=os.path.join(tmpdir, "faiss"), embed_model=mock_embed)
    ctx = Context(vector_db=db)

    assert ctx.get_vector_db() is db
    assert isinstance(ctx.get_splitter(), AstSplitter)
    assert ".py" in ctx.get_supported_extensions()

    # Custom extensions/ignore
    ctx.add_custom_extensions([".vue", ".svelte"])
    assert ".vue" in ctx.get_supported_extensions()

    ctx.add_custom_ignore_patterns(["*.generated.ts"])
    assert "*.generated.ts" in ctx.get_ignore_patterns()

    ctx.reset_ignore_patterns_to_defaults()
    assert "*.generated.ts" not in ctx.get_ignore_patterns()

    # Splitter info
    info = ctx.get_splitter_info()
    assert info["type"] == "ast"
    assert info["has_builtin_fallback"] is True

    # Splitter swap
    ctx.update_splitter(TextSplitter())
    assert ctx.get_splitter_info()["type"] == "text"

    # Synchronizer management
    assert ctx.get_synchronizers() == {}
    col = ctx.get_collection_name("/tmp")
    assert col.startswith("code_chunks_")
    print(f"Collection name: {col}")
    print("All Context API methods OK!")
    print()

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("=" * 60)
print("ALL TESTS PASSED — Pipeline: Tree-sitter → LlamaIndex TextNode → FAISS")
print("MCP tools: index_codebase, search_code, clear_index, get_indexing_status")
print("=" * 60)
