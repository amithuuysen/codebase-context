"""RAG accuracy evaluation — index a real codebase and measure retrieval quality.

Metrics:
  - Hit Rate @K : fraction of queries where ≥1 expected file appears in top K
  - MRR @K      : Mean Reciprocal Rank — how high the first correct result is
  - Recall @K   : fraction of expected files found in top K (multi-file queries)

Usage:
    uv run python eval_rag.py /path/to/codebase          # index + evaluate
    uv run python eval_rag.py /path/to/codebase --skip-index  # evaluate only
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from codecontext.core.context import Context
from codecontext.core.embedding import create_embedding
from codecontext.core.reranker import Reranker
from codecontext.core.splitter import AstSplitter
from codecontext.core.types import Config
from codecontext.core.vectordb import FaissVectorDB

# ---------------------------------------------------------------------------
# Evaluation dataset — queries a developer would ask on Zoho CRM
# Each case has:
#   query:    natural language question
#   expected: list of path fragments that MUST appear in results
#   kind:     "semantic" (meaning-based) | "keyword" (exact match) | "hybrid"
# ---------------------------------------------------------------------------

EVAL_CASES = [
    # --- Semantic queries (meaning-based, no exact keyword match) ---
    {
        "query": "how does lead conversion work",
        "expected": ["leads/util/LeadConvertUtil", "leads/util/ConvertFieldUpdateUtil"],
        "kind": "semantic",
    },
    {
        "query": "where is the OAuth token validated",
        "expected": ["iam/oauth/OAuth", "iam/oauth/OAuthRule"],
        "kind": "semantic",
    },
    {
        "query": "redis caching mechanism",
        "expected": ["cache/Redis"],
        "kind": "semantic",
    },
    {
        "query": "how are webhooks triggered in automation",
        "expected": ["Webhook"],
        "kind": "semantic",
    },
    {
        "query": "forecast authorization and permission checks",
        "expected": ["forecast/ForecastAuth"],
        "kind": "semantic",
    },
    {
        "query": "deal stage anomaly detection",
        "expected": ["dealsStage/DealsStageAnomaly"],
        "kind": "semantic",
    },
    {
        "query": "mass email sending utility",
        "expected": ["massmail/util/CrmMassMailUtil"],
        "kind": "semantic",
    },
    {
        "query": "social media workflow processing",
        "expected": ["social/workflow/SocialWorkflowProcessor"],
        "kind": "semantic",
    },
    {
        "query": "custom authorization and permission logic",
        "expected": ["authorization/CrmCustomAuth"],
        "kind": "semantic",
    },
    {
        "query": "AI chatbot workflow handler",
        "expected": ["chatbot/handler/ChatbotCreateWorkflowHandler", "chatbot/core/AskziaWorkflowHandler"],
        "kind": "semantic",
    },

    # --- Keyword queries (exact symbol/class name match) ---
    {
        "query": "CrmSharing class",
        "expected": ["accessibility/CrmSharing"],
        "kind": "keyword",
    },
    {
        "query": "LeadDBService",
        "expected": ["leads/util/LeadDBService"],
        "kind": "keyword",
    },
    {
        "query": "RedisConstants",
        "expected": ["cache/RedisConstants"],
        "kind": "keyword",
    },
    {
        "query": "ForecastEnum",
        "expected": ["forecast/ForecastEnum"],
        "kind": "keyword",
    },
    {
        "query": "WebhookLogService",
        "expected": ["WebhookLogService"],
        "kind": "keyword",
    },

    # --- Hybrid queries (mix of meaning + keywords) ---
    {
        "query": "lead conversion anomaly configuration",
        "expected": ["leadConversion/LeadsConversionAnomalyConfig"],
        "kind": "hybrid",
    },
    {
        "query": "platform installed plugins cache",
        "expected": ["CrmInstalledPluginsCacheUtil"],
        "kind": "hybrid",
    },
    {
        "query": "email intelligence notification handler",
        "expected": ["EmailIntelligenceHandler"],
        "kind": "hybrid",
    },
    {
        "query": "audit log table constants",
        "expected": ["AuditLogTableConstants"],
        "kind": "hybrid",
    },
    {
        "query": "Zia workflow utilities",
        "expected": ["ZiaWorkflowUtil"],
        "kind": "hybrid",
    },
]


def _match(expected_fragment: str, result_path: str) -> bool:
    """Check if an expected path fragment appears in a result path."""
    # Normalize separators
    norm = result_path.replace("\\", "/")
    return expected_fragment in norm


async def run_evaluation(codebase_path: str, top_k: int = 5, skip_index: bool = False, index_only: bool = False, max_files: int = 0):
    cfg = Config.from_env()

    emb_kwargs: dict = {"model": cfg.embedding_model}
    if cfg.embedding_provider == "ollama":
        emb_kwargs["host"] = cfg.ollama_host
    elif cfg.embedding_provider == "openai":
        emb_kwargs["api_key"] = cfg.openai_api_key
        if cfg.openai_base_url:
            emb_kwargs["base_url"] = cfg.openai_base_url
    elif cfg.embedding_provider == "llamacpp":
        emb_kwargs["model"] = cfg.llamacpp_model_path
    embed_model = create_embedding(cfg.embedding_provider, **emb_kwargs)

    persist_dir = Path(cfg.data_dir) / "faiss_store"
    vector_db = FaissVectorDB(persist_dir=persist_dir, embed_model=embed_model)
    splitter = AstSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    reranker_provider = os.getenv("RERANKER_PROVIDER", "none")
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = Reranker(provider=reranker_provider, model=reranker_model)

    ctx = Context(vector_db=vector_db, splitter=splitter, config=cfg, reranker=reranker)

    # --- Index if needed ---
    if not skip_index:
        print(f"Indexing {codebase_path}...")
        print(f"  Provider: {cfg.embedding_provider} / {cfg.embedding_model}")
        print(f"  Chunk size: {cfg.chunk_size}, overlap: {cfg.chunk_overlap}")
        print(f"  Index dir: {cfg.data_dir}")
        t0 = time.time()
        _last_report = [0.0]

        def progress(phase: str, current: int, total: int, pct: int):
            now = time.time()
            if now - _last_report[0] >= 1.0:  # report every second
                _last_report[0] = now
                elapsed_so_far = now - t0
                rate = current / elapsed_so_far if elapsed_so_far > 0 else 0
                eta = (total - current) / rate if rate > 0 else 0
                print(f"  [{phase}] {pct}% ({current}/{total}) — {rate:.1f} files/s, ETA {eta:.0f}s")

        result = await ctx.index_codebase(codebase_path, progress=progress, max_files=max_files)
        elapsed = time.time() - t0
        print(f"  Indexed: {result['indexed_files']} files, {result['total_chunks']} chunks in {elapsed:.1f}s")
        print()
        if index_only:
            return {"indexed": True, "files": result['indexed_files'], "chunks": result['total_chunks'], "elapsed": elapsed}
    else:
        has = await ctx.has_index(codebase_path)
        if not has:
            print(f"ERROR: No index found for {codebase_path}. Run without --skip-index first.")
            sys.exit(1)
        print(f"Using existing index for {codebase_path}")
        print()

    # --- Evaluate ---
    print(f"Running {len(EVAL_CASES)} evaluation queries (top_k={top_k})")
    print(f"Reranker: {reranker_provider}")
    print("=" * 70)

    total_hits = 0
    total_mrr = 0.0
    total_recall = 0.0
    results_by_kind: dict[str, dict] = {}

    for case in EVAL_CASES:
        query = case["query"]
        expected = case["expected"]
        kind = case["kind"]

        if kind not in results_by_kind:
            results_by_kind[kind] = {"hits": 0, "mrr": 0.0, "recall": 0.0, "n": 0}

        try:
            results = await ctx.semantic_search(
                codebase_path, query, top_k=top_k, threshold=0.0
            )
        except Exception as exc:
            print(f"  ❌ \"{query}\" — ERROR: {exc}")
            results_by_kind[kind]["n"] += 1
            continue

        found_paths = [r.relative_path for r in results]

        # Hit @K
        hit = any(
            any(_match(exp, fp) for fp in found_paths)
            for exp in expected
        )
        total_hits += int(hit)
        results_by_kind[kind]["hits"] += int(hit)

        # MRR
        rr = 0.0
        for rank, fp in enumerate(found_paths, 1):
            if any(_match(exp, fp) for exp in expected):
                rr = 1.0 / rank
                break
        total_mrr += rr
        results_by_kind[kind]["mrr"] += rr

        # Recall @K
        found_expected = sum(
            1 for exp in expected
            if any(_match(exp, fp) for fp in found_paths)
        )
        recall = found_expected / len(expected) if expected else 0.0
        total_recall += recall
        results_by_kind[kind]["recall"] += recall

        results_by_kind[kind]["n"] += 1

        # Print result
        status = "✅" if hit else "❌"
        print(f"  {status} [{kind:8s}] \"{query}\"")
        if not hit:
            print(f"       Expected: {expected}")
            print(f"       Got top {min(3, len(found_paths))}:")
            for fp in found_paths[:3]:
                print(f"         - {fp}")
        elif rr < 1.0:
            print(f"       Found at rank {int(1/rr)} (not #1)")

    # --- Summary ---
    n = len(EVAL_CASES)
    print()
    print("=" * 70)
    print(f"OVERALL (n={n})")
    print(f"  Hit Rate @{top_k}: {total_hits}/{n} ({total_hits/n*100:.1f}%)")
    print(f"  MRR @{top_k}:      {total_mrr/n:.3f}")
    print(f"  Recall @{top_k}:   {total_recall/n:.3f}")
    print()

    print("BY QUERY TYPE:")
    for kind, stats in sorted(results_by_kind.items()):
        k_n = stats["n"]
        if k_n == 0:
            continue
        print(f"  {kind:10s}  Hit={stats['hits']}/{k_n} ({stats['hits']/k_n*100:.0f}%)  "
              f"MRR={stats['mrr']/k_n:.3f}  Recall={stats['recall']/k_n:.3f}")

    print()
    print("INTERPRETATION:")
    overall_hit = total_hits / n * 100
    if overall_hit >= 80:
        print("  🟢 Good: ≥80% hit rate — hybrid search is working well")
    elif overall_hit >= 60:
        print("  🟡 Fair: 60-80% — consider enabling cross-encoder reranker")
    else:
        print("  🔴 Low: <60% — check embedding model, chunk size, or eval queries")

    overall_mrr = total_mrr / n
    if overall_mrr >= 0.7:
        print("  🟢 MRR ≥0.7 — correct results are typically rank 1-2")
    elif overall_mrr >= 0.4:
        print("  🟡 MRR 0.4-0.7 — correct results found but not top-ranked, reranker may help")
    else:
        print("  🔴 MRR <0.4 — relevant results are buried, investigate embedding quality")

    return {
        "hit_rate": total_hits / n,
        "mrr": total_mrr / n,
        "recall": total_recall / n,
        "by_kind": {
            k: {"hit_rate": v["hits"] / v["n"], "mrr": v["mrr"] / v["n"], "recall": v["recall"] / v["n"]}
            for k, v in results_by_kind.items() if v["n"] > 0
        },
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python eval_rag.py <codebase_path> [--skip-index] [--top-k N]")
        sys.exit(1)

    codebase_path = sys.argv[1]
    skip_index = "--skip-index" in sys.argv
    index_only = "--index-only" in sys.argv
    top_k = 5
    max_files = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--top-k" and i + 1 < len(sys.argv):
            top_k = int(sys.argv[i + 1])
        if arg == "--max-files" and i + 1 < len(sys.argv):
            max_files = int(sys.argv[i + 1])

    if index_only:
        # Index only — no evaluation
        asyncio.run(run_evaluation(codebase_path, top_k=top_k, skip_index=False, index_only=True, max_files=max_files))
    else:
        asyncio.run(run_evaluation(codebase_path, top_k=top_k, skip_index=skip_index, max_files=max_files))
