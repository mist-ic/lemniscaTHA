"""
Comprehensive embedding model benchmark.
Tests all viable models against our 93 chunks on the same 14 queries.
Measures: retrieval quality (top-1/top-5 scores) + query embedding speed.
"""

import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ─── Test queries ──────────────────────────────────────────
TEST_QUERIES = [
    ("pricing-001", "What does the Pro plan cost?"),
    ("pricing-003", "Compare Pro and Enterprise pricing"),
    ("feature-001", "What are the keyboard shortcuts to create a task and search?"),
    ("feature-002", "Does the mobile app work offline?"),
    ("feature-003", "How do I set up the Slack integration?"),
    ("policy-001", "What is the PTO policy?"),
    ("policy-002", "Can I work remotely?"),
    ("edge-003", "Is Clearpath a good tool for managing construction projects?"),
    ("inject-001", "What does the Pro plan cost?"),
    ("inject-004", "How do I set up integrations?"),
    ("conflict-001", "How much does the Pro plan cost per month?"),
    ("conflict-002", "When was the timeline view feature released or shipped?"),
    ("technical-001", "How do I use the API?"),
    ("router-002", "My timeline view isn't loading after upgrading"),
]

# ─── Models to test ──────────────────────────────────────
MODELS = [
    {
        "name": "all-MiniLM-L6-v2",
        "hf_id": "all-MiniLM-L6-v2",
        "params": "22M",
        "dims": 384,
        "mteb_retrieval": "~49.5",
        "prefix": None,
    },
    {
        "name": "BGE-small-en-v1.5",
        "hf_id": "BAAI/bge-small-en-v1.5",
        "params": "33M",
        "dims": 384,
        "mteb_retrieval": "~51.7",
        "prefix": "Represent this sentence for searching relevant passages: ",
    },
    {
        "name": "BGE-large-en-v1.5",
        "hf_id": "BAAI/bge-large-en-v1.5",
        "params": "335M",
        "dims": 1024,
        "mteb_retrieval": "~54.3",
        "prefix": "Represent this sentence for searching relevant passages: ",
    },
    {
        "name": "GTE-large-en-v1.5",
        "hf_id": "Alibaba-NLP/gte-large-en-v1.5",
        "params": "335M",
        "dims": 1024,
        "mteb_retrieval": "~55.2",
        "prefix": None,
        "trust_remote_code": True,
    },
    {
        "name": "Snowflake-arctic-embed-s",
        "hf_id": "Snowflake/snowflake-arctic-embed-s",
        "params": "33M",
        "dims": 384,
        "mteb_retrieval": "~51.0",
        "prefix": "Represent this sentence for searching relevant passages: ",
    },
    {
        "name": "Stella-400M-v5",
        "hf_id": "dunzhang/stella_en_400M_v5",
        "params": "400M",
        "dims": 1024,
        "mteb_retrieval": "~57.4",
        "prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: ",
        "trust_remote_code": True,
    },
    {
        "name": "BGE-M3 (dense)",
        "hf_id": "BAAI/bge-m3",
        "params": "568M",
        "dims": 1024,
        "mteb_retrieval": "~54.9",
        "prefix": None,
    },
]


def test_model(model_config, texts, chunks_meta):
    """Test a single model: embed all chunks + queries, measure quality and speed."""
    from sentence_transformers import SentenceTransformer

    name = model_config["name"]
    hf_id = model_config["hf_id"]
    prefix = model_config.get("prefix")
    trust = model_config.get("trust_remote_code", False)

    print(f"\n{'-' * 60}")
    print(f"  {name} ({model_config['params']}, {model_config['dims']}d)")
    print(f"  HF: {hf_id}")
    print(f"{'-' * 60}")

    try:
        print(f"  Loading model...", end=" ", flush=True)
        t0 = time.time()
        model = SentenceTransformer(hf_id, trust_remote_code=trust)
        load_time = time.time() - t0
        print(f"done ({load_time:.1f}s)")
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None

    # Embed all chunks
    print(f"  Embedding {len(texts)} chunks...", end=" ", flush=True)
    t0 = time.time()
    chunk_embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    chunk_time = time.time() - t0
    print(f"done ({chunk_time:.2f}s)")

    # Embed queries and measure speed
    query_times = []
    results = {}

    for qid, query in TEST_QUERIES:
        q_text = f"{prefix}{query}" if prefix else query

        t0 = time.time()
        q_vec = model.encode([q_text], normalize_embeddings=True)[0]
        q_time = (time.time() - t0) * 1000  # ms
        query_times.append(q_time)

        scores = chunk_embeddings @ q_vec
        top_indices = np.argsort(scores)[::-1][:5]

        above_threshold = sum(1 for i in top_indices if scores[i] >= 0.25)

        results[qid] = {
            "top_score": float(scores[top_indices[0]]),
            "top_doc": chunks_meta[top_indices[0]]["document"],
            "scores": [float(scores[i]) for i in top_indices[:5]],
            "num_above_025": above_threshold,
            "query_ms": q_time,
        }

    avg_query_ms = np.mean(query_times)
    avg_top1 = np.mean([r["top_score"] for r in results.values()])

    print(f"  Avg top-1 score: {avg_top1:.4f}")
    print(f"  Avg query time:  {avg_query_ms:.1f}ms")
    print(f"  Model load:      {load_time:.1f}s")

    return {
        "name": name,
        "params": model_config["params"],
        "dims": model_config["dims"],
        "mteb": model_config["mteb_retrieval"],
        "avg_top1": avg_top1,
        "avg_query_ms": avg_query_ms,
        "load_time_s": load_time,
        "chunk_embed_s": chunk_time,
        "results": results,
    }


def main():
    # Load chunks
    chunks_path = os.path.join(os.path.dirname(__file__), "..", "backend", "index", "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_meta = json.load(f)
    texts = [c["text"] for c in chunks_meta]
    print(f"Loaded {len(chunks_meta)} chunks\n")

    all_results = []

    for model_config in MODELS:
        result = test_model(model_config, texts, chunks_meta)
        if result:
            all_results.append(result)

    # ─── Final comparison table ────────────────────────
    print("\n\n" + "=" * 110)
    print("FINAL COMPARISON — RANKED BY RETRIEVAL QUALITY")
    print("=" * 110)

    # Sort by avg_top1 descending
    all_results.sort(key=lambda r: r["avg_top1"], reverse=True)

    print(f"\n{'Rank':<5} {'Model':<25} {'Params':<8} {'Dims':<6} {'Avg Top-1':>10} {'Query ms':>10} {'MTEB(BEIR)':>12}")
    print("-" * 110)

    baseline = None
    for i, r in enumerate(all_results):
        if r["name"] == "all-MiniLM-L6-v2":
            baseline = r["avg_top1"]

    for i, r in enumerate(all_results):
        delta = f"(+{r['avg_top1'] - baseline:.3f})" if baseline else ""
        print(f"{i+1:<5} {r['name']:<25} {r['params']:<8} {r['dims']:<6} {r['avg_top1']:>10.4f} {r['avg_query_ms']:>10.1f} {r['mteb']:>12}  {delta}")

    # Per-query breakdown for top 3
    print(f"\n\n{'=' * 110}")
    print("PER-QUERY BREAKDOWN (Top 3 models)")
    print("=" * 110)

    top3 = all_results[:3]
    header = f"{'Query ID':<16}"
    for r in top3:
        header += f" {r['name']:>20}"
    print(header)
    print("-" * (16 + 21 * len(top3)))

    for qid, query in TEST_QUERIES:
        row = f"{qid:<16}"
        for r in top3:
            score = r["results"][qid]["top_score"]
            row += f" {score:>20.4f}"
        print(row)

    # Agreement: how often do models agree on top-1 document?
    if len(all_results) >= 2:
        print(f"\n\nTOP-1 DOCUMENT AGREEMENT vs best model ({all_results[0]['name']}):")
        best = all_results[0]
        for r in all_results[1:]:
            agree = sum(
                1 for qid, _ in TEST_QUERIES
                if r["results"][qid]["top_doc"] == best["results"][qid]["top_doc"]
            )
            print(f"  {r['name']:<25}: {agree}/{len(TEST_QUERIES)} agree")


if __name__ == "__main__":
    main()
