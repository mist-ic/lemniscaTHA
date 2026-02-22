"""
Empirical comparison: all-MiniLM-L6-v2 (current) vs BAAI/bge-small-en-v1.5

Tests retrieval quality across the same queries used in our eval harness.
Compares top-5 retrieval scores for each model.
"""

import json
import os
import sys
import time

import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# ─── Test queries (from our eval harness) ──────────────────
TEST_QUERIES = [
    # Pricing
    ("pricing-001", "What does the Pro plan cost?"),
    ("pricing-003", "Compare Pro and Enterprise pricing"),
    # Features
    ("feature-001", "What are the keyboard shortcuts to create a task and search?"),
    ("feature-002", "Does the mobile app work offline?"),
    ("feature-003", "How do I set up the Slack integration?"),
    # Policy
    ("policy-001", "What is the PTO policy?"),
    ("policy-002", "Can I work remotely?"),
    # Edge cases
    ("edge-003", "Is Clearpath a good tool for managing construction projects?"),
    # Injection-related
    ("inject-001", "What does the Pro plan cost?"),
    ("inject-004", "How do I set up integrations?"),
    # Conflict
    ("conflict-001", "How much does the Pro plan cost per month?"),
    ("conflict-002", "When was the timeline view feature released or shipped?"),
    # Technical
    ("technical-001", "How do I use the API?"),
    # Router
    ("router-002", "My timeline view isn't loading after upgrading"),
]


def test_onnx_minilm(chunks_meta):
    """Current model: all-MiniLM-L6-v2 via ONNX"""
    from app.pipeline.embedder import Embedder

    # Use absolute path for index dir
    index_dir = os.path.join(os.path.dirname(__file__), "..", "backend", "index")
    embedder = Embedder(index_dir=index_dir)
    if embedder.has_cached_index():
        embeddings, _ = embedder.load_index()
    else:
        print(f"No cached index at {index_dir}, skipping ONNX MiniLM")
        return None

    results = {}
    for qid, query in TEST_QUERIES:
        qvec = embedder.embed_query(query)
        scores = embeddings @ qvec
        top_indices = np.argsort(scores)[::-1][:5]
        top_results = []
        for idx in top_indices:
            s = float(scores[idx])
            if s >= 0.25:
                top_results.append((chunks_meta[idx]["document"], s))
        results[qid] = {
            "top_score": float(scores[top_indices[0]]),
            "top_doc": chunks_meta[top_indices[0]]["document"],
            "num_above_threshold": len(top_results),
            "scores": [float(scores[i]) for i in top_indices[:5]],
        }
    return results


def test_sentence_transformers_bge(chunks_meta):
    """Alternative: BAAI/bge-small-en-v1.5 via sentence-transformers"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])
        from sentence_transformers import SentenceTransformer

    print("\n[BGE] Loading BAAI/bge-small-en-v1.5...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # Re-embed all chunks
    texts = [c["text"] for c in chunks_meta]
    print(f"[BGE] Embedding {len(texts)} chunks...")
    t0 = time.time()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embed_time = time.time() - t0
    print(f"[BGE] Embedded in {embed_time:.2f}s")

    results = {}
    for qid, query in TEST_QUERIES:
        qvec = model.encode([query], normalize_embeddings=True)[0]
        scores = embeddings @ qvec
        top_indices = np.argsort(scores)[::-1][:5]
        top_results = []
        for idx in top_indices:
            s = float(scores[idx])
            if s >= 0.25:
                top_results.append((chunks_meta[idx]["document"], s))
        results[qid] = {
            "top_score": float(scores[top_indices[0]]),
            "top_doc": chunks_meta[top_indices[0]]["document"],
            "num_above_threshold": len(top_results),
            "scores": [float(scores[i]) for i in top_indices[:5]],
        }
    return results


def test_sentence_transformers_minilm(chunks_meta):
    """Control: all-MiniLM-L6-v2 via sentence-transformers (to compare with ONNX)"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    print("\n[ST-MiniLM] Loading all-MiniLM-L6-v2 via sentence-transformers...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["text"] for c in chunks_meta]
    print(f"[ST-MiniLM] Embedding {len(texts)} chunks...")
    t0 = time.time()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embed_time = time.time() - t0
    print(f"[ST-MiniLM] Embedded in {embed_time:.2f}s")

    results = {}
    for qid, query in TEST_QUERIES:
        qvec = model.encode([query], normalize_embeddings=True)[0]
        scores = embeddings @ qvec
        top_indices = np.argsort(scores)[::-1][:5]
        top_results = []
        for idx in top_indices:
            s = float(scores[idx])
            if s >= 0.25:
                top_results.append((chunks_meta[idx]["document"], s))
        results[qid] = {
            "top_score": float(scores[top_indices[0]]),
            "top_doc": chunks_meta[top_indices[0]]["document"],
            "num_above_threshold": len(top_results),
            "scores": [float(scores[i]) for i in top_indices[:5]],
        }
    return results


def main():
    # Load chunk metadata
    chunks_path = os.path.join(os.path.dirname(__file__), "..", "backend", "index", "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_meta = json.load(f)
    print(f"Loaded {len(chunks_meta)} chunks from index")

    # Test current ONNX MiniLM
    print("\n" + "=" * 70)
    print("MODEL 1: all-MiniLM-L6-v2 (ONNX — current production)")
    print("=" * 70)
    onnx_results = test_onnx_minilm(chunks_meta)

    # Test BGE-small
    print("\n" + "=" * 70)
    print("MODEL 2: BAAI/bge-small-en-v1.5 (sentence-transformers)")
    print("=" * 70)
    bge_results = test_sentence_transformers_bge(chunks_meta)

    # Test sentence-transformers MiniLM (control)
    print("\n" + "=" * 70)
    print("MODEL 3: all-MiniLM-L6-v2 (sentence-transformers — control)")
    print("=" * 70)
    st_results = test_sentence_transformers_minilm(chunks_meta)

    # ─── Comparison ────────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Query ID':<16} {'ONNX-MiniLM':>12} {'BGE-small':>12} {'ST-MiniLM':>12}  {'Winner':>12}")
    print("-" * 90)

    onnx_wins = 0
    bge_wins = 0
    ties = 0

    for qid, query in TEST_QUERIES:
        onnx_s = onnx_results[qid]["top_score"] if onnx_results else 0
        bge_s = bge_results[qid]["top_score"] if bge_results else 0
        st_s = st_results[qid]["top_score"] if st_results else 0

        diff = bge_s - onnx_s
        if abs(diff) < 0.02:
            winner = "TIE"
            ties += 1
        elif diff > 0:
            winner = f"BGE +{diff:.3f}"
            bge_wins += 1
        else:
            winner = f"ONNX +{-diff:.3f}"
            onnx_wins += 1

        print(f"{qid:<16} {onnx_s:>12.4f} {bge_s:>12.4f} {st_s:>12.4f}  {winner:>12}")

    print("-" * 90)
    print(f"\nONNX MiniLM wins: {onnx_wins}  |  BGE-small wins: {bge_wins}  |  Ties (<0.02): {ties}")

    # Average scores
    if onnx_results and bge_results:
        avg_onnx = np.mean([onnx_results[qid]["top_score"] for qid, _ in TEST_QUERIES])
        avg_bge = np.mean([bge_results[qid]["top_score"] for qid, _ in TEST_QUERIES])
        print(f"\nAverage top-1 score — ONNX-MiniLM: {avg_onnx:.4f} | BGE-small: {avg_bge:.4f} | Δ: {avg_bge - avg_onnx:+.4f}")

    # Show top-doc agreement
    if onnx_results and bge_results:
        agree = sum(
            1 for qid, _ in TEST_QUERIES
            if onnx_results[qid]["top_doc"] == bge_results[qid]["top_doc"]
        )
        print(f"Top-1 document agreement: {agree}/{len(TEST_QUERIES)}")


if __name__ == "__main__":
    main()
