"""
ClearPath RAG Chatbot — Vector Retriever

Pure NumPy cosine similarity search over pre-computed embeddings.
No external vector DB — sufficient for ~300 vectors.
"""

from typing import List, Tuple

import numpy as np


class Retriever:
    """Performs cosine similarity search over embedded chunks."""

    def __init__(self, embeddings: np.ndarray, chunks_meta: List[dict]):
        """
        Initialize the retriever with an embedding matrix and chunk metadata.

        Args:
            embeddings: numpy array of shape (N, dim), L2-normalized.
            chunks_meta: list of dicts with chunk metadata (matching chunks.json).
        """
        self.embeddings = embeddings  # (N, 384)
        self.chunks_meta = chunks_meta
        print(f"[Retriever] Initialized with {len(chunks_meta)} chunks")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.25,
    ) -> List[Tuple[dict, float]]:
        """
        Find the most relevant chunks for a query.

        Args:
            query_embedding: 1D normalized embedding vector for the query.
            top_k: Maximum number of results to return.
            threshold: Minimum cosine similarity score.

        Returns:
            List of (chunk_metadata, score) tuples, sorted by descending score.
            Only includes chunks with score >= threshold.
        """
        # Cosine similarity via dot product (vectors are L2-normalized)
        scores = self.embeddings @ query_embedding  # shape: (N,)

        # Get top-k indices sorted by descending score
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Filter by threshold
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold:
                results.append((self.chunks_meta[idx], score))

        return results


if __name__ == "__main__":
    from app.pipeline.embedder import Embedder

    embedder = Embedder()
    if not embedder.has_cached_index():
        print("No cached index found. Run embedder.py first to build the index.")
    else:
        embeddings, meta = embedder.load_index()
        retriever = Retriever(embeddings, meta)

        # Test queries
        test_queries = [
            "What is the Pro plan price?",
            "How do I set up Slack integration?",
            "What is ClearPath's PTO policy?",
            "What keyboard shortcuts are available?",
        ]

        for q in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {q}")
            query_vec = embedder.embed_query(q)
            results = retriever.search(query_vec)
            for chunk, score in results:
                print(f"  {score:.3f} | {chunk['document']} p{chunk['page']} | {chunk['text'][:80]}...")
