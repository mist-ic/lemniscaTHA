"""
ClearPath RAG Chatbot — Embedding Module

Uses sentence-transformers all-MiniLM-L6-v2 to embed chunks.
Saves/loads embeddings to/from disk for fast subsequent startups.
"""

import json
import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.pipeline.chunker import Chunk


class Embedder:
    """Embeds text chunks and caches results to disk."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_dir: str = "index",
    ):
        self.model_name = model_name
        self.index_dir = index_dir
        self.embeddings_path = os.path.join(index_dir, "embeddings.npz")
        self.chunks_path = os.path.join(index_dir, "chunks.json")

        # Load the sentence-transformer model
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"[Embedder] Model loaded successfully (dim={self.model.get_sentence_embedding_dimension()})")

    def has_cached_index(self) -> bool:
        """Check if pre-computed embeddings exist on disk."""
        return (
            os.path.exists(self.embeddings_path)
            and os.path.exists(self.chunks_path)
        )

    def load_index(self) -> tuple:
        """
        Load pre-computed embeddings and chunk metadata from disk.

        Returns:
            (embeddings_matrix, chunks_metadata): numpy array and list of dicts.
        """
        print(f"[Embedder] Loading cached index from {self.index_dir}/")
        data = np.load(self.embeddings_path)
        embeddings = data["embeddings"]

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks_meta = json.load(f)

        print(f"[Embedder] Loaded {len(chunks_meta)} chunks, embeddings shape: {embeddings.shape}")
        return embeddings, chunks_meta

    def build_index(self, chunks: List[Chunk]) -> tuple:
        """
        Embed all chunks and save to disk.

        Args:
            chunks: List of Chunk objects to embed.

        Returns:
            (embeddings_matrix, chunks_metadata): numpy array and list of dicts.
        """
        print(f"[Embedder] Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]

        # Encode with L2 normalization (enables dot-product = cosine similarity)
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build metadata list
        chunks_meta = [
            {
                "chunk_id": c.chunk_id,
                "document": c.document,
                "page": c.page,
                "section_heading": c.section_heading,
                "text": c.text,
                "token_count": c.token_count,
            }
            for c in chunks
        ]

        # Save to disk
        os.makedirs(self.index_dir, exist_ok=True)

        np.savez_compressed(self.embeddings_path, embeddings=embeddings)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_meta, f, ensure_ascii=False, indent=2)

        print(f"[Embedder] Index saved to {self.index_dir}/ — {embeddings.shape}")
        return embeddings, chunks_meta

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string for retrieval.

        Args:
            text: Query text.

        Returns:
            Normalized 1D embedding vector.
        """
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)


if __name__ == "__main__":
    from app.pipeline.extractor import extract_all_pdfs
    from app.pipeline.chunker import chunk_documents

    # Extract → Chunk → Embed
    docs = extract_all_pdfs()
    chunks = chunk_documents(docs)

    embedder = Embedder()
    if embedder.has_cached_index():
        embeddings, meta = embedder.load_index()
    else:
        embeddings, meta = embedder.build_index(chunks)

    # Test query embedding
    q = embedder.embed_query("What is the Pro plan price?")
    print(f"Query embedding shape: {q.shape}")
    print(f"Embedding norm: {np.linalg.norm(q):.4f}")
