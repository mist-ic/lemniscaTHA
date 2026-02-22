"""
ClearPath RAG Chatbot — Embedding Module (ONNX Runtime)

Uses ONNX-exported all-MiniLM-L6-v2 for embedding inference.
Replaces sentence-transformers/PyTorch to reduce image size from ~4GB to ~1.2GB.
"""

import json
import os
from pathlib import Path
from typing import List

import numpy as np
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

from app.pipeline.chunker import Chunk

# Resolve ONNX model directory relative to project root
ONNX_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "onnx_model"


class Embedder:
    """Embeds text chunks using ONNX runtime and caches results to disk."""

    def __init__(
        self,
        model_dir: str = str(ONNX_MODEL_DIR),
        index_dir: str = "index",
    ):
        self.index_dir = index_dir
        self.embeddings_path = os.path.join(index_dir, "embeddings.npz")
        self.chunks_path = os.path.join(index_dir, "chunks.json")

        # Load tokenizer and ONNX session
        print(f"[Embedder] Loading ONNX model from: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.session = InferenceSession(os.path.join(model_dir, "model.onnx"))
        print(f"[Embedder] ONNX model loaded successfully (384 dimensions)")

    def _mean_pool_and_normalize(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Apply mean pooling over token embeddings, then L2 normalize."""
        # Expand attention mask for broadcasting: (batch, seq_len) → (batch, seq_len, 1)
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        # Sum token embeddings weighted by attention mask
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        # Divide by number of non-padding tokens
        counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts
        # L2 normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return (pooled / norms).astype(np.float32)

    def has_cached_index(self) -> bool:
        """Check if pre-computed embeddings exist on disk."""
        return (
            os.path.exists(self.embeddings_path)
            and os.path.exists(self.chunks_path)
        )

    def load_index(self) -> tuple:
        """Load pre-computed embeddings and chunk metadata from disk."""
        print(f"[Embedder] Loading cached index from {self.index_dir}/")
        data = np.load(self.embeddings_path)
        embeddings = data["embeddings"]
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks_meta = json.load(f)
        print(f"[Embedder] Loaded {len(chunks_meta)} chunks, embeddings shape: {embeddings.shape}")
        return embeddings, chunks_meta

    def build_index(self, chunks: List[Chunk]) -> tuple:
        """Embed all chunks using ONNX and save to disk."""
        print(f"[Embedder] Embedding {len(chunks)} chunks with ONNX...")
        texts = [c.text for c in chunks]

        # Process in batches to avoid OOM on constrained environments (Cloud Run 2Gi)
        batch_size = 16
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=512, return_tensors="np"
            )
            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)
            token_type_ids = encoded.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64)

            outputs = self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            })
            batch_emb = self._mean_pool_and_normalize(outputs[0], attention_mask)
            all_embeddings.append(batch_emb)
            print(f"[Embedder]   Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size} done")

        embeddings = np.concatenate(all_embeddings, axis=0)

        # Build metadata
        chunks_meta = [
            {
                "chunk_id": c.chunk_id, "document": c.document, "page": c.page,
                "section_heading": c.section_heading, "text": c.text, "token_count": c.token_count,
            }
            for c in chunks
        ]

        # Save
        os.makedirs(self.index_dir, exist_ok=True)
        np.savez_compressed(self.embeddings_path, embeddings=embeddings)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_meta, f, ensure_ascii=False, indent=2)

        print(f"[Embedder] Index saved to {self.index_dir}/ — {embeddings.shape}")
        return embeddings, chunks_meta

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string for retrieval."""
        encoded = self.tokenizer(
            [text], padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        token_type_ids = encoded.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int64)

        outputs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })
        embeddings = self._mean_pool_and_normalize(outputs[0], attention_mask)
        return embeddings[0]


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
