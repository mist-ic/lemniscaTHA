"""Quantize BGE-small ONNX model to INT8 and compare quality + speed vs FP32."""

import os
import time
import json
import numpy as np
from onnxruntime import InferenceSession
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

MODEL_DIR = "backend/onnx_model"
FP32_PATH = os.path.join(MODEL_DIR, "model.onnx")
INT8_PATH = os.path.join(MODEL_DIR, "model_int8.onnx")

# ---- Step 1: Quantize ----
print("=== Quantizing FP32 -> INT8 ===")
t0 = time.time()
quantize_dynamic(FP32_PATH, INT8_PATH, weight_type=QuantType.QUInt8)
print(f"Done in {time.time()-t0:.1f}s")

fp32_mb = os.path.getsize(FP32_PATH) / 1024 / 1024
int8_mb = os.path.getsize(INT8_PATH) / 1024 / 1024
print(f"FP32: {fp32_mb:.1f} MB")
print(f"INT8: {int8_mb:.1f} MB")
print(f"Reduction: {(1 - int8_mb/fp32_mb)*100:.1f}%")

# ---- Step 2: Load both models ----
print("\n=== Loading both models ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
sess_fp32 = InferenceSession(FP32_PATH)
sess_int8 = InferenceSession(INT8_PATH)

with open("backend/index/chunks.json", "r", encoding="utf-8") as f:
    chunks_meta = json.load(f)
texts = [c["text"] for c in chunks_meta]


def embed_batch(session, texts_batch):
    encoded = tokenizer(
        texts_batch, padding=True, truncation=True, max_length=512, return_tensors="np"
    )
    ids = encoded["input_ids"].astype(np.int64)
    mask = encoded["attention_mask"].astype(np.int64)
    tids = encoded.get("token_type_ids", np.zeros_like(ids)).astype(np.int64)
    out = session.run(
        None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": tids}
    )
    mask_exp = np.expand_dims(mask, -1).astype(np.float32)
    pooled = np.sum(out[0] * mask_exp, axis=1) / np.clip(mask_exp.sum(axis=1), 1e-9, None)
    norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
    return (pooled / norms).astype(np.float32)


def embed_all(session, texts):
    all_emb = []
    for i in range(0, len(texts), 16):
        all_emb.append(embed_batch(session, texts[i : i + 16]))
    return np.concatenate(all_emb, axis=0)


# ---- Step 3: Embed all chunks ----
print("Embedding chunks with FP32...")
t0 = time.time()
emb_fp32 = embed_all(sess_fp32, texts)
fp32_chunk_time = time.time() - t0

print("Embedding chunks with INT8...")
t0 = time.time()
emb_int8 = embed_all(sess_int8, texts)
int8_chunk_time = time.time() - t0

# ---- Step 4: Compare retrieval ----
QUERIES = [
    ("pricing-001", "What does the Pro plan cost?"),
    ("pricing-003", "Compare Pro and Enterprise pricing"),
    ("feature-001", "What are the keyboard shortcuts to create a task and search?"),
    ("feature-002", "Does the mobile app work offline?"),
    ("policy-001", "What is the PTO policy?"),
    ("policy-002", "Can I work remotely?"),
    ("edge-003", "Is Clearpath a good tool for managing construction projects?"),
    ("conflict-001", "How much does the Pro plan cost per month?"),
    ("technical-001", "How do I use the API?"),
    ("router-002", "My timeline view isn't loading after upgrading"),
]

print(f"\n{'='*50}")
print("RETRIEVAL QUALITY: FP32 vs INT8")
print(f"{'='*50}")
print(f"{'Query':<16} {'FP32':>8} {'INT8':>8} {'Diff':>8}")
print("-" * 46)

fp32_scores = []
int8_scores = []
fp32_times = []
int8_times = []

for qid, q in QUERIES:
    t0 = time.time()
    qv_fp32 = embed_batch(sess_fp32, [q])[0]
    fp32_times.append((time.time() - t0) * 1000)

    t0 = time.time()
    qv_int8 = embed_batch(sess_int8, [q])[0]
    int8_times.append((time.time() - t0) * 1000)

    s_fp32 = float(np.max(emb_fp32 @ qv_fp32))
    s_int8 = float(np.max(emb_int8 @ qv_int8))
    fp32_scores.append(s_fp32)
    int8_scores.append(s_int8)
    diff = s_int8 - s_fp32
    print(f"{qid:<16} {s_fp32:>8.4f} {s_int8:>8.4f} {diff:>+8.4f}")

print("-" * 46)
avg_fp32 = np.mean(fp32_scores)
avg_int8 = np.mean(int8_scores)
print(f"{'AVERAGE':<16} {avg_fp32:>8.4f} {avg_int8:>8.4f} {avg_int8 - avg_fp32:>+8.4f}")

# Embedding similarity
cos_sims = np.sum(emb_fp32 * emb_int8, axis=1)
print(f"\nFP32 vs INT8 embedding cosine similarity:")
print(f"  avg={np.mean(cos_sims):.6f}, min={np.min(cos_sims):.6f}")

# Speed
print(f"\n{'='*50}")
print("SPEED COMPARISON")
print(f"{'='*50}")
pct_chunk = (1 - int8_chunk_time / fp32_chunk_time) * 100
pct_query = (1 - np.mean(int8_times) / np.mean(fp32_times)) * 100
print(f"Chunk embed (93):  FP32={fp32_chunk_time:.2f}s  INT8={int8_chunk_time:.2f}s  ({pct_chunk:+.1f}%)")
print(f"Query embed (avg): FP32={np.mean(fp32_times):.1f}ms  INT8={np.mean(int8_times):.1f}ms  ({pct_query:+.1f}%)")

# Top-1 agreement
agree = 0
for qid, q in QUERIES:
    qv_fp32 = embed_batch(sess_fp32, [q])[0]
    qv_int8 = embed_batch(sess_int8, [q])[0]
    top_fp32 = chunks_meta[np.argmax(emb_fp32 @ qv_fp32)]["document"]
    top_int8 = chunks_meta[np.argmax(emb_int8 @ qv_int8)]["document"]
    if top_fp32 == top_int8:
        agree += 1
print(f"\nTop-1 doc agreement: {agree}/{len(QUERIES)}")

print(f"\n{'='*50}")
print("VERDICT")
print(f"{'='*50}")
quality_loss = abs(avg_int8 - avg_fp32) / avg_fp32 * 100
print(f"Size:    {fp32_mb:.1f}MB -> {int8_mb:.1f}MB ({(1-int8_mb/fp32_mb)*100:.1f}% smaller)")
print(f"Quality: {quality_loss:.2f}% difference (avg retrieval score)")
print(f"Speed:   {pct_query:+.1f}% query latency change")
print(f"Agreement: {agree}/{len(QUERIES)} same top document")
