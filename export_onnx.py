"""
One-time script: Export BAAI/bge-small-en-v1.5 to ONNX format.

Produces backend/onnx_model/ directory (~130MB) containing:
  - model.onnx
  - tokenizer.json, tokenizer_config.json, etc.

Usage:
    cd lemniscaTHA
    python export_onnx.py
"""

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
OUTPUT_DIR = "backend/onnx_model"

print(f"[Export] Exporting {MODEL_NAME} to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
model.save_pretrained(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"[Export] Done! ONNX model saved to {OUTPUT_DIR}/")
