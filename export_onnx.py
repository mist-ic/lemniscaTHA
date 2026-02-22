"""
One-time script: Export all-MiniLM-L6-v2 to ONNX format.

Produces backend/onnx_model/ directory (~90MB) containing:
  - model.onnx
  - tokenizer.json, tokenizer_config.json, etc.

Usage:
    cd lemniscaTHA
    python export_onnx.py
"""

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "backend/onnx_model"

print(f"[Export] Exporting {MODEL_NAME} to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
model.save_pretrained(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"[Export] Done! ONNX model saved to {OUTPUT_DIR}/")
