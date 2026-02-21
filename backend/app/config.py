"""
ClearPath RAG Chatbot — Configuration

Uses pydantic-settings to load environment variables from .env file.
Contains all constants for the RAG pipeline.
"""

import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings

# Project root = lemniscaTHA/ (two levels up from this file: config.py -> app/ -> backend/ -> lemniscaTHA/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # lemniscaTHA/


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- API Keys ---
    GROQ_API_KEY: str

    # --- Server ---
    PORT: int = 8000

    # --- Models ---
    SIMPLE_MODEL: str = "llama-3.1-8b-instant"
    COMPLEX_MODEL: str = "llama-3.3-70b-versatile"

    # --- RAG Pipeline ---
    CHUNK_SIZE: int = 400       # max tokens per chunk
    CHUNK_OVERLAP: int = 60     # overlap tokens between adjacent chunks
    TOP_K: int = 5              # number of top chunks to retrieve
    SIMILARITY_THRESHOLD: float = 0.25  # minimum cosine similarity score

    # --- Paths (absolute, relative to project root) ---
    DOCS_DIR: str = str(PROJECT_ROOT / "docs")
    INDEX_DIR: str = str(PROJECT_ROOT / "backend" / "index")

    # --- Embedding Model ---
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # --- Generation ---
    SIMPLE_MAX_TOKENS: int = 512
    COMPLEX_MAX_TOKENS: int = 1024

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Returns cached Settings instance (singleton)."""
    return Settings()
