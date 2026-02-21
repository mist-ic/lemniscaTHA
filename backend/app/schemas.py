"""
ClearPath RAG Chatbot — Pydantic Schemas

Request/Response models matching API_CONTRACT.md exactly.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ─── Request ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    """POST /query request body."""
    question: str
    conversation_id: Optional[str] = None


# ─── Response (nested models) ─────────────────────────────

class TokenUsage(BaseModel):
    """Token usage breakdown."""
    input: int
    output: int


class QueryMetadata(BaseModel):
    """Metadata about the query processing."""
    model_used: str
    classification: str  # "simple" or "complex"
    tokens: TokenUsage
    latency_ms: int
    chunks_retrieved: int
    evaluator_flags: List[str] = Field(default_factory=list)


class SourceInfo(BaseModel):
    """Information about a retrieved source document."""
    document: str
    page: Optional[int] = None
    relevance_score: Optional[float] = None


class QueryResponse(BaseModel):
    """POST /query response body — matches API_CONTRACT.md."""
    answer: str
    metadata: QueryMetadata
    sources: List[SourceInfo]
    conversation_id: str
