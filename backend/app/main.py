"""
ClearPath RAG Chatbot — FastAPI Main Application

API endpoints, startup, CORS, and static serving.
Wires together the complete RAG pipeline.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import QueryRequest, QueryResponse, QueryMetadata, TokenUsage, SourceInfo
from app.router.classifier import classify_query, is_greeting, GREETING_RESPONSE
from app.pipeline.embedder import Embedder
from app.pipeline.extractor import extract_all_pdfs
from app.pipeline.chunker import chunk_documents
from app.pipeline.retriever import Retriever
from app.pipeline.prompt import build_messages
from app.groq_client import GroqClient
from app.evaluator.flags import evaluate
from app.logger import log_query


# ─── Global state (populated on startup) ────────────────────

embedder: Embedder = None  # type: ignore
retriever: Retriever = None  # type: ignore
groq_client: GroqClient = None  # type: ignore


# ─── Lifespan (startup / shutdown) ──────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load or build embeddings index on startup."""
    global embedder, retriever, groq_client

    settings = get_settings()

    # 1. Initialize embedder
    embedder = Embedder(
        model_name=settings.EMBEDDING_MODEL,
        index_dir=settings.INDEX_DIR,
    )

    # 2. Load or build index
    if embedder.has_cached_index():
        print("[Startup] Loading cached embeddings index...")
        embeddings, chunks_meta = embedder.load_index()
    else:
        print("[Startup] Building embeddings index from PDFs...")
        docs = extract_all_pdfs(settings.DOCS_DIR)
        chunks = chunk_documents(
            docs,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        embeddings, chunks_meta = embedder.build_index(chunks)

    # 3. Initialize retriever
    retriever = Retriever(embeddings, chunks_meta)

    # 4. Initialize Groq client
    groq_client = GroqClient()
    print("[Startup] All components initialized. Ready to serve.")

    yield  # app runs here

    # Shutdown (nothing to clean up for now)
    print("[Shutdown] ClearPath RAG Chatbot shutting down.")


# ─── App ────────────────────────────────────────────────────

app = FastAPI(
    title="ClearPath RAG Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main query endpoint — full RAG pipeline.

    Flow:
    1. Parse request
    2. Generate/use conversation_id
    3. Classify query (router)
    4. If greeter → return canned response
    5. Embed query → retrieve chunks
    6. Build prompt with salted tags
    7. Call Groq API
    8. Run evaluator
    9. Log everything
    10. Return structured response
    """
    settings = get_settings()
    request_start = time.perf_counter()

    # 1. Conversation ID
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    # 2. Check for greeting
    if is_greeting(question):
        total_latency = int((time.perf_counter() - request_start) * 1000)

        log_query(
            query=question,
            classification="greeter",  # internal log keeps "greeter" for observability
            model_used=settings.SIMPLE_MODEL,
            complexity_score=0,
            signals={},
            tokens_input=0,
            tokens_output=0,
            latency_ms=total_latency,
            conversation_id=conversation_id,
            evaluator_flags=[],
            chunks_retrieved=0,
        )

        return QueryResponse(
            answer=GREETING_RESPONSE,
            metadata=QueryMetadata(
                model_used=settings.SIMPLE_MODEL,
                classification="simple",
                tokens=TokenUsage(input=0, output=0),
                latency_ms=total_latency,
                chunks_retrieved=0,
                evaluator_flags=[],
            ),
            sources=[],
            conversation_id=conversation_id,
        )

    # 3. Classify query
    classification, model, complexity_score, signals = classify_query(question)

    # 4. Embed query and retrieve chunks
    query_embedding = embedder.embed_query(question)
    results = retriever.search(
        query_embedding,
        top_k=settings.TOP_K,
        threshold=settings.SIMILARITY_THRESHOLD,
    )

    retrieved_chunks = [chunk for chunk, _ in results]
    chunks_retrieved = len(retrieved_chunks)

    # 5. Build prompt
    messages, salt = build_messages(
        query=question,
        chunks=retrieved_chunks,
        history=None,  # Phase 3 will add conversation memory
    )

    # 6. Call Groq API
    max_tokens = (
        settings.COMPLEX_MAX_TOKENS if classification == "complex"
        else settings.SIMPLE_MAX_TOKENS
    )

    try:
        result = groq_client.generate(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )
    except Exception as e:
        # Fallback: if 70B fails, try 8B
        if classification == "complex":
            print(f"[Query] 70B failed ({e}), falling back to 8B")
            model = settings.SIMPLE_MODEL
            result = groq_client.generate(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
            )
        else:
            raise HTTPException(status_code=503, detail=f"LLM service unavailable: {e}")

    answer = result["content"]
    prompt_tokens = result["prompt_tokens"]
    completion_tokens = result["completion_tokens"]
    llm_latency = result["latency_ms"]

    # 7. Run evaluator
    evaluator_flags = evaluate(
        answer=answer,
        chunks_retrieved=chunks_retrieved,
        retrieved_chunks=retrieved_chunks,
    )

    # 8. Total latency
    total_latency = int((time.perf_counter() - request_start) * 1000)

    # 9. Build sources
    sources = [
        SourceInfo(
            document=chunk["document"],
            page=chunk.get("page"),
            relevance_score=round(score, 4),
        )
        for chunk, score in results
    ]

    # 10. Log
    log_query(
        query=question,
        classification=classification,
        model_used=model,
        complexity_score=complexity_score,
        signals=signals,
        tokens_input=prompt_tokens,
        tokens_output=completion_tokens,
        latency_ms=total_latency,
        conversation_id=conversation_id,
        evaluator_flags=evaluator_flags,
        chunks_retrieved=chunks_retrieved,
    )

    # 11. Return response
    return QueryResponse(
        answer=answer,
        metadata=QueryMetadata(
            model_used=model,
            classification=classification,
            tokens=TokenUsage(input=prompt_tokens, output=completion_tokens),
            latency_ms=total_latency,
            chunks_retrieved=chunks_retrieved,
            evaluator_flags=evaluator_flags,
        ),
        sources=sources,
        conversation_id=conversation_id,
    )
