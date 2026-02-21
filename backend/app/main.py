"""
ClearPath RAG Chatbot — FastAPI Main Application

API endpoints, startup, CORS, and static serving.
Wires together the complete RAG pipeline.
"""

import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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
from app.memory.conversation import ConversationMemory


# ─── Global state (populated on startup) ────────────────────

embedder: Embedder = None  # type: ignore
retriever: Retriever = None  # type: ignore
groq_client: GroqClient = None  # type: ignore
memory: ConversationMemory = None  # type: ignore


# ─── Lifespan (startup / shutdown) ──────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load or build embeddings index on startup."""
    global embedder, retriever, groq_client, memory

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

    # 5. Initialize conversation memory
    memory = ConversationMemory()

    print("[Startup] All components initialized. Ready to serve.")

    yield  # app runs here

    # Shutdown
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


# ─── Helper: process query through pipeline ────────────────

async def _process_query_pipeline(question: str, conversation_id: str):
    """
    Shared pipeline logic for both /query and /query/stream.

    Returns a dict with all pipeline results needed for response building.
    """
    settings = get_settings()
    original_question = question

    # 1. Check for follow-up and rewrite if needed
    rewritten_query = question
    if memory.is_followup(question, conversation_id):
        rewritten_query = await memory.rewrite_query(
            query=question,
            conversation_id=conversation_id,
            groq_client=groq_client,
            model=settings.SIMPLE_MODEL,
        )

    # 2. Classify the (possibly rewritten) query
    classification, model, complexity_score, signals = classify_query(rewritten_query)

    # 3. Embed and retrieve
    query_embedding = embedder.embed_query(rewritten_query)
    results = retriever.search(
        query_embedding,
        top_k=settings.TOP_K,
        threshold=settings.SIMILARITY_THRESHOLD,
    )

    retrieved_chunks = [chunk for chunk, _ in results]
    chunks_retrieved = len(retrieved_chunks)

    # 4. Get conversation history for prompt
    history = memory.get_history(conversation_id)

    # 5. Build prompt
    messages, salt = build_messages(
        query=rewritten_query,
        chunks=retrieved_chunks,
        history=history,
    )

    # 6. Max tokens
    max_tokens = (
        settings.COMPLEX_MAX_TOKENS if classification == "complex"
        else settings.SIMPLE_MAX_TOKENS
    )

    return {
        "original_question": original_question,
        "rewritten_query": rewritten_query,
        "classification": classification,
        "model": model,
        "complexity_score": complexity_score,
        "signals": signals,
        "results": results,
        "retrieved_chunks": retrieved_chunks,
        "chunks_retrieved": chunks_retrieved,
        "messages": messages,
        "max_tokens": max_tokens,
    }


# ─── Endpoints ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main query endpoint — full RAG pipeline (non-streaming).

    Flow:
    1. Parse request + generate conversation_id
    2. Check for greeting
    3. Process through pipeline (follow-up detection, rewrite, classify, retrieve, prompt)
    4. Call Groq API (non-streaming)
    5. Run evaluator
    6. Store turn in memory
    7. Log everything
    8. Return structured response
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

        # Store greeting turn in memory
        memory.add_turn(conversation_id, question, GREETING_RESPONSE)

        log_query(
            query=question,
            classification="greeter",
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

    # 3. Process through pipeline
    pipeline = await _process_query_pipeline(question, conversation_id)

    # 4. Call Groq API (non-streaming)
    model = pipeline["model"]
    try:
        result = groq_client.generate(
            messages=pipeline["messages"],
            model=model,
            max_tokens=pipeline["max_tokens"],
        )
    except Exception as e:
        # Fallback: if 70B fails, try 8B
        if pipeline["classification"] == "complex":
            print(f"[Query] 70B failed ({e}), falling back to 8B")
            model = settings.SIMPLE_MODEL
            result = groq_client.generate(
                messages=pipeline["messages"],
                model=model,
                max_tokens=pipeline["max_tokens"],
            )
        else:
            raise HTTPException(status_code=503, detail=f"LLM service unavailable: {e}")

    answer = result["content"]
    prompt_tokens = result["prompt_tokens"]
    completion_tokens = result["completion_tokens"]

    # 5. Run evaluator
    evaluator_flags = evaluate(
        answer=answer,
        chunks_retrieved=pipeline["chunks_retrieved"],
        retrieved_chunks=pipeline["retrieved_chunks"],
    )

    # 6. Total latency
    total_latency = int((time.perf_counter() - request_start) * 1000)

    # 7. Build sources
    sources = [
        SourceInfo(
            document=chunk["document"],
            page=chunk.get("page"),
            relevance_score=round(score, 4),
        )
        for chunk, score in pipeline["results"]
    ]

    # 8. Store turn in conversation memory
    memory.add_turn(conversation_id, question, answer)

    # 9. Log
    log_query(
        query=question,
        classification=pipeline["classification"],
        model_used=model,
        complexity_score=pipeline["complexity_score"],
        signals=pipeline["signals"],
        tokens_input=prompt_tokens,
        tokens_output=completion_tokens,
        latency_ms=total_latency,
        conversation_id=conversation_id,
        evaluator_flags=evaluator_flags,
        chunks_retrieved=pipeline["chunks_retrieved"],
    )

    # 10. Return response
    return QueryResponse(
        answer=answer,
        metadata=QueryMetadata(
            model_used=model,
            classification=pipeline["classification"],
            tokens=TokenUsage(input=prompt_tokens, output=completion_tokens),
            latency_ms=total_latency,
            chunks_retrieved=pipeline["chunks_retrieved"],
            evaluator_flags=evaluator_flags,
        ),
        sources=sources,
        conversation_id=conversation_id,
    )


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming query endpoint — SSE (Server-Sent Events).

    Streams tokens as `data: {"token": "..."}` events.
    Final event: `data: {"done": true, "metadata": {...}, "sources": [...]}`.
    Error event: `data: {"error": "message"}`.
    """
    settings = get_settings()
    request_start = time.perf_counter()

    # 1. Conversation ID
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    # 2. Check for greeting — return as single SSE event
    if is_greeting(question):
        total_latency = int((time.perf_counter() - request_start) * 1000)
        memory.add_turn(conversation_id, question, GREETING_RESPONSE)

        log_query(
            query=question,
            classification="greeter",
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

        async def greeting_stream():
            # Send greeting as a single token event
            yield f"data: {json.dumps({'token': GREETING_RESPONSE})}\n\n"
            # Send done event
            yield f"data: {json.dumps({'done': True, 'metadata': {'model_used': settings.SIMPLE_MODEL, 'classification': 'simple', 'tokens': {'input': 0, 'output': 0}, 'latency_ms': total_latency, 'chunks_retrieved': 0, 'evaluator_flags': []}, 'sources': [], 'conversation_id': conversation_id})}\n\n"

        return StreamingResponse(greeting_stream(), media_type="text/event-stream")

    # 3. Process through pipeline
    pipeline = await _process_query_pipeline(question, conversation_id)

    # 4. Stream generator
    async def event_stream():
        nonlocal pipeline
        model = pipeline["model"]
        full_answer = []

        try:
            # Stream from Groq
            prompt_tokens = 0
            completion_tokens = 0
            llm_latency = 0

            try:
                stream = groq_client.generate_stream(
                    messages=pipeline["messages"],
                    model=model,
                    max_tokens=pipeline["max_tokens"],
                )
            except Exception as e:
                # Fallback: if 70B fails, try 8B
                if pipeline["classification"] == "complex":
                    print(f"[Stream] 70B failed ({e}), falling back to 8B")
                    model = settings.SIMPLE_MODEL
                    stream = groq_client.generate_stream(
                        messages=pipeline["messages"],
                        model=model,
                        max_tokens=pipeline["max_tokens"],
                    )
                else:
                    yield f"data: {json.dumps({'error': f'LLM service unavailable: {str(e)}'})}\n\n"
                    return

            for event in stream:
                if "token" in event:
                    # Stream token to client
                    full_answer.append(event["token"])
                    yield f"data: {json.dumps({'token': event['token']})}\n\n"

                elif event.get("done"):
                    # Capture final usage stats from stream
                    prompt_tokens = event.get("prompt_tokens", 0)
                    completion_tokens = event.get("completion_tokens", 0)
                    llm_latency = event.get("latency_ms", 0)

            # Assemble full answer
            answer = "".join(full_answer)

            # Run evaluator on complete answer
            evaluator_flags = evaluate(
                answer=answer,
                chunks_retrieved=pipeline["chunks_retrieved"],
                retrieved_chunks=pipeline["retrieved_chunks"],
            )

            # Total latency
            total_latency = int((time.perf_counter() - request_start) * 1000)

            # Build sources
            sources = [
                {
                    "document": chunk["document"],
                    "page": chunk.get("page"),
                    "relevance_score": round(score, 4),
                }
                for chunk, score in pipeline["results"]
            ]

            # Store turn in conversation memory
            memory.add_turn(conversation_id, question, answer)

            # Log
            log_query(
                query=question,
                classification=pipeline["classification"],
                model_used=model,
                complexity_score=pipeline["complexity_score"],
                signals=pipeline["signals"],
                tokens_input=prompt_tokens,
                tokens_output=completion_tokens,
                latency_ms=total_latency,
                conversation_id=conversation_id,
                evaluator_flags=evaluator_flags,
                chunks_retrieved=pipeline["chunks_retrieved"],
            )

            # Final done event with metadata
            yield f"data: {json.dumps({'done': True, 'metadata': {'model_used': model, 'classification': pipeline['classification'], 'tokens': {'input': prompt_tokens, 'output': completion_tokens}, 'latency_ms': total_latency, 'chunks_retrieved': pipeline['chunks_retrieved'], 'evaluator_flags': evaluator_flags}, 'sources': sources, 'conversation_id': conversation_id})}\n\n"

        except Exception as e:
            print(f"[Stream] Error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
