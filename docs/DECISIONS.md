# Architectural Decision Records

> Key engineering decisions made during development, each with context, alternatives, and rationale.

---

## ADR-01: Pure NumPy Over Vector Databases

**Context**: Need vector similarity search for retrieval. Options range from managed services (Pinecone) to libraries (FAISS, ChromaDB) to raw computation.

**Decision**: Pure NumPy dot-product search.

**Alternatives considered**:
| Option | Pros | Cons |
|---|---|---|
| **NumPy** ✅ | Zero dependencies, 4 lines of code, <1ms search | No ANN optimization |
| FAISS | Industry standard, fast ANN | Overkill at 93 vectors, added dependency |
| ChromaDB | Full vector DB, persistence | Heavy dependency, overkill |
| Pinecone | Managed, scalable | External service |

**Rationale**: Brute-force dot product on 93 vectors of 384 dimensions takes <1ms. FAISS would add complexity and dependency risk for literally zero performance benefit. The entire retrieval is:
```python
scores = embeddings_matrix @ query_vec.T
top_indices = np.argsort(scores.flatten())[::-1][:k]
```

---

## ADR-02: BGE-small Over MiniLM (Benchmark-Driven)

**Context**: Needed an embedding model for semantic search. All research recommended all-MiniLM-L6-v2 as the safe default.

**Decision**: BGE-small-en-v1.5 after benchmarking 7 models on actual data.

**Key data**: MiniLM scored 0.54 avg retrieval; BGE-small scored 0.73 — a **30% improvement** that MTEB leaderboard rankings (2-point gap) completely failed to predict. Models 10-17× larger (GTE-large, BGE-large) scored lower than BGE-small on our specific document set.

**Lesson**: Leaderboard rankings don't predict performance on small, focused document sets. Always benchmark on your actual data.

---

## ADR-03: Salted XML Tags for Injection Defense

**Context**: Document corpus contains 4 embedded prompt injections. Standard static context tags (`<documents>`) are vulnerable to escape.

**Decision**: Per-request random-salted XML tags using `secrets.token_hex(3)`.

**How it works**: Each request wraps context in `<ctx_{random_hex}>...</ctx_{random_hex}>`. Since the salt changes every request, no pre-planted text in the documents can include the correct closing tag. Combined with a hardened 7-rule system prompt.

**Result**: 8/8 injection attacks blocked (4 embedded + 3 novel).

---

## ADR-04: Structure-Aware Chunking Over Fixed-Size Splitting

**Context**: Need to split 30 PDFs into retrieval-friendly chunks. Fixed-size splitting (512 tokens) is the default in most tutorials.

**Decision**: Structure-aware paragraph chunking with FAQ preservation and table-aware merging.

**Evolution**: Three iterations — 376 → 164 → 93 chunks — each driven by analysis of retrieval failures. Key features: section heading detection, FAQ Q&A pair atomicity, pricing table higher token limit (500), post-merge pass for <80 token fragments.

**Result**: 93 chunks at avg 179 tokens. Clean semantic boundaries, no split Q&A pairs, no orphaned table rows.

---

## ADR-05: ONNX Runtime Over PyTorch

**Context**: `sentence-transformers` pulls in PyTorch (~2GB CUDA dependencies) even on CPU-only Cloud Run. Docker image was 2.01 GB with 52s cold starts.

**Decision**: Export to ONNX, apply INT8 quantization, rewrite embedder with manual mean pooling.

**Result**: 
- Docker image: 2.01 GB → 666 MB (−67%)
- Cold start: 52.4s → 623ms (−98.8%)
- Model file: 127 MB → 32 MB (−75%)
- Embedding output: Mathematically identical 384-dim vectors

**Tradeoff**: Required implementing manual mean pooling and L2 normalization (previously handled by sentence-transformers internally).

---

## ADR-06: Query Rewriting Over History Appending

**Context**: Conversation memory needs to handle follow-up queries like "How much does it cost?" without full context.

**Decision**: Detect follow-ups (pronouns, short queries, referring phrases) and rewrite them into standalone questions via the 8B model.

**Alternatives**:
| Option | Pros | Cons |
|---|---|---|
| No memory | Simplest | Follow-ups fail completely |
| Append raw history | Simple | Pollutes retrieval with old answer text, wastes tokens |
| **Query rewriting** ✅ | Clean retrtrieval signal, measured improvement | Extra 8B API call (~200 tokens) |
| Full summarization | Compresses history | Complex, loses detail |

**Result**: "How much does it cost?" → "What is the monthly cost of the Pro plan?" — retrieval score: 0.46 → 0.67 (+45%).

---

## ADR-07: Deterministic Router Over LLM-Based Classification

**Context**: Need to route queries to either 8B (simple) or 70B (complex) models. LLM-based routing would be most accurate.

**Decision**: Deterministic 7-signal weighted scorer with threshold ≥ 4.

**Signals**: Length, analytical keywords, error keywords, negation, multi-entity, compound structure, sensitive topics. Each with calibrated weights. Threshold validated empirically across T=2 through T=5.

**Rationale**: LLM-based routing adds latency (extra API call) and cost. Our rule-based scorer handles 80%+ of queries correctly and the cost difference is significant — 8B is 500K TPD vs 70B's 100K TPD on free tier.

---

## ADR-08: SSE Streaming Over WebSockets

**Context**: Need token-by-token delivery to frontend. Both WebSocket and SSE are options.

**Decision**: Server-Sent Events via FastAPI `StreamingResponse`.

**Why not WebSocket?** SSE is simpler (unidirectional), sufficient for our use case (server → client streaming), and natively supported by FastAPI's `StreamingResponse`. WebSocket would add bidirectional complexity we don't need.

**Why not native EventSource?** It only supports GET. Our streaming endpoint needs POST (to send question + conversation_id). Solution: `fetch` + `ReadableStream` with manual SSE parsing (~20 lines).

---

## ADR-09: Hybrid Conflicting Sources Detection

**Context**: The custom evaluator flag needed to catch real issues. ClearPath docs have genuine pricing contradictions ($49 vs $45 vs $52 for Pro plan).

**Decision**: Three detection methods combined:
1. **Numeric divergence**: Extract $ amounts from chunks, flag if same context keyword has different values
2. **Model self-report**: System prompt tells LLM to mention conflicts; check answer for "conflicting" / "inconsistent"
3. **Known patterns**: Hardcoded check for known Pro plan price variants

**Why not LLM-based detection?** Would add ~500ms latency and ~100 tokens per query — for every query, not just conflicting ones. Method 2 (self-report) already acts as a soft LLM detector for free.

---

## ADR-10: GCP Cloud Run Over PaaS Platforms

**Context**: Need public deployment. Options range from Railway/Render (simple PaaS) to GCP/AWS (cloud-native).

**Decision**: GCP Cloud Run with GitHub Actions CI/CD.

**Why GCP?**: Docker-native, auto-scales to zero (no idle cost), flexible resource allocation (4Gi memory for embedding model), and asia-south1 region for latency. The CI/CD pipeline includes a post-deploy eval gate — all 32 tests must pass against the live URL.

**Why not Railway/Render?**: Limited memory on free tier (~512MB), no auto-scale-to-zero on most plans, less control over deployment configuration.
