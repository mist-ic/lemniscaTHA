# Engineering Notes

> Technical analysis of routing logic, retrieval challenges, cost projections, and known limitations.

---

## Routing Logic

The model router uses a deterministic 7-signal weighted scorer. Each query accumulates a complexity score from: query length over 15/25 words (+1/+2), analytical keywords like "how", "why", "compare", "troubleshoot" (+2), error keywords like "not working", "broken", "crash" (+1), negation words (+1), multiple named entities (+2), compound structure like multiple question marks or semicolons (+1), and sensitive topics like "pricing", "security", "compliance" (+1). Score 4 or above routes to `llama-3.3-70b-versatile`, below that to `llama-3.1-8b-instant`. Greetings are caught before the scorer and return a canned response with zero API calls.

Thresholds 2-5 were tested against 30 queries: T=3 over-routed simple price lookups to 70B, T=5 missed genuine comparisons. T=4 sent only 17.2% of traffic to 70B, requiring at least two independent signal groups to fire before escalating.

**Known misclassification**: "My timeline view isn't loading after upgrading" scores only 2 (error +1, negation +1) and routes to 8B, but it actually requires cross-referencing release notes, the troubleshooting guide, and upgrade docs — a task better suited for 70B. The 8B model still produces a usable answer, just less thorough.

**Improvement path**: Add a retrieval-aware escalation signal — if top-k chunks span 3+ distinct source documents after retrieval, auto-escalate to complex. Still deterministic but informed by document structure rather than surface-level keywords. Alternatively, train a lightweight logistic regression classifier on ~200 labelled queries using the 7 existing lexical signals plus retrieval dispersion and similarity variance.

---

## Retrieval Challenges

**Table extraction**: "What keyboard shortcuts are available in ClearPath?" correctly identifies the shortcuts document as the top result (score 0.66), but the actual shortcuts table ranked fourth or fifth because PyMuPDF extracts tables as sequences of short decontextualised lines ("C", "Create new task") that don't form semantically rich embeddings. When merged during chunking, these short tokens dilute the paragraph-level embedding.

**Fix**: Table-aware chunking that serialises rows as "Shortcut: C — Action: Create new task" before embedding, and metadata-based re-ranking that boosts table-derived chunks when the query implies a catalogue lookup.

**Conflicting sources**: The document corpus has a real pricing conflict — Pro plan is $49/month in the Pricing Sheet, $45/month in Enterprise Plan Details, and $52/month in the Onboarding Checklist. The `conflicting_sources` evaluator flag catches this via three methods: model self-reporting conflict keywords in its answer (free — piggybacks on the generation call), numeric divergence detection across chunks from different documents, and a hardcoded check for known Pro plan price variants.

---

## Cost and Scale Analysis

### Current Cost: $0

The system runs on Groq's free tier. No billing, no charges. The free tier provides 500K TPD on 8B and 100K TPD on 70B at zero cost.

### Projected at 5,000 Queries/Day

| Component | Queries/Day | Avg Tokens | Daily Tokens | Est. Cost |
|---|---|---|---|---|
| 8B (simple) | 4,150 (83%) | ~700 | 2.91M | ~$0.16 |
| 8B (rewrites) | ~500 | ~200 | 0.10M | ~$0.01 |
| 70B (complex) | 850 (17%) | ~2,900 | 2.47M | ~$1.72 |
| **Total** | **5,000** | | **5.48M** | **~$1.89/day ($57/month)** |

The biggest cost driver is 70B input tokens — ~2,000 tokens of context per complex query (5 chunks × ~400 tokens). Even at only 17% of traffic, complex queries consume ~45% of token spend.

### Highest-ROI Optimization

Semantic caching: hash the query embedding and cache responses. Support queries follow a Zipf distribution — a 30% cache hit rate eliminates ~1,500 LLM calls/day, cutting cost to ~$1.32/day and keeping the 70B model within its 1,000 RPD free-tier limit.

### Optimization to Avoid

Lowering the router threshold to T=5 (only 6.9% on 70B). Testing showed this misclassifies genuine comparison queries to 8B, degrading answer quality to save ~$0.40/day. The real free-tier bottleneck is 70B RPD (1,000 requests/day), not cost.

### Production Tier Analysis

With a Groq Dev tier account, the calculus changes fundamentally:
- 70B gets **500K RPD** (vs 1K free) and **1K RPM** (vs 30 free)
- Rate limits are no longer a factor — the constraint shifts to pure cost optimization
- Router threshold could relax from ≥4 to ≥3 for better answer quality
- Semantic cache becomes optional (nice-to-have, not survival requirement)
- Can handle ~16 concurrent requests/second vs free tier's ~2/second

---

## Known Limitations

### 1. Nuanced Intent Routing
The rule-based router cannot detect semantic complexity. "What's happening with Timeline view?" (5 words, score 1) routes to 8B, but answering well requires synthesising across three contradicting documents: Advanced Features Overview ("Coming Q2 2023"), Q4 Retro ("Shipped October 2024"), and Release Notes ("v3.1.0 — August 2024"). A keyword scanner will systematically underserve queries whose complexity lies in the relationship between retrieved documents rather than the structure of the question.

### 2. In-Memory Conversation State
Conversations are lost on restart. Production deployment needs Redis or a database-backed session store.

### 3. Groq Free-Tier Constraints
70B is capped at 1,000 RPD / 30 RPM. The built-in 70B→8B fallback with exponential backoff handles spikes, but sustained load requires a paid tier.

### 4. Table Extraction Fidelity
Complex PDF tables with merged cells don't chunk optimally due to PyMuPDF's text-based extraction. Pdfplumber would improve this at the cost of extraction speed.

---

## Key Engineering Decisions

### ONNX Runtime Migration
Sentence-transformers pulls in PyTorch (~2GB of CUDA libraries) even on CPU-only Cloud Run. Replaced with ONNX Runtime (~50MB). Docker image dropped from 2.01GB to 666MB, cold start from 52s to 623ms — a 98.8% reduction. Required manually implementing mean pooling and L2 normalisation since sentence-transformers was removed entirely.

### Embedding Model Swap
Research recommended all-MiniLM-L6-v2 as the safe default. MTEB benchmarks showed BGE-small was only 2 points higher — seemingly negligible. Benchmarking 7 candidates on actual 93 chunks revealed the truth: BGE-small scored 0.73 avg retrieval vs MiniLM's 0.54 — a 30% improvement that MTEB didn't predict. Applied INT8 quantization on top: model file dropped from 127MB to 32MB with less than 1% accuracy loss.

### Salted XML Injection Defense
Most RAG systems use static context tags like `<documents>`. Pre-planted injections can include `</documents>` to escape. Using `secrets.token_hex(3)` generates a random 6-char hex salt per request — tags become `<ctx_a7f3b2>`. Since the salt changes every request, no pre-planted text can guess the closing tag. Combined with a hardened system prompt listing untrusted data handling rules.

### Query Rewriting Over History Appending
Most conversation-memory implementations append raw history to the prompt. This pollutes the embedding space with old answer text and wastes tokens. Instead, follow-ups are detected (pronouns, short queries, referring phrases) and rewritten into standalone questions via the 8B model. Cost: ~200 tokens. Benefit: retrieval score 0.46 → 0.67.

### Greeter Bypass
"Hello" and "Hi" skip the LLM entirely — zero tokens, zero API calls, instant response. Simple but effective token optimisation that most implementations miss.

### SSE Streaming with Fallback
Token-by-token streaming via POST (not GET, so native EventSource doesn't work — manual ReadableStream parsing). Stop button using AbortController. If streaming fails, automatic fallback to non-streaming `/query` endpoint. Includes SSE buffer fix for when a ReadableStream `read()` splits a `data:` line across two chunks.

### CI/CD with Eval Gate
GitHub Actions pipeline triggers on push to main (only on backend/frontend/Dockerfile changes). Builds Docker image, deploys to Cloud Run, then runs the 32-test eval harness against the live URL. If any test fails, the deploy fails. Path filtering and Docker layer caching keep builds fast.
