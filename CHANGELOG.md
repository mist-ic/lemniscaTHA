# Changelog

All notable optimizations and improvements to the ClearPath RAG system.

---

## v3.1.0 — CI/CD Pipeline

- Added GitHub Actions workflow with Docker Buildx caching
- Post-deploy eval harness as quality gate (32/32 tests must pass)
- Path filtering: only triggers on `backend/`, `frontend/`, or `Dockerfile` changes

## v3.0.0 — ONNX Runtime Migration

**The big optimization.** Replaced PyTorch/sentence-transformers with ONNX Runtime.

| Metric | Before | After | Change |
|---|---|---|---|
| Docker image | 2.01 GB | **666 MB** | −67% |
| Cold start | 52.4s | **623ms** | −98.8% |
| API latency (avg) | 652ms | **395ms** | −39% |
| pip install size | ~536 MB | ~148 MB | −72% |

- Implemented manual mean pooling + L2 normalization
- Applied INT8 quantization (127MB → 32MB model, <1% accuracy loss)
- Removed PyTorch, CUDA, and sentence-transformers dependencies entirely

## v2.0.0 — BGE-small Embedding Model

Switched from all-MiniLM-L6-v2 to BGE-small-en-v1.5 after benchmarking 7 models.

- **Retrieval quality**: 0.54 → **0.73 avg** (+30%)
- 5 tests improved, 0 regressions (32/32 still passing)
- Same 384 dimensions — zero retriever code changes
- Biggest win: conversation follow-ups improved from 0.50 → 1.00 relevancy

## v1.2.0 — Conversation Memory

- Follow-up detection: pronouns, short queries, 16 referring phrases
- Query rewriting via 8B model: retrieval score 0.46 → 0.67 (+45%)
- 7 multi-turn conversation tests added to eval harness (7/7 passing)

## v1.1.0 — Chunking Optimization

Three iterations to find optimal chunk boundaries:

| Version | Chunks | Avg Tokens | Issue |
|---|---|---|---|
| v1 | 376 | ~95 | Over-fragmented, tiny chunks |
| v2 | 164 | ~135 | Still split some related content |
| **v3** | **93** | **179** | **Optimal — clean semantic boundaries** |

- Added FAQ Q&A pair preservation (6 pairs kept atomic)
- Added table-aware merging (pricing tables → 500 token limit)
- Added post-merge pass for <80 token fragments

## v1.0.0 — Initial Implementation

- RAG pipeline: PDF extraction → structure-aware chunking → embedding → retrieval
- Model router: 7-signal deterministic weighted scorer
- Output evaluator: 3 flags (no_context, refusal, conflicting_sources)
- Prompt injection defense: salted XML tags (8/8 attacks blocked)
- Streaming: Token-by-token SSE with stop button and fallback
- Frontend: React 18 + TypeScript + Tailwind + shadcn/ui
- Eval harness: 32 tests across 3 suites
- Deployment: GCP Cloud Run (asia-south1)
