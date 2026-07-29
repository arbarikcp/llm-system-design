# llm_platform — the book's runnable project

A minimal but production-shaped **LLM serving + RAG platform**. It is the single
project that runs through the *LLM System Design* ebook; each chapter explains one
module.

| Module | Responsibility | Chapter |
|---|---|---|
| `estimation.py` | tokens/sec, KV-cache memory, cost & GPU sizing | 1 |
| `pipeline.py` | end-to-end request lifecycle (data plane) | 2 |
| `app.py` | FastAPI: sync, streaming, control-plane views | 2 |
| `router.py` | model routing + cheap-first cascade | 3 |
| `retrieval.py` / `embeddings.py` | RAG retrieval | Part IV |
| `cache.py` | exact + semantic response cache | Part IX |
| `guardrails.py` | input/output safety | Part VI |
| `eval.py` | offline eval harness | Part VIII |
| `training.py` | MFU, ZeRO memory, GPU-hours, all-reduce, pipeline bubble, LoRA, MinHash/LSH dedup, step health checks | 4–8 |
| `serving.py` | prefill/decode roofline, paged KV, speculative speedup, quant memory, continuous batching | 9–11 |
| `gateway.py` | token-bucket + token-budget rate limiting, priority queue | 12 |
| `chunking.py` | fixed / recursive / structure-aware chunking | 15 |
| `retrieval_hybrid.py` | BM25, reciprocal rank fusion, cross-encoder reranker, hybrid retriever | 14 |
| `rag_eval.py` | recall@k / precision@k / MRR / NDCG, faithfulness, hallucination rate | 16 |
| `rag.py` | offline `RAGIndex` + online `RAGQueryEngine` (retrieve → rerank → generate) | 13 |
| `tools.py` | tool registry + executor (validate, retry/backoff, cache) | 19 |
| `agents.py` | ReAct agent loop with pluggable policy | 17 |
| `multi_agent.py` | supervisor/worker routing + blackboard | 18 |
| `memory.py` | short-term / long-term / working memory + `AgentMemory` | 20 |
| `safety.py` | input/output safety pipelines, `SafetyMonitor`, `RedTeamSuite` | 21–23 |
| `context_assembler.py` | prompt templates/library + budget-aware context packing | 24 |
| `feedback.py` | feedback collection, curation, preference pairs, active learning | 25 |
| `sessions.py` | conversation store with branching + `SessionManager` (TTL, cross-device) | 26 |
| `metrics.py` | exact match, token F1, ROUGE-L, pass@k, LLM-as-judge proxies | 27 |
| `eval_offline.py` | versioned eval datasets, contamination check, runner, bootstrap/regression | 27 |
| `experiments.py` | variant assignment, A/B (z-test/Welch), shadow, canary controller | 28 |
| `observability.py` | latency percentiles, error/cost metrics, PSI drift, agent traces | 29 |
| `scaling.py` | queue-depth autoscaler, token-aware load balancer, pre-warm sizing | 30 |
| `reliability.py` | circuit breaker, retry/backoff, k-of-n availability, fallback chain | 31 |
| `caching.py` | LRU/TTL cache, embedding cache, prefix-cache savings, `MultiLevelCache` | 32 |
| `blueprints.py` | 8 reference designs composing the modules: `SearchEngine`, `CodingAssistant`, `ChatbotPlatform`, `DocumentQA`, `ContentModerationSystem`, `RecommendationSystem`, `AgentPlatform`, `MultiTenantPlatform` | 33–40 |
| `cost.py` | build-vs-buy break-even, TCO, cost-per-token, capacity-for-growth | 41–42 |
| `registry.py` | model registry: versions, lineage, promote/rollback | 43 |
| `ci_gate.py` | release gate: quality regression + red-team catch-rate floor | 45 |
| `compound.py` | DAG flow engine (nodes, deps, retry, conditional) | 46 |
| `reasoning.py` | best-of-N, self-consistency, thinking-budget allocation | 47 |
| `llm_gateway.py` | multi-provider gateway: abstraction, cost/latency routing, failover | 50 |

## Run it (no credentials, fully offline)

```bash
cd project
python run_demo.py          # end-to-end demo with telemetry
python -m pytest -q         # test suite
```

The core depends only on the Python standard library (numpy/FastAPI optional),
so it runs anywhere.

## Optional: HTTP API

```bash
pip install -r requirements.txt
uvicorn llm_platform.app:app --reload
# POST http://127.0.0.1:8000/v1/chat   {"query": "What is the KV cache?"}
```

## Optional: use a real model

Point the platform at any OpenAI-compatible endpoint (vLLM, TGI, or a vendor):

```bash
export OPENAI_BASE_URL="https://api.your-provider.com/v1"
export OPENAI_API_KEY="sk-..."
python run_demo.py
```

With no env vars set, a deterministic mock backend is used so everything is
reproducible.
