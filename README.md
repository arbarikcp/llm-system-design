# LLM System Design — Project & Lab Companion

A production-grade, interview-focused implementation of Large Language Model systems featuring:
1. **`project/` (`llm_platform`)**: A minimal, production-shaped LLM serving + RAG reference platform built in pure Python with zero heavy framework dependencies.
2. **`lab/` (`llm_lab`)**: An interactive hands-on Streamlit web application built with **LangChain**, backing local **Ollama** models (`llama3.1` + `nomic-embed-text`) with a deterministic offline fallback.

---

## 📁 Repository Layout

```
LLM-system-design/
├── project/                  # Reference LLM serving + RAG platform (pure Python)
│   ├── llm_platform/         # Core platform modules (pipeline, router, RAG, agents, etc.)
│   ├── tests/                # Offline unit test suite (150+ tests)
│   ├── run_demo.py           # End-to-end demo script with telemetry
│   └── README.md
└── lab/                      # Interactive web lab (LangChain + Streamlit)
    ├── llm_lab/              # Lab modules & LangChain implementations
    │   └── platform_bridge.py  # Bridges all reference modules to the UI
    ├── tests/                # Lab test suite
    ├── app.py                # Streamlit web dashboard
    └── README.md
```

---

## 🚀 Quick Start

### 1. Running the Reference Platform (`project/`)
*(Dependency-free, pure Python standard library; optional `numpy`/`FastAPI`)*

```bash
cd project
python run_demo.py          # Run end-to-end platform demo with telemetry
python -m pytest -q         # Run full offline test suite (150+ tests)
```

**Optional: Launch HTTP API Server**
```bash
pip install -r requirements.txt
uvicorn llm_platform.app:app --reload
# POST http://127.0.0.1:8000/v1/chat   {"query": "What is the KV cache?"}
```

**Optional: Connect to Real OpenAI-Compatible Endpoints (vLLM, TGI, Ollama, Vendor)**
```bash
export OPENAI_BASE_URL="https://api.your-provider.com/v1"
export OPENAI_API_KEY="sk-..."
python run_demo.py
```

---

### 2. Running the Interactive UI Lab (`lab/`)
*(LangChain + Streamlit UI with local Ollama models or offline fallback)*

```bash
cd lab
pip install -r requirements.txt

# (Optional) Real models via Ollama:
ollama pull llama3.1
ollama pull nomic-embed-text
ollama serve            # if not already running

# Generate seed datasets (also runnable from UI)
python -m llm_lab.seed

# Launch Streamlit Web UI
streamlit run app.py
```
Open `http://localhost:8501` in your browser to run interactive system demos and inspect execution traces.

**Running Lab Tests:**
```bash
LAB_BACKEND=offline python -m pytest -q      # Deterministic, no server required
```

---

## 🛠️ Reference Platform Architecture (`project/llm_platform`)

The reference platform (`llm_platform`) provides a complete, production-shaped LLM serving, retrieval, safety, evaluation, and orchestration stack written in clean Python.

### System Modules Breakdown

| Module | Responsibility | System Focus |
|---|---|---|
| `estimation.py` | Tokens/sec, KV-cache memory, cost & GPU sizing math | Capacity & Cost Engineering |
| `pipeline.py` | End-to-end request lifecycle (stateless data plane) | System Architecture |
| `app.py` | FastAPI surface: sync, streaming (`/v1/chat/stream`), control-plane | API Surface |
| `router.py` | Heuristic classifier + cheap-first cascade router | Model Strategy & Routing |
| `retrieval.py` / `embeddings.py` | Vector store and embedding generation | Vector Indexing |
| `cache.py` | Exact + semantic response caching | Response Caching |
| `guardrails.py` | Input injection screening & output safety filter | Safety & Moderation |
| `training.py` | MFU, ZeRO memory, GPU-hours, all-reduce, pipeline bubble, LoRA, MinHash dedup | Training Infrastructure |
| `serving.py` | Prefill/decode roofline, paged KV, speculative decoding, quant memory, continuous batching | Inference Engine |
| `gateway.py` | Token-bucket & token-budget rate limiting, priority queues | API Gateway |
| `chunking.py` | Fixed-size, recursive, and structure-aware document chunking | RAG Data Processing |
| `retrieval_hybrid.py` | BM25, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking | Hybrid Retrieval |
| `rag_eval.py` | Recall@k, Precision@k, MRR, NDCG, faithfulness, hallucination rate | RAG Quality & Metrics |
| `rag.py` | Offline `RAGIndex` + online `RAGQueryEngine` (retrieve → rerank → generate) | End-to-End RAG |
| `tools.py` | Tool registry, parameter validation, retries, sandboxing, tool caching | Tool Execution |
| `agents.py` | ReAct agent loop with pluggable policy & scratchpad | LLM Agent Architecture |
| `multi_agent.py` | Supervisor/worker routing, blackboard shared state | Multi-Agent Orchestration |
| `memory.py` | Short-term, long-term (vector recall), working memory + `AgentMemory` | Agent Memory Systems |
| `safety.py` | Multi-stage safety pipelines, `SafetyMonitor`, `RedTeamSuite` | Safety Operations |
| `context_assembler.py` | Prompt templates, token budget-aware context packing | Context Assembly |
| `feedback.py` | Feedback collection, curation, preference pairs, active learning | Data Flywheel |
| `sessions.py` | Conversation store with branching & `SessionManager` (TTL, cross-device) | Session State |
| `metrics.py` | Exact match, token F1, ROUGE-L, pass@k, LLM-as-judge proxies | Offline Metrics |
| `eval_offline.py` | Versioned eval datasets, contamination check, bootstrap regression | Offline Evaluation |
| `experiments.py` | Deterministic assignment, A/B (z-test/Welch), shadow, canary controller | Online Experimentation |
| `observability.py` | Latency percentiles, error/cost metrics, PSI drift, agent tracing | Monitoring & Observability |
| `scaling.py` | Queue-depth autoscaler, token-aware load balancer, pre-warm sizing | Scalability & Autoscaling |
| `reliability.py` | Circuit breaker, retry/backoff, k-of-n availability, fallback chains | Reliability Engineering |
| `caching.py` | LRU/TTL cache, embedding cache, prefix-cache savings, `MultiLevelCache` | Multi-Level Caching |
| `blueprints.py` | 8 reference designs: `SearchEngine`, `CodingAssistant`, `ChatbotPlatform`, `DocumentQA`, `ContentModerationSystem`, `RecommendationSystem`, `AgentPlatform`, `MultiTenantPlatform` | System Blueprints |
| `cost.py` | Build-vs-buy break-even, TCO, cost-per-token, capacity planning | Cost Optimization |
| `registry.py` | Model registry: versions, lineage, promote/rollback | Model Governance |
| `ci_gate.py` | Release gate: quality regression + red-team catch-rate floor | CI/CD Quality Gates |
| `compound.py` | Compound AI systems: DAG flow engine (nodes, deps, retry, conditional) | Compound Systems |
| `reasoning.py` | Test-time compute: Best-of-N, self-consistency, thinking budgets | Reasoning Systems |
| `llm_gateway.py` | Multi-provider gateway: abstraction, cost/latency routing, failover | Multi-Provider Gateway |

---

## 🧪 Interactive Web Lab (`lab/llm_lab`)

The interactive lab (`llm_lab`) layers **LangChain** and a **Streamlit UI** on top of the reference architecture so you can visually test and trace every system.

### Interactive Pages (Streamlit UI)

| Page (UI) | System | Implementation Module |
|---|---|---|
| 🔀 Model Routing & Cascade | LCEL routing by prompt difficulty + telemetry | `routing.py` |
| 📚 RAG | LangChain vector store, hybrid retrieval, cited answers | `rag.py` |
| 🤖 Agent & Tools | ReAct loop over LangChain tools (sandboxed) | `agents.py` |
| 🛡️ Guardrails & Red-Team | Input/output screening + catch-rate evaluation | `guardrails.py` |
| ⚡ Semantic Cache | Embedding-similarity response cache | `caching.py` |
| 🎯 Recommendation | Collaborative filtering + cold-start profile matching | `recommend.py` |
| 📊 Evaluation Harness | RAG eval + LLM-as-judge scoring, pass-rate & cost | `evaluation.py` |
| 🧰 Platform Modules | Auto-generated UI form for **every callable** in `llm_platform` | `platform_bridge.py` |
| 🌱 Seed Data | Dataset generation & inspection tool | `seed.py` |
| 🧾 Source Code Browser | View any Python module source directly in the UI | — |
| ✅ Run Tests | Execute the offline `pytest` suite from the UI | `tests/` |

### Backends & Selection Logic (`config.py`)

- **`ollama`**: Uses `ChatOllama(llama3.1)` and `OllamaEmbeddings(nomic-embed-text)` running locally on your machine. Set `LAB_BACKEND=ollama` or pick it in the UI sidebar.
- **`offline`**: Uses a deterministic `OfflineChatModel` + `HashEmbeddings` (no server, no GPU, no API keys needed).
- **`auto`** (default): Automatically checks if Ollama is reachable; uses Ollama if running, otherwise falls back to `offline`.

### Non-Functional Requirements (NFR) Telemetry
- **Latency Tracking**: Reports per-request latency (ms) for LCEL chains and agent execution steps.
- **Cost & Throughput**: Displays token estimates and estimated cost per query.
- **Quality & Safety**: Reports faithfulness scores, LLM-as-judge pass rates, and red-team catch rates.
# llm-system-design
