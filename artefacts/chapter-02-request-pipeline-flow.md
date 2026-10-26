# LLMPipeline Request Lifecycle Flow

## High-Level Pipeline Overview

```mermaid
sequenceDiagram
    participant Client
    participant Pipeline as LLMPipeline

    Client->>Pipeline: handle(ChatRequest)

    rect rgb(254, 226, 226)
        Note right of Pipeline: 1. Input Safety
        Pipeline->>Pipeline: InputGuard.check(query)
        alt blocked
            Pipeline-->>Client: "Request blocked"
        end
    end

    rect rgb(220, 252, 231)
        Note right of Pipeline: 2. Cache Lookup
        Pipeline->>Pipeline: ResponseCache.get(query)
        alt cache hit
            Pipeline-->>Client: cached response
        end
    end

    rect rgb(243, 232, 255)
        Note right of Pipeline: 3. RAG Retrieval (if enabled)
        Pipeline->>Pipeline: Retriever.retrieve(query, k=4)
    end

    rect rgb(254, 243, 199)
        Note right of Pipeline: 4-5. Route + Generate
        Pipeline->>Pipeline: CascadeRouter.run(prompt)
        Note right of Pipeline: escalate if low confidence
    end

    rect rgb(254, 226, 226)
        Note right of Pipeline: 6. Output Safety
        Pipeline->>Pipeline: OutputGuard.check(response)
        alt blocked
            Pipeline-->>Client: "Response withheld"
        end
    end

    rect rgb(224, 242, 254)
        Note right of Pipeline: 7. Telemetry + Cache Write
        Pipeline->>Pipeline: record metrics + cache.put()
    end

    Pipeline-->>Client: ChatResponse(answer, citations, telemetry)
```

---

## Phase Diagrams

### Phase 1 — Input Safety (Red)

```mermaid
sequenceDiagram
    box rgb(219, 234, 254) Orchestrator
        participant Pipeline as LLMPipeline
    end
    box rgb(254, 226, 226) Safety
        participant IG as InputGuard
    end

    Pipeline->>IG: check(query)

    rect rgb(254, 226, 226)
        IG->>IG: len(text) > 24,000 chars?
        alt too long
            IG-->>Pipeline: GuardResult(allowed=False, reason="input_too_long")
        end

        IG->>IG: regex scan for injection patterns
        Note right of IG: "ignore all instructions",<br/>"reveal system prompt", etc.
        alt injection detected
            IG-->>Pipeline: GuardResult(allowed=False, reason="prompt_injection")
        end

        IG->>IG: scrub_pii(text)
        Note right of IG: Replace SSN, email, phone<br/>with [REDACTED_*] tokens
    end

    IG-->>Pipeline: GuardResult(allowed=True, text=cleaned, redactions=N)
```

### Phase 2 — Cache Lookup (Green)

```mermaid
sequenceDiagram
    box rgb(219, 234, 254) Orchestrator
        participant Pipeline as LLMPipeline
    end
    box rgb(220, 252, 231) Caching
        participant Cache as ResponseCache
        participant Emb as HashingEmbedder
    end

    Pipeline->>Cache: get(clean_query)

    rect rgb(220, 252, 231)
        Note over Cache: Tier 1: Exact Match
        Cache->>Cache: normalize(query) → SHA-256 hash
        Cache->>Cache: lookup hash in _exact dict
        alt exact hit + not expired
            Cache-->>Pipeline: ("exact", answer, citations)
        end

        Note over Cache,Emb: Tier 2: Semantic Match
        Cache->>Emb: embed(query)
        Emb-->>Cache: vector[256]
        Cache->>Cache: linear scan all live entries
        Cache->>Cache: cosine_similarity(query_vec, entry_vec)
        alt best score >= 0.92
            Cache-->>Pipeline: ("semantic", answer, citations)
        end
    end

    Cache-->>Pipeline: None (cache miss)
```

### Phase 3 — RAG Retrieval (Purple)

```mermaid
sequenceDiagram
    box rgb(219, 234, 254) Orchestrator
        participant Pipeline as LLMPipeline
    end
    box rgb(243, 232, 255) Retrieval
        participant Ret as Retriever
        participant VS as InMemoryVectorStore
    end
    box rgb(220, 252, 231) Embeddings
        participant Emb as HashingEmbedder
    end

    alt use_rag = true
        Pipeline->>Ret: retrieve(clean_query, k=4)

        rect rgb(243, 232, 255)
            Ret->>VS: search(query, k=4, min_score=0.15)
            VS->>Emb: embed(query)
            Emb-->>VS: query vector[256]
            VS->>VS: cosine_similarity vs all stored docs
            VS->>VS: sort descending, filter score >= 0.15
            VS->>VS: take top-k results
            VS-->>Ret: List[ScoredDoc]
        end

        Ret->>Ret: format context: "[doc_id] text" blocks
        Ret->>Ret: extract citation sources
        Ret-->>Pipeline: (context, citations, hits)
    else use_rag = false
        Note over Pipeline: skip retrieval, context = ""
    end
```

### Phase 4–5 — Routing, Cascade & Generation (Amber)

```mermaid
sequenceDiagram
    box rgb(219, 234, 254) Orchestrator
        participant Pipeline as LLMPipeline
    end
    box rgb(254, 243, 199) Routing & Inference
        participant CR as CascadeRouter
        participant MR as ModelRouter
        participant BE as InferenceBackend
    end

    Pipeline->>Pipeline: _build_prompt(query, context)
    Pipeline->>CR: run(prompt, query, context_tokens, max_tokens, temp)

    rect rgb(254, 243, 199)
        Note over MR: Step A: Route to a model tier
        CR->>MR: route(query, context_tokens, force_model)
        alt force_model is set
            MR-->>CR: RouteDecision(forced_model, "forced")
        else tokens > 80% of default ctx_window
            MR-->>CR: RouteDecision(large-flagship, "long_context")
        else hard intent regex matches
            MR-->>CR: RouteDecision(mid-balanced, "reasoning_intent")
        else simple intent + short query
            MR-->>CR: RouteDecision(small-fast, "simple_intent")
        else default
            MR-->>CR: RouteDecision(mid-balanced, "default")
        end
    end

    rect rgb(255, 251, 235)
        Note over CR,BE: Step B: Generate with chosen model
        CR->>BE: generate(prompt, model, max_tokens, temp)
        BE-->>CR: Completion(text, confidence, tokens)
    end

    rect rgb(254, 243, 199)
        Note over CR,BE: Step C: Cascade escalation
        alt confidence < 0.55 AND tier != "large" AND !force_model
            Note over CR: Low confidence → escalate
            CR->>BE: generate(prompt, large-flagship, ...)
            BE-->>CR: Completion (stronger model)
            CR->>CR: reason += "→escalated(low_confidence)"
        end
    end

    CR-->>Pipeline: (Completion, reason, escalated)
```

### Phase 6 — Output Safety (Red)

```mermaid
sequenceDiagram
    box rgb(219, 234, 254) Orchestrator
        participant Pipeline as LLMPipeline
    end
    box rgb(254, 226, 226) Safety
        participant OG as OutputGuard
    end

    Pipeline->>OG: check(text, require_grounding, has_context)

    rect rgb(254, 226, 226)
        OG->>OG: scan for banned phrases
        alt unsafe content found
            OG-->>Pipeline: GuardResult(allowed=False, reason="unsafe_output")
        end

        OG->>OG: scrub_pii(text)
        Note right of OG: Redact any PII leaked by the model

        alt require_grounding AND no context retrieved
            OG-->>Pipeline: GuardResult(allowed=True, text="I don't have enough<br/>grounded context...", reason="ungrounded_refusal")
        end
    end

    OG-->>Pipeline: GuardResult(allowed=True, text=cleaned)
```

### Phase 7 — Telemetry & Cache Write (Blue + Green)

```mermaid
sequenceDiagram
    box rgb(219, 234, 254) Orchestrator
        participant Pipeline as LLMPipeline
    end
    box rgb(224, 242, 254) Observability
        participant Tel as Telemetry
    end
    box rgb(220, 252, 231) Caching
        participant Cache as ResponseCache
        participant Emb as HashingEmbedder
    end

    rect rgb(224, 242, 254)
        Pipeline->>Tel: model, route_reason, escalated
        Pipeline->>Tel: input_tokens, output_tokens, confidence
        Pipeline->>Tel: cost_usd = ModelSpec.request_cost_usd(in, out)
        Pipeline->>Tel: latency_ms = (now - t0) * 1000
    end

    rect rgb(220, 252, 231)
        Pipeline->>Cache: put(clean_query, answer, citations)
        Cache->>Emb: embed(query)
        Emb-->>Cache: vector[256]
        Cache->>Cache: store _Entry(answer, citations, embedding, expires_at)
    end

    Pipeline-->>Pipeline: return ChatResponse(answer, citations, telemetry)
```

---

## Full Detailed Sequence Diagram (Reference)

```mermaid
sequenceDiagram
    box rgb(240, 240, 240) Client
        participant Client
    end
    box rgb(219, 234, 254) Orchestrator
        participant Pipeline as LLMPipeline
    end
    box rgb(254, 226, 226) Safety Layer
        participant IG as InputGuard
        participant OG as OutputGuard
    end
    box rgb(220, 252, 231) Caching & Embeddings
        participant Cache as ResponseCache
        participant Emb as HashingEmbedder
    end
    box rgb(243, 232, 255) RAG / Retrieval
        participant Ret as Retriever
        participant VS as InMemoryVectorStore
    end
    box rgb(254, 243, 199) Routing & Inference
        participant Est as estimate_tokens
        participant CR as CascadeRouter
        participant MR as ModelRouter
        participant BE as InferenceBackend
    end
    box rgb(224, 242, 254) Observability
        participant Tel as Telemetry
    end

    Client->>Pipeline: handle(ChatRequest)
    Note over Pipeline: t0 = perf_counter()

    %% Step 1: Input Guard
    rect rgb(254, 226, 226)
        Note over IG: Step 1: Input Safety
        Pipeline->>IG: check(query)
        IG->>IG: length check (max 24k chars)
        IG->>IG: injection pattern scan
        IG->>IG: scrub_pii (SSN, email, phone)
        IG-->>Pipeline: GuardResult(allowed, clean_text)
        alt blocked
            Pipeline-->>Client: ChatResponse("Request blocked by input policy.")
        end
    end

    %% Step 2: Cache Lookup
    rect rgb(220, 252, 231)
        Note over Cache,Emb: Step 2: Cache Lookup
        Pipeline->>Cache: get(clean_query)
        Cache->>Cache: prompt_hash(normalize(query))
        Cache->>Cache: exact lookup by SHA-256 hash
        alt exact miss
            Cache->>Emb: embed(query)
            Emb-->>Cache: vector[256]
            Cache->>Cache: linear scan + cosine_similarity >= 0.92
        end
        Cache-->>Pipeline: (kind, answer, citations) or None
        alt cache hit (exact or semantic)
            Pipeline-->>Client: ChatResponse(answer, citations, tel)
        end
    end

    %% Step 3: Retrieval (if use_rag)
    rect rgb(243, 232, 255)
        Note over Ret,VS: Step 3: RAG Retrieval
        alt use_rag = true
            Pipeline->>Ret: retrieve(clean_query, k=4)
            Ret->>VS: search(query, k=4, min_score=0.15)
            VS->>Emb: embed(query)
            Emb-->>VS: query vector[256]
            VS->>VS: cosine_similarity vs all docs
            VS->>VS: sort + filter by min_score
            VS-->>Ret: List[ScoredDoc]
            Ret->>Ret: build context blocks + citations
            Ret-->>Pipeline: (context, citations, hits)
        end
    end

    %% Step 4 & 5: Routing + Cascade + Generation
    rect rgb(254, 243, 199)
        Note over Est,BE: Step 4-5: Route + Generate
        Pipeline->>Pipeline: _build_prompt(query, context)
        Pipeline->>Est: estimate_tokens(context)
        Est-->>Pipeline: token count

        Pipeline->>CR: run(prompt, query, context_tokens, ...)
        CR->>MR: route(query, context_tokens, force_model)
        MR->>MR: check force_model override
        MR->>Est: estimate_tokens(query)
        MR->>MR: intent classification (hard/simple regex)
        MR->>MR: context window overflow check
        MR-->>CR: RouteDecision(model, reason)

        CR->>BE: generate(prompt, model, max_tokens, temperature)
        BE-->>CR: Completion(text, tokens, confidence, model)

        alt confidence < floor AND tier != "large" AND !force_model
            Note over CR: Escalate to large-flagship
            CR->>BE: generate(prompt, strong_model, ...)
            BE-->>CR: Completion (stronger model)
        end
        CR-->>Pipeline: (Completion, reason, escalated)
    end

    %% Step 6: Output Guard
    rect rgb(254, 226, 226)
        Note over OG: Step 6: Output Safety
        Pipeline->>OG: check(text, require_grounding, has_context)
        OG->>OG: banned phrase scan
        OG->>OG: scrub_pii on output
        OG->>OG: grounding check (refuse if RAG but no context)
        OG-->>Pipeline: GuardResult(allowed, clean_text)
        alt blocked
            Pipeline-->>Client: ChatResponse("Response withheld by output policy.")
        end
    end

    %% Step 7: Telemetry + Cache Write
    rect rgb(224, 242, 254)
        Note over Tel,Cache: Step 7: Telemetry + Cache Write
        Pipeline->>Tel: populate(model, cost, tokens, latency, ...)
        Pipeline->>Cache: put(clean_query, answer, citations)
        Cache->>Emb: embed(query)
        Emb-->>Cache: vector[256]
        Cache->>Cache: store _Entry with TTL
    end

    Pipeline-->>Client: ChatResponse(answer, citations, telemetry)
```

## Class Diagram

```mermaid
classDiagram
    class LLMPipeline {
        +Settings settings
        +InferenceBackend backend
        +Retriever retriever
        +ResponseCache cache
        +InputGuard input_guard
        +OutputGuard output_guard
        +CascadeRouter cascade
        +handle(ChatRequest) ChatResponse
        +stream(ChatRequest) Iterator~str~
        -_build_prompt(query, context) str
    }

    class ChatRequest {
        +str query
        +str user_id
        +str session_id
        +bool use_rag
        +bool stream
        +int max_tokens
        +float temperature
        +str force_model
    }

    class ChatResponse {
        +str answer
        +List~str~ citations
        +Telemetry telemetry
    }

    class Telemetry {
        +str model
        +str route_reason
        +str cache
        +int retrieved_docs
        +int input_tokens
        +int output_tokens
        +float cost_usd
        +float latency_ms
        +bool escalated
        +bool blocked
        +float confidence
    }

    class InputGuard {
        +int max_chars
        +check(text) GuardResult
    }

    class OutputGuard {
        +check(text, require_grounding, has_context) GuardResult
    }

    class GuardResult {
        +bool allowed
        +str text
        +str reason
        +int redactions
    }

    class ResponseCache {
        +int ttl_seconds
        +float semantic_threshold
        +HashingEmbedder embedder
        +get(query) Optional~Tuple~
        +put(query, answer, citations) None
    }

    class Retriever {
        +InMemoryVectorStore store
        +retrieve(query, k) Tuple
    }

    class InMemoryVectorStore {
        +HashingEmbedder embedder
        +add(Document) None
        +search(query, k, min_score) List~ScoredDoc~
    }

    class HashingEmbedder {
        +int dim
        +embed(text) List~float~
    }

    class CascadeRouter {
        +InferenceBackend backend
        +ModelRouter router
        +float confidence_floor
        +run(prompt, query, ...) Tuple
    }

    class ModelRouter {
        +Dict registry
        +route(query, context_tokens, force_model) RouteDecision
    }

    class RouteDecision {
        +ModelSpec model
        +str reason
    }

    class InferenceBackend {
        <<abstract>>
        +generate(prompt, model, ...) Completion
        +stream(prompt, model, ...) Iterator~str~
    }

    class MockBackend {
        +generate() Completion
        +stream() Iterator~str~
    }

    class OpenAICompatBackend {
        +str base_url
        +str api_key
        +generate() Completion
    }

    class Completion {
        +str text
        +int input_tokens
        +int output_tokens
        +float confidence
        +str model
    }

    class ModelSpec {
        +str name
        +str tier
        +float quality
        +int ctx_window
        +float usd_per_1k_input
        +float usd_per_1k_output
        +request_cost_usd(in, out) float
    }

    class Settings {
        +str default_model
        +str openai_base_url
        +int cache_ttl_seconds
        +float cascade_confidence_floor
        +use_real_backend() bool
    }

    LLMPipeline --> InputGuard
    LLMPipeline --> OutputGuard
    LLMPipeline --> ResponseCache
    LLMPipeline --> Retriever
    LLMPipeline --> CascadeRouter
    LLMPipeline --> InferenceBackend
    LLMPipeline --> Settings

    InputGuard ..> GuardResult
    OutputGuard ..> GuardResult

    CascadeRouter --> ModelRouter
    CascadeRouter --> InferenceBackend
    ModelRouter --> ModelSpec
    ModelRouter ..> RouteDecision

    Retriever --> InMemoryVectorStore
    InMemoryVectorStore --> HashingEmbedder
    ResponseCache --> HashingEmbedder

    InferenceBackend <|-- MockBackend
    InferenceBackend <|-- OpenAICompatBackend
    InferenceBackend ..> Completion

    LLMPipeline ..> ChatRequest : input
    LLMPipeline ..> ChatResponse : output
    ChatResponse --> Telemetry

    %% Orchestrator — blue
    style LLMPipeline fill:#DBEAFE,stroke:#3B82F6,stroke-width:2px,color:#1E3A5F

    %% Schemas / Data contracts — light gray
    style ChatRequest fill:#F3F4F6,stroke:#6B7280,stroke-width:1px,color:#1F2937
    style ChatResponse fill:#F3F4F6,stroke:#6B7280,stroke-width:1px,color:#1F2937
    style Telemetry fill:#E0F2FE,stroke:#0EA5E9,stroke-width:1px,color:#0C4A6E

    %% Safety layer — red/rose
    style InputGuard fill:#FEE2E2,stroke:#EF4444,stroke-width:2px,color:#7F1D1D
    style OutputGuard fill:#FEE2E2,stroke:#EF4444,stroke-width:2px,color:#7F1D1D
    style GuardResult fill:#FEE2E2,stroke:#F87171,stroke-width:1px,color:#7F1D1D

    %% Caching & embeddings — green
    style ResponseCache fill:#DCFCE7,stroke:#22C55E,stroke-width:2px,color:#14532D
    style HashingEmbedder fill:#DCFCE7,stroke:#22C55E,stroke-width:1px,color:#14532D

    %% RAG / Retrieval — purple
    style Retriever fill:#F3E8FF,stroke:#A855F7,stroke-width:2px,color:#3B0764
    style InMemoryVectorStore fill:#F3E8FF,stroke:#A855F7,stroke-width:1px,color:#3B0764

    %% Routing & inference — amber/yellow
    style CascadeRouter fill:#FEF3C7,stroke:#F59E0B,stroke-width:2px,color:#78350F
    style ModelRouter fill:#FEF3C7,stroke:#F59E0B,stroke-width:1px,color:#78350F
    style RouteDecision fill:#FEF3C7,stroke:#F59E0B,stroke-width:1px,color:#78350F
    style InferenceBackend fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#78350F
    style MockBackend fill:#FEF3C7,stroke:#D97706,stroke-width:1px,color:#78350F
    style OpenAICompatBackend fill:#FEF3C7,stroke:#D97706,stroke-width:1px,color:#78350F
    style Completion fill:#FEF3C7,stroke:#F59E0B,stroke-width:1px,color:#78350F

    %% Config — slate
    style ModelSpec fill:#F1F5F9,stroke:#64748B,stroke-width:1px,color:#1E293B
    style Settings fill:#F1F5F9,stroke:#64748B,stroke-width:1px,color:#1E293B
```

## Component Groups & Pipeline Roles

---

### 1. Orchestrator (Blue)

**Purpose:** Single entry point that wires every stage into the canonical request path and owns the request lifecycle clock.

- **`LLMPipeline`** — Receives a `ChatRequest`, executes steps 1–7 in sequence, and returns a `ChatResponse`. It is deliberately stateless: all state (cache entries, vector store, settings) lives in injected collaborators, so any replica can serve any request. Also exposes a `stream()` variant for low time-to-first-token UX.

**Where in the pipeline:** Wraps the entire flow — every other group is called *by* the pipeline.

---

### 2. Safety Layer (Red)

**Purpose:** Fast, deterministic guardrails on the hot path — the cheap first line of defence before and after the model call.

- **`InputGuard`** — Runs *before* anything else. Rejects oversized prompts (>24k chars), detects prompt-injection patterns via regex, and scrubs PII (SSN, email, phone) from the query. Returns a `GuardResult` with the cleaned text or a block reason.
- **`OutputGuard`** — Runs *after* generation. Scans the model's response for banned phrases, redacts any leaked PII, and enforces a grounding policy: if RAG was requested but no context was retrieved, it refuses to answer from parametric memory.
- **`GuardResult`** — Shared data contract carrying the allow/block decision, cleaned text, reason, and redaction count.

**Where in the pipeline:** Step 1 (input) and Step 6 (output). Both guards can short-circuit the pipeline and return early.

---

### 3. Caching & Embeddings (Green)

**Purpose:** Avoid redundant model calls by recognising repeated or semantically equivalent queries.

- **`ResponseCache`** — Two-tier lookup. *Tier 1 (exact):* SHA-256 hash of the normalised query, O(1) lookup with TTL expiry. *Tier 2 (semantic):* embeds the query and scans cached entries by cosine similarity (threshold ≥ 0.92). On a hit, the pipeline returns immediately — no retrieval, no generation.
- **`HashingEmbedder`** — Deterministic, dependency-free hashing vectoriser (dim=256) with L2 normalisation. Used by both the cache and the vector store. In production, swap for a real bi-encoder (e.g. SentenceTransformer) — the rest of the platform codes to the `embed()` interface.

**Where in the pipeline:** Step 2 (cache read before generation) and Step 7 (cache write after generation). The embedder is also used transitively by the retrieval group.

---

### 4. RAG / Retrieval (Purple)

**Purpose:** Ground the model's answer in external knowledge by retrieving relevant documents.

- **`Retriever`** — Thin façade that calls the vector store, assembles context blocks (formatted as `[doc_id] text`), and extracts citation sources. Returns a `(context, citations, hits)` tuple.
- **`InMemoryVectorStore`** — Stores `Document` objects with their embeddings. On `search()`, embeds the query, computes cosine similarity against all documents, filters by a relevance floor (min_score=0.15), and returns the top-k `ScoredDoc` results. The interface (`add`, `search`) maps directly to FAISS / pgvector in production.

**Where in the pipeline:** Step 3, only when `use_rag=True`. The retrieved context is injected into the prompt for grounded generation.

---

### 5. Routing & Inference (Amber)

**Purpose:** Pick the right model for each request and generate the answer — the core cost and quality lever.

- **`ModelRouter`** — Heuristic first-hop router. Checks for a forced model override, then uses token count and intent regex (hard vs. simple) to select from three tiers: `small-fast`, `mid-balanced`, or `large-flagship`. Returns a `RouteDecision(model, reason)`.
- **`CascadeRouter`** — Wraps `ModelRouter` with confidence-based escalation. Runs the routed (cheap) model first; if the `Completion.confidence` falls below the floor (default 0.55) and the model isn't already the strongest tier, it re-generates with `large-flagship`. This is the single highest-leverage cost optimisation.
- **`InferenceBackend`** (abstract) — Defines the `generate()` and `stream()` contract.
  - **`MockBackend`** — Deterministic, offline stand-in. Confidence is derived from whether retrieved context is present, which lets the cascade logic exercise without a real model.
  - **`OpenAICompatBackend`** — Calls any OpenAI-compatible `/chat/completions` endpoint (vLLM, TGI, vendor API). Extracts confidence from logprobs.
- **`Completion`** — Data object carrying the generated text, token counts, confidence score, and model name.
- **`RouteDecision`** — Data object pairing the chosen `ModelSpec` with the routing reason.
- **`estimate_tokens`** — Cheap, provider-agnostic token estimator (~4 chars/token). Used by the router for context-window overflow checks and by the pipeline for cost calculation.

**Where in the pipeline:** Steps 4–5. The router decides, the backend generates, and the cascade may re-generate with a stronger model.

---

### 6. Observability (Light Blue)

**Purpose:** Per-request telemetry record for monitoring, cost tracking, and debugging.

- **`Telemetry`** — Populated at the end of the pipeline with: model used, routing reason, cache hit type, retrieved doc count, token counts, USD cost, latency, escalation flag, and block status. Returned inside every `ChatResponse` so callers can log, alert, or display it.

**Where in the pipeline:** Step 7. Filled right before the final response is returned.

---

### 7. Config & Schemas (Gray)

**Purpose:** Global settings and data contracts that every other group depends on.

- **`Settings`** — Runtime configuration sourced from environment variables: default model, cache TTL, semantic threshold, cascade confidence floor, and optional OpenAI-compatible endpoint credentials. `use_real_backend()` switches between mock and live inference.
- **`ModelSpec`** — Frozen dataclass describing a servable model: name, tier, quality score, context window, pricing, latency, and throughput. The `MODEL_REGISTRY` dict is the single source of truth for routing and cost decisions.
- **`ChatRequest`** — Inbound contract: query text, user/session IDs, RAG toggle, streaming flag, generation params, and optional model override.
- **`ChatResponse`** — Outbound contract: answer text, citation list, and the `Telemetry` record.

**Where in the pipeline:** Injected at construction (`Settings`, `ModelSpec` via registry) and used as input/output boundaries (`ChatRequest` → `ChatResponse`).

---

### End-to-End Flow Summary

```
ChatRequest
  │
  ▼
┌─────────────── LLMPipeline ───────────────┐
│                                            │
│  1. InputGuard ──── block? → early return  │
│  2. ResponseCache ─ hit?  → early return   │
│  3. Retriever ───── (if RAG)               │
│  4. estimate_tokens + build prompt         │
│  5. CascadeRouter → ModelRouter            │
│        → InferenceBackend.generate()       │
│        → escalate if low confidence        │
│  6. OutputGuard ─── block? → early return  │
│  7. Telemetry + Cache write                │
│                                            │
└────────────────────────────────────────────┘
  │
  ▼
ChatResponse(answer, citations, telemetry)
```