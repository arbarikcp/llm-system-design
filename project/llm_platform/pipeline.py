"""End-to-end request lifecycle orchestration (Chapter 2).

LLMPipeline wires the modules into the canonical request path:

    input guard -> cache -> retrieve -> route+cascade -> generate
                -> output guard -> cache write -> telemetry

The pipeline is the *data plane*. It is deliberately stateless: session and
config state live in injected collaborators, so any replica can serve any
request (Chapter 2: stateless inference).
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

from .config import MODEL_REGISTRY, Settings
from .cache import ResponseCache
from .estimation import estimate_tokens
from .guardrails import InputGuard, OutputGuard
from .inference import InferenceBackend, build_backend
from .retrieval import Retriever, InMemoryVectorStore
from .router import CascadeRouter
from .schemas import ChatRequest, ChatResponse, Telemetry

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the provided context. "
    "If the context is insufficient, say so."
)


class LLMPipeline:
    def __init__(self, *, settings: Optional[Settings] = None,
                 store: Optional[InMemoryVectorStore] = None,
                 backend: Optional[InferenceBackend] = None) -> None:
        self.settings = settings or Settings()
        self.backend = backend or build_backend(self.settings)
        self.retriever = Retriever(store or InMemoryVectorStore())
        self.cache = ResponseCache(
            ttl_seconds=self.settings.cache_ttl_seconds,
            semantic_threshold=self.settings.semantic_cache_threshold,
        )
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()
        self.cascade = CascadeRouter(
            self.backend, confidence_floor=self.settings.cascade_confidence_floor,
            registry=MODEL_REGISTRY)

    def _build_prompt(self, query: str, context: str) -> str:
        ctx = context or "(no context retrieved)"
        return f"{SYSTEM_PROMPT}\n\n# Context\n{ctx}\n\n# Question\n{query}\n\n# Answer\n"

    def handle(self, req: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()
        tel = Telemetry()

        # 1) Input safety (fast, deterministic, on the hot path).
        guard = self.input_guard.check(req.query)
        if not guard.allowed:
            tel.blocked, tel.block_reason = True, guard.reason
            tel.latency_ms = (time.perf_counter() - t0) * 1000
            return ChatResponse("Request blocked by input policy.", [], tel)
        clean_query = guard.text

        # 2) Cache lookup (exact -> semantic).
        cached = self.cache.get(clean_query)
        if cached:
            kind, answer, citations = cached
            tel.cache = kind
            tel.latency_ms = (time.perf_counter() - t0) * 1000
            return ChatResponse(answer, citations, tel)

        # 3) Retrieval.
        context, citations, hits = ("", [], [])
        if req.use_rag:
            context, citations, hits = self.retriever.retrieve(clean_query, k=4)
            tel.retrieved_docs = len(hits)

        # 4) Routing + cascade + generation.
        prompt = self._build_prompt(clean_query, context)
        comp, reason, escalated = self.cascade.run(
            prompt, clean_query,
            context_tokens=estimate_tokens(context),
            max_tokens=req.max_tokens, temperature=req.temperature,
            force_model=req.force_model)

        # 5) Output safety + grounding check.
        out = self.output_guard.check(
            comp.text, require_grounding=req.use_rag, has_context=bool(hits))
        if not out.allowed:
            tel.blocked, tel.block_reason = True, out.reason
            tel.latency_ms = (time.perf_counter() - t0) * 1000
            return ChatResponse("Response withheld by output policy.", [], tel)

        # 6) Telemetry + cache write.
        spec = MODEL_REGISTRY[comp.model]
        tel.model = comp.model
        tel.route_reason = reason
        tel.escalated = escalated
        tel.confidence = round(comp.confidence, 3)
        tel.input_tokens = comp.input_tokens
        tel.output_tokens = comp.output_tokens
        tel.cost_usd = round(spec.request_cost_usd(comp.input_tokens, comp.output_tokens), 6)
        tel.latency_ms = (time.perf_counter() - t0) * 1000

        self.cache.put(clean_query, out.text, citations)
        return ChatResponse(out.text, citations, tel)

    def stream(self, req: ChatRequest) -> Iterator[str]:
        """Token-streaming variant for low time-to-first-token UX (Chapter 2)."""
        guard = self.input_guard.check(req.query)
        if not guard.allowed:
            yield "Request blocked by input policy."
            return
        context, citations, _ = self.retriever.retrieve(guard.text, k=4) if req.use_rag else ("", [], [])
        prompt = self._build_prompt(guard.text, context)
        decision = self.cascade.router.route(guard.text, estimate_tokens(context), req.force_model)
        for chunk in self.backend.stream(
                prompt, model=decision.model,
                max_tokens=req.max_tokens, temperature=req.temperature):
            yield chunk
