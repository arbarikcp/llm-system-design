"""Reference system designs composed from the platform's building blocks
(Part X: Chapters 33-40).

Each case study is an *assembly* of modules built in earlier parts, not a new
subsystem -- which is the whole point of system design: the same primitives
(routing, retrieval, RAG, caching, guardrails, memory, sessions, context
assembly, tools) recombine into very different products.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .rag import RAGIndex, RAGQueryEngine
from .retrieval_hybrid import CrossEncoderReranker
from .router import ModelRouter
from .context_assembler import ContextAssembler, ContextBlock
from .safety import InputSafetyPipeline, OutputSafetyPipeline
from .sessions import ConversationStore, SessionManager
from .memory import AgentMemory
from .tools import ToolRegistry, ToolExecutor, make_calculator_tool
from .inference import MockBackend, InferenceBackend
from .config import MODEL_REGISTRY, DEFAULT_MODEL, CHEAP_MODEL
from .estimation import estimate_tokens
from .safety import toxicity_score, detect_injection, detect_jailbreak
from .gateway import RateLimiter, TIER_LIMITS
from .observability import Trace
from .agents import ReActAgent
from .embeddings import HashingEmbedder, cosine_similarity
from collections import Counter


# ============================================================ Ch.33 Search
def classify_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ("buy", "price", "download", "sign up", "order", "purchase")):
        return "transactional"
    if any(w in q for w in ("login", "homepage", "official site", "website of")):
        return "navigational"
    return "informational"


class SearchEngine:
    """Ch.33: query understanding -> hybrid retrieval -> rerank -> cited answer."""

    def __init__(self, index: RAGIndex, *, backend: Optional[InferenceBackend] = None):
        self.engine = RAGQueryEngine(index, backend=backend or MockBackend())
        self.input_guard = InputSafetyPipeline()

    def search(self, query: str) -> dict:
        guard = self.input_guard.check(query)
        if not guard.allowed:
            return {"blocked": True, "reason": guard.reason}
        intent = classify_intent(guard.text)
        result = self.engine.answer(guard.text, k=8, top_n=4)
        return {
            "intent": intent,
            "answer": result.answer,
            "citations": result.citations,
            "confidence": result.confidence,
            "blocked": False,
        }


# =================================================== Ch.34 Coding assistant
class CodingAssistant:
    """Ch.34: repo-context assembly -> completion; sandboxed test execution."""

    def __init__(self, *, backend: Optional[InferenceBackend] = None, max_context_tokens: int = 1500):
        self.backend = backend or MockBackend()
        self.assembler = ContextAssembler(max_tokens=max_context_tokens)
        self.tools = ToolRegistry()
        self.tools.register(make_calculator_tool())   # stand-in for a code sandbox
        self.executor = ToolExecutor(self.tools)

    def complete(self, *, open_file: str, cursor_prefix: str,
                 related_files: Optional[Dict[str, str]] = None,
                 mode: str = "inline") -> dict:
        blocks = [
            ContextBlock("current_file", open_file, required=True),
            ContextBlock("cursor_prefix", cursor_prefix, priority=100, required=True),
        ]
        for i, (name, content) in enumerate((related_files or {}).items()):
            blocks.append(ContextBlock(f"related:{name}", content, priority=50 - i))
        ctx = self.assembler.assemble(blocks)
        max_tokens = 64 if mode == "inline" else 512    # inline is short, chat is long
        prompt = f"# Repo context\n{ctx.text}\n\n# Complete:\n{cursor_prefix}"
        comp = self.backend.generate(prompt, model=MODEL_REGISTRY[DEFAULT_MODEL],
                                     max_tokens=max_tokens, temperature=0.1)
        return {"completion": comp.text, "mode": mode,
                "context_included": ctx.included, "context_dropped": ctx.dropped}

    def run_in_sandbox(self, expression: str) -> dict:
        """Execute generated code in a sandbox (here: the safe calculator tool)."""
        res = self.executor.execute("calculator", {"expression": expression})
        return {"ok": res.ok, "output": res.output, "error": res.error}


# ==================================================== Ch.35 Chatbot platform
class ChatbotPlatform:
    """Ch.35: sessions + cross-session memory + persona + routing + guardrails."""

    def __init__(self, *, persona: str = "You are a helpful, friendly assistant.",
                 backend: Optional[InferenceBackend] = None):
        self.persona = persona
        self.backend = backend or MockBackend()
        self.store = ConversationStore()
        self.sessions = SessionManager(self.store, ttl_seconds=3600)
        self.router = ModelRouter()
        self.input_guard = InputSafetyPipeline()
        self.output_guard = OutputSafetyPipeline()
        self._memory: Dict[str, AgentMemory] = {}

    def start(self, user_id: str):
        session = self.sessions.start(user_id)
        self._memory.setdefault(user_id, AgentMemory())
        return session

    def message(self, session, text: str) -> dict:
        guard = self.input_guard.check(text)
        if not guard.allowed:
            return {"blocked": True, "reason": guard.reason}
        mem = self._memory[session.user_id]
        mem.observe("user", guard.text)
        decision = self.router.route(guard.text)
        context = mem.assemble_context(guard.text, k=2)
        prompt = f"{self.persona}\n\n{context}\n\nUser: {guard.text}\nAssistant:"
        comp = self.backend.generate(prompt, model=decision.model,
                                     max_tokens=128, temperature=0.3)
        out = self.output_guard.check(comp.text, require_grounding=False)
        mem.observe("assistant", out.text)
        conv = self.store.get(session.conv_id)
        conv.append("user", guard.text)
        conv.append("assistant", out.text)
        return {"answer": out.text, "model": comp.model,
                "route_reason": decision.reason, "blocked": False}

    def remember_fact(self, user_id: str, fact: str) -> None:
        self._memory.setdefault(user_id, AgentMemory()).remember(fact)


# ==================================================== Ch.36 Document Q&A
class DocumentQA:
    """Ch.36: ingest -> ACL-filtered retrieval -> cited, grounded answer."""

    def __init__(self, *, backend: Optional[InferenceBackend] = None):
        self.index = RAGIndex()
        self.reranker = CrossEncoderReranker(self.index.embedder)
        self.backend = backend or MockBackend()
        self.output_guard = OutputSafetyPipeline()
        self._acl: Dict[str, set] = {}            # source -> set(user_ids)

    def ingest(self, doc_id: str, text: str, *, allowed_users: List[str],
               strategy: str = "structure") -> int:
        n = self.index.add_document(doc_id, text, strategy=strategy)
        self._acl[doc_id] = set(allowed_users)
        return n

    def _can_read(self, source: str, user_id: str) -> bool:
        return user_id in self._acl.get(source, set())

    def ask(self, query: str, *, user_id: str, top_n: int = 3) -> dict:
        self.index.build()
        hits = self.index.retriever.search(query, k=12)
        # Enforce document-level permissions BEFORE the model sees anything.
        allowed = [h for h in hits if self._can_read(h["metadata"].get("source", ""), user_id)]
        ranked = self.reranker.rerank(query, allowed, top_k=top_n)
        if not ranked:
            return {"answer": "No accessible documents contain an answer.",
                    "citations": [], "grounded": False}
        ctx = "\n\n".join(f"[{d['doc_id']}] {d['text']}" for d in ranked)
        prompt = (f"Answer using ONLY the context and cite chunk ids.\n\n"
                  f"# Context\n{ctx}\n\n# Question\n{query}\n\n# Answer\n")
        comp = self.backend.generate(prompt, model=MODEL_REGISTRY[DEFAULT_MODEL],
                                     max_tokens=256, temperature=0.1)
        out = self.output_guard.check(comp.text, context=ctx, require_grounding=True)
        return {"answer": out.text, "citations": [d["doc_id"] for d in ranked],
                "grounded": out.reason != "ungrounded"}


# ============================================ Ch.37 Content moderation
@dataclass
class ModerationDecision:
    action: str                # "allow" | "block" | "review"
    stage: str                 # "fast_filter" | "llm_judge"
    score: float
    reason: str = ""


class ContentModerationSystem:
    """Ch.37: cascade -- a fast filter handles the obvious; an LLM judge
    adjudicates the ambiguous middle. Per-content-type policy thresholds."""

    def __init__(self, *, policy: Optional[Dict[str, Tuple[float, float]]] = None):
        # content_type -> (review_threshold, block_threshold)
        self.policy = policy or {
            "text": (0.30, 0.70), "image_caption": (0.25, 0.65),
            "comment": (0.35, 0.75),
        }

    def _llm_judge(self, content: str) -> Tuple[str, str]:
        # Stand-in for an LLM moderation judge on the ambiguous middle.
        if detect_injection(content) or detect_jailbreak(content):
            return "block", "policy_violation"
        return "allow", "judged_safe"

    def moderate(self, content: str, *, content_type: str = "text") -> ModerationDecision:
        review_thr, block_thr = self.policy.get(content_type, (0.3, 0.7))
        score = toxicity_score(content)                  # fast filter (sub-ms)
        if score >= block_thr:
            return ModerationDecision("block", "fast_filter", round(score, 3), "high_toxicity")
        if score < review_thr:
            return ModerationDecision("allow", "fast_filter", round(score, 3))
        action, reason = self._llm_judge(content)        # escalate ambiguous middle
        return ModerationDecision(action, "llm_judge", round(score, 3), reason)


# ============================================ Ch.38 Recommendation
class RecommendationSystem:
    """Ch.38: collaborative filtering for known users, LLM/embedding cold-start
    for new ones, with generated explanations."""

    def __init__(self, *, embedder: Optional[HashingEmbedder] = None):
        self.embedder = embedder or HashingEmbedder()
        self.catalog: Dict[str, str] = {}                # item_id -> description
        self._item_emb: Dict[str, list] = {}
        self.interactions: Dict[str, set] = {}           # user -> set(item_id)

    def add_item(self, item_id: str, description: str) -> None:
        self.catalog[item_id] = description
        self._item_emb[item_id] = self.embedder.embed(description)

    def record_interaction(self, user_id: str, item_id: str) -> None:
        self.interactions.setdefault(user_id, set()).add(item_id)

    def recommend(self, user_id: str, *, k: int = 3,
                  profile_text: Optional[str] = None) -> List[dict]:
        liked = self.interactions.get(user_id, set())
        if liked:                                         # collaborative filtering
            scores: Counter = Counter()
            for other, items in self.interactions.items():
                if other == user_id:
                    continue
                overlap = len(liked & items)
                for it in items - liked:
                    if overlap:
                        scores[it] += overlap
            ranked = [it for it, _ in scores.most_common(k)]
            why = "users with similar taste also liked it"
        else:                                             # cold start via profile
            pv = self.embedder.embed(profile_text or "")
            ranked = sorted(self.catalog,
                            key=lambda it: cosine_similarity(pv, self._item_emb[it]),
                            reverse=True)[:k]
            why = "it matches your stated interests"
        return [{"item": it, "title": self.catalog.get(it, it),
                 "explanation": f"Recommended because {why}."} for it in ranked]


# ============================================ Ch.39 Agent platform
@dataclass
class AgentRun:
    answer: str
    success: bool
    steps: int
    tokens: int
    cost_usd: float
    approved: bool


class AgentPlatform:
    """Ch.39: sandboxed agent execution with tracing, per-run billing, and a
    human-in-the-loop approval gate for high-risk runs."""

    def __init__(self, *, registry: Optional[ToolRegistry] = None,
                 model: str = DEFAULT_MODEL):
        if registry is None:
            registry = ToolRegistry()
            registry.register(make_calculator_tool())     # sandbox tool
        self.registry = registry
        self.model = MODEL_REGISTRY[model]

    def run(self, task: str, *, approve=None) -> AgentRun:
        # Human-in-the-loop: high-risk runs must be approved before executing.
        if approve is not None and not approve(task):
            return AgentRun("Run rejected by approval gate.", False, 0, 0, 0.0, False)
        agent = ReActAgent(self.registry, executor=ToolExecutor(self.registry), max_steps=6)
        result = agent.run(task)
        trace = Trace("agent_run")                          # observability
        tokens = 0
        for s in result.steps:                              # billing attribution
            sp = trace.start_span(s.action or "think"); trace.end_span(sp)
            tokens += estimate_tokens((s.thought or "") + (s.observation or ""))
        cost = self.model.request_cost_usd(tokens, tokens // 2)
        return AgentRun(result.answer, result.success, trace.step_count(),
                        tokens, round(cost, 6), True)


# ============================================ Ch.40 Multi-tenant platform
@dataclass
class Tenant:
    tenant_id: str
    tier: str = "free"
    persona: str = "You are a helpful assistant."
    model: str = DEFAULT_MODEL


class MultiTenantPlatform:
    """Ch.40: per-tenant config, quota enforcement, isolation, cost allocation."""

    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.rate = RateLimiter()
        self.input_guard = InputSafetyPipeline()
        self.output_guard = OutputSafetyPipeline()
        self.backend = MockBackend()
        self.router = ModelRouter()
        self._cost: Dict[str, float] = {}                  # per-tenant chargeback
        self._memory: Dict[str, AgentMemory] = {}          # isolated per tenant

    def onboard(self, tenant_id: str, *, tier: str = "free",
                persona: str = "You are a helpful assistant.",
                model: str = DEFAULT_MODEL) -> Tenant:
        t = Tenant(tenant_id, tier, persona, model)
        self.tenants[tenant_id] = t
        self._cost[tenant_id] = 0.0
        self._memory[tenant_id] = AgentMemory()
        return t

    def handle(self, tenant_id: str, user_id: str, query: str,
               *, now: Optional[float] = None) -> dict:
        tenant = self.tenants.get(tenant_id)
        if tenant is None:
            return {"error": "unknown_tenant"}
        est = estimate_tokens(query) + 256
        if not self.rate.check(user_id=f"{tenant_id}:{user_id}", tier=tenant.tier,
                               est_tokens=est, now=now):
            return {"blocked": True, "reason": "quota_exceeded", "tenant": tenant_id}
        guard = self.input_guard.check(query)
        if not guard.allowed:
            return {"blocked": True, "reason": guard.reason, "tenant": tenant_id}
        decision = self.router.route(guard.text, force_model=tenant.model)
        prompt = f"{tenant.persona}\n\nUser: {guard.text}\nAssistant:"
        comp = self.backend.generate(prompt, model=decision.model,
                                     max_tokens=128, temperature=0.3)
        out = self.output_guard.check(comp.text, require_grounding=False)
        cost = decision.model.request_cost_usd(comp.input_tokens, comp.output_tokens)
        self._cost[tenant_id] += cost                      # cost allocation
        return {"answer": out.text, "tenant": tenant_id, "model": comp.model,
                "cost_usd": round(cost, 6), "blocked": False}

    def chargeback(self, tenant_id: str) -> float:
        return round(self._cost.get(tenant_id, 0.0), 6)
