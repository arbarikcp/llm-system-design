"""Model strategy: routing and cascading (Chapter 3).

``ModelRouter`` picks an initial model from cheap features (length, intent,
context size). ``CascadeRouter`` wraps it: run the cheap model first, and
escalate to a stronger model only when confidence falls below a floor. This is
the single highest-leverage cost lever in most LLM products.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .config import MODEL_REGISTRY, ModelSpec, CHEAP_MODEL, DEFAULT_MODEL, STRONG_MODEL
from .estimation import estimate_tokens
from .inference import Completion, InferenceBackend

_HARD_INTENT = re.compile(
    r"\b(prove|derive|analyze|debug|refactor|architect|trade-?off|why|"
    r"step by step|reason)\b", re.I)
_SIMPLE_INTENT = re.compile(
    r"\b(define|what is|list|translate|summari[sz]e|yes or no)\b", re.I)


@dataclass
class RouteDecision:
    model: ModelSpec
    reason: str


class ModelRouter:
    """Heuristic first-hop router. Replace the heuristics with a learned
    classifier once you have labelled (query -> best-model) data."""

    def __init__(self, registry=MODEL_REGISTRY) -> None:
        self.registry = registry

    def route(self, query: str, context_tokens: int = 0,
              force_model: Optional[str] = None) -> RouteDecision:
        if force_model and force_model in self.registry:
            return RouteDecision(self.registry[force_model], "forced")

        total_tokens = estimate_tokens(query) + context_tokens

        # 1) Context that overflows smaller windows forces a large-context model.
        if total_tokens > self.registry[DEFAULT_MODEL].ctx_window * 0.8:
            return RouteDecision(self.registry[STRONG_MODEL], "long_context")
        # 2) Hard reasoning intent -> default/strong model.
        if _HARD_INTENT.search(query):
            return RouteDecision(self.registry[DEFAULT_MODEL], "reasoning_intent")
        # 3) Clearly simple intent and short -> cheapest model.
        if _SIMPLE_INTENT.search(query) and total_tokens < 400:
            return RouteDecision(self.registry[CHEAP_MODEL], "simple_intent")
        # 4) Default.
        return RouteDecision(self.registry[DEFAULT_MODEL], "default")


class CascadeRouter:
    """Cheap-first cascade with confidence-based escalation."""

    def __init__(self, backend: InferenceBackend, router: Optional[ModelRouter] = None,
                 confidence_floor: float = 0.55, registry=MODEL_REGISTRY) -> None:
        self.backend = backend
        self.router = router or ModelRouter(registry)
        self.confidence_floor = confidence_floor
        self.registry = registry

    def run(self, prompt: str, query: str, *, context_tokens: int,
            max_tokens: int, temperature: float,
            force_model: Optional[str] = None) -> Tuple[Completion, str, bool]:
        decision = self.router.route(query, context_tokens, force_model)
        comp = self.backend.generate(
            prompt, model=decision.model, max_tokens=max_tokens,
            temperature=temperature)

        escalated = False
        reason = decision.reason
        # Only escalate when we did not start at the strongest tier and the
        # cheap answer is under-confident. force_model disables the cascade.
        if (not force_model and comp.confidence < self.confidence_floor
                and decision.model.tier != "large"):
            strong = self.registry[STRONG_MODEL]
            comp = self.backend.generate(
                prompt, model=strong, max_tokens=max_tokens,
                temperature=temperature)
            escalated = True
            reason = f"{decision.reason}->escalated(low_confidence)"
        return comp, reason, escalated
