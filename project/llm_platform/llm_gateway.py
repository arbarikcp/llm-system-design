"""Unified multi-provider LLM gateway (Chapter 50).

Abstracts many providers (OpenAI, Anthropic, Google, self-hosted) behind one
interface, routes by capability/cost/latency, and fails over to a healthy
provider when one is down -- the LiteLLM/Portkey-style pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Provider:
    name: str
    models: set                      # capability tags / model names it serves
    usd_per_1k: float
    p50_latency_ms: int
    healthy: bool = True
    # call(request) -> response text; defaults to an echo stand-in.
    call: Optional[Callable[[str], str]] = None

    def serve(self, request: str) -> str:
        if self.call:
            return self.call(request)
        return f"[{self.name}] {request[:64]}"


class MultiProviderGateway:
    def __init__(self) -> None:
        self._providers: List[Provider] = []

    def register(self, provider: Provider) -> None:
        self._providers.append(provider)

    def _candidates(self, capability: str) -> List[Provider]:
        return [p for p in self._providers if p.healthy and capability in p.models]

    def route(self, capability: str, *, optimize: str = "cost") -> Provider:
        cands = self._candidates(capability)
        if not cands:
            raise RuntimeError(f"no healthy provider for capability '{capability}'")
        key = (lambda p: p.usd_per_1k) if optimize == "cost" else (lambda p: p.p50_latency_ms)
        return min(cands, key=key)

    def call(self, capability: str, request: str, *, optimize: str = "cost") -> dict:
        """Route, call, and fail over across healthy providers on error."""
        cands = sorted(self._candidates(capability),
                       key=(lambda p: p.usd_per_1k) if optimize == "cost"
                       else (lambda p: p.p50_latency_ms))
        if not cands:
            raise RuntimeError(f"no healthy provider for capability '{capability}'")
        last_error = ""
        for attempt, provider in enumerate(cands, 1):
            try:
                text = provider.serve(request)
                return {"provider": provider.name, "response": text, "attempts": attempt}
            except Exception as exc:                 # failover to the next provider
                last_error = f"{type(exc).__name__}: {exc}"
                continue
        raise RuntimeError(f"all providers failed: {last_error}")

    def mark(self, name: str, *, healthy: bool) -> None:
        for p in self._providers:
            if p.name == name:
                p.healthy = healthy
