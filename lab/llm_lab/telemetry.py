"""Lightweight telemetry: token estimation, timing, and cost (Chapter 1)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict

CHARS_PER_TOKEN = 4.0

# Illustrative $/1K tokens by model tier (edit to your provider / Ollama = free).
PRICES: Dict[str, Dict[str, float]] = {
    "small": {"in": 0.00015, "out": 0.0006},
    "large": {"in": 0.01, "out": 0.03},
    "offline-heuristic": {"in": 0.0, "out": 0.0},
    "llama3.1": {"in": 0.0, "out": 0.0},   # local models are free at the margin
}


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text or "") / CHARS_PER_TOKEN))


def request_cost_usd(tier: str, in_tokens: int, out_tokens: int) -> float:
    p = PRICES.get(tier, PRICES["small"])
    return round((in_tokens / 1000) * p["in"] + (out_tokens / 1000) * p["out"], 6)


@dataclass
class Telemetry:
    tier: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    extra: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"tier": self.tier, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "latency_ms": round(self.latency_ms, 2),
                "cost_usd": self.cost_usd, **self.extra}


@contextmanager
def timed(tel: Telemetry):
    t0 = time.perf_counter()
    try:
        yield tel
    finally:
        tel.latency_ms = (time.perf_counter() - t0) * 1000.0
