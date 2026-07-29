"""Configuration, model registry, and global settings.

The model registry is the single source of truth for routing and cost
decisions (Chapter 3). Prices are illustrative $-per-1K-tokens figures used
for back-of-the-envelope math (Chapter 1); update them from your provider's
price sheet in production.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a servable model.

    Attributes mirror the four dimensions every LLM design trades off:
    quality, latency, throughput, and cost.
    """
    name: str
    tier: str                      # "small" | "mid" | "large"
    quality: float                 # 0..1 subjective quality on target tasks
    ctx_window: int                # max context tokens
    usd_per_1k_input: float
    usd_per_1k_output: float
    p50_latency_ms: int            # typical time-to-last-token for ~256 out tokens
    max_tokens_per_sec: int        # decode throughput per replica
    supports_tools: bool = True
    is_local: bool = False         # served on our own GPUs vs vendor API

    def request_cost_usd(self, in_tokens: int, out_tokens: int) -> float:
        """Blended cost of a single request."""
        return (in_tokens / 1000.0) * self.usd_per_1k_input + (
            out_tokens / 1000.0
        ) * self.usd_per_1k_output


# Illustrative registry mixing vendor APIs and a self-hosted small model.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "small-fast": ModelSpec(
        name="small-fast", tier="small", quality=0.62, ctx_window=8192,
        usd_per_1k_input=0.00015, usd_per_1k_output=0.0006,
        p50_latency_ms=320, max_tokens_per_sec=180, is_local=True,
    ),
    "mid-balanced": ModelSpec(
        name="mid-balanced", tier="mid", quality=0.81, ctx_window=32768,
        usd_per_1k_input=0.003, usd_per_1k_output=0.012,
        p50_latency_ms=900, max_tokens_per_sec=85,
    ),
    "large-flagship": ModelSpec(
        name="large-flagship", tier="large", quality=0.93, ctx_window=128000,
        usd_per_1k_input=0.01, usd_per_1k_output=0.03,
        p50_latency_ms=2200, max_tokens_per_sec=40,
    ),
}

DEFAULT_MODEL = "mid-balanced"
CHEAP_MODEL = "small-fast"
STRONG_MODEL = "large-flagship"


@dataclass
class Settings:
    """Runtime settings, overridable via environment variables."""
    default_model: str = field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", DEFAULT_MODEL))
    # If set, OpenAI-compatible backend is used; otherwise the deterministic mock runs.
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    cache_ttl_seconds: int = int(os.getenv("LLM_CACHE_TTL", "900"))
    semantic_cache_threshold: float = float(os.getenv("LLM_SEMCACHE_THRESHOLD", "0.92"))
    cascade_confidence_floor: float = float(os.getenv("LLM_CASCADE_FLOOR", "0.55"))
    max_input_tokens: int = int(os.getenv("LLM_MAX_INPUT_TOKENS", "6000"))

    def use_real_backend(self) -> bool:
        return bool(self.openai_base_url and self.openai_api_key)
