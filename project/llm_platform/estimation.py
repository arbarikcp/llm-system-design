"""Back-of-the-envelope capacity & cost estimation (Chapter 1).

These are the numbers you compute on a whiteboard in a system-design
interview: tokens/sec/GPU, KV-cache memory per concurrent request, cost per
million tokens, and the GPU count needed to hit a throughput SLO.

All functions are pure and unit-tested so you can trust them under pressure.
"""
from __future__ import annotations

from dataclasses import dataclass

# A widely used rule of thumb: ~4 characters per token for English text.
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Cheap, provider-agnostic token estimate.

    Real systems call the model's tokenizer; this heuristic is good enough
    for budgeting and admission control and never needs network access.
    """
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass
class KVCacheEstimate:
    bytes_per_token: int
    per_request_gb: float
    max_concurrent_requests: int


def kv_cache_memory(
    *,
    num_layers: int,
    hidden_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,        # fp16/bf16
    gpu_memory_gb: float = 80.0,
    weights_gb: float = 16.0,    # model weights resident on the GPU
) -> KVCacheEstimate:
    """KV-cache memory per request and how many fit on one GPU.

    KV bytes/token = 2 (K and V) * num_layers * num_kv_heads * head_dim * dtype.
    With grouped-query attention ``num_kv_heads`` is far smaller than the number
    of query heads, which is exactly why GQA slashes serving memory.
    """
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    per_request_gb = bytes_per_token * seq_len / (1024 ** 3)
    usable_gb = max(0.0, gpu_memory_gb - weights_gb)
    max_concurrent = int(usable_gb / per_request_gb) if per_request_gb else 0
    return KVCacheEstimate(bytes_per_token, round(per_request_gb, 4), max_concurrent)


def cost_per_million_tokens(usd_per_1k: float) -> float:
    return usd_per_1k * 1000.0


def monthly_inference_cost(
    *,
    requests_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    usd_per_1k_input: float,
    usd_per_1k_output: float,
    cache_hit_rate: float = 0.0,
) -> float:
    """Projected monthly $ spend, discounted by the cache hit rate."""
    billable = requests_per_day * (1.0 - cache_hit_rate)
    per_req = (avg_input_tokens / 1000.0) * usd_per_1k_input + (
        avg_output_tokens / 1000.0
    ) * usd_per_1k_output
    return round(billable * per_req * 30.0, 2)


def gpus_for_throughput(
    *,
    target_output_tokens_per_sec: int,
    per_replica_tokens_per_sec: int,
    utilization: float = 0.7,
) -> int:
    """Replica count to sustain a decode-throughput SLO with headroom.

    Dividing by ``utilization`` leaves slack for bursts, batching gaps, and
    p99 tails -- never size a fleet at 100% of theoretical peak.
    """
    import math
    effective = per_replica_tokens_per_sec * utilization
    return max(1, math.ceil(target_output_tokens_per_sec / effective))


if __name__ == "__main__":  # pragma: no cover - manual sanity run
    kv = kv_cache_memory(
        num_layers=32, hidden_size=4096, seq_len=4096,
        num_kv_heads=8, head_dim=128,
    )
    print("KV/token bytes:", kv.bytes_per_token)
    print("Per-request GB:", kv.per_request_gb)
    print("Concurrent reqs/GPU:", kv.max_concurrent_requests)
    print("Cost/1M out tokens ($):", cost_per_million_tokens(0.03))
    print("GPUs for 5k tok/s:", gpus_for_throughput(
        target_output_tokens_per_sec=5000, per_replica_tokens_per_sec=40))
