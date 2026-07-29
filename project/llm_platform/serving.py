"""Inference-engine estimation and a continuous-batching simulator
(Part III: Chapters 9-11).

Covers the quantitative core of inference system design: the prefill-vs-decode
roofline distinction, paged-KV block accounting, quantization memory, the
expected speedup from speculative decoding, and a tiny iteration-level
(continuous batching) scheduler that demonstrates why it beats static batching.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


# --- Prefill vs. decode roofline ----------------------------------------
@dataclass
class PhaseProfile:
    phase: str
    arithmetic_intensity: float   # FLOPs per byte
    bound: str                    # "compute" | "memory"


def phase_profile(*, num_params: int, batch_tokens: int,
                  weight_bytes_per_param: int = 2,
                  ridge_point: float = 200.0) -> PhaseProfile:
    """Classify a phase as compute- or memory-bound by arithmetic intensity.

    Per forward token costs ~2N FLOPs; reading the weights costs ~weight_bytes*N
    bytes but that read is *shared* across all tokens processed together. So
    intensity ~= 2 * batch_tokens / weight_bytes_per_param. Prefill processes
    the whole prompt at once (large batch_tokens -> compute-bound); decode emits
    one token per request (small batch_tokens -> memory-bound).
    """
    intensity = (2.0 * batch_tokens) / weight_bytes_per_param
    phase = "prefill" if batch_tokens > 1 else "decode"
    bound = "compute" if intensity >= ridge_point else "memory"
    return PhaseProfile(phase, round(intensity, 2), bound)


# --- Paged KV cache -----------------------------------------------------
@dataclass
class PagedKV:
    blocks_needed: int
    tokens_wasted: int
    internal_frag_pct: float


def paged_kv_blocks(*, seq_len: int, block_size: int = 16) -> PagedKV:
    """PagedAttention block accounting: KV is allocated in fixed-size blocks,
    so only the last partial block is wasted (internal fragmentation), versus
    reserving max_seq_len contiguously per request."""
    blocks = math.ceil(seq_len / block_size)
    wasted = blocks * block_size - seq_len
    frag = wasted / (blocks * block_size) * 100 if blocks else 0.0
    return PagedKV(blocks, wasted, round(frag, 2))


def quant_weight_memory_gb(*, num_params: int, bits: int) -> float:
    """Weight memory at a given precision (W4A16 -> bits=4, FP8 -> 8, BF16 -> 16)."""
    return round(num_params * bits / 8 / (1024 ** 3), 2)


# --- Speculative decoding ------------------------------------------------
def speculative_expected_tokens(*, acceptance_rate: float, gamma: int) -> float:
    """Expected number of tokens accepted per verification step.

    With draft proposing gamma tokens and per-token acceptance alpha, the
    expected accepted length is (1 - alpha^(gamma+1)) / (1 - alpha).
    """
    a = acceptance_rate
    if a >= 1.0:
        return float(gamma + 1)
    return (1 - a ** (gamma + 1)) / (1 - a)


def speculative_speedup(*, acceptance_rate: float, gamma: int,
                        draft_cost_ratio: float = 0.15) -> float:
    """Approximate wall-clock speedup vs. vanilla decoding.

    Cost per verification step ~= 1 target forward + gamma*draft forwards.
    Speedup = expected_tokens / (1 + gamma*draft_cost_ratio). Returns <1 when
    acceptance is too low to pay for the draft -- the 'when it hurts' regime.
    """
    exp_tokens = speculative_expected_tokens(acceptance_rate=acceptance_rate, gamma=gamma)
    step_cost = 1.0 + gamma * draft_cost_ratio
    return round(exp_tokens / step_cost, 3)


# --- Continuous batching simulator --------------------------------------
@dataclass
class SimRequest:
    rid: int
    arrival: int            # arrival iteration
    output_len: int         # tokens to generate
    started: int = -1
    finished: int = -1


@dataclass
class BatchStats:
    makespan: int
    throughput_tokens_per_iter: float
    avg_latency: float
    max_batch: int


def simulate_continuous_batching(requests: List[SimRequest], *,
                                 max_batch: int) -> BatchStats:
    """Iteration-level scheduler: each iteration advances every active request
    by one token and immediately admits waiting requests into freed slots.

    Contrast with static batching, where a slot is held until the *whole* batch
    finishes; continuous batching refills slots the moment a request completes,
    which is the central throughput win of vLLM-style engines.
    """
    pending = sorted(requests, key=lambda r: r.arrival)
    active: List[SimRequest] = []
    remaining: Dict[int, int] = {}
    it = 0
    completed = 0
    total_tokens = 0
    peak = 0
    n = len(requests)
    while completed < n:
        # admit any arrived, waiting requests into free slots
        i = 0
        while i < len(pending) and len(active) < max_batch:
            r = pending[i]
            if r.arrival <= it:
                r.started = it
                active.append(r)
                remaining[r.rid] = r.output_len
                pending.pop(i)
            else:
                i += 1
        peak = max(peak, len(active))
        if not active:
            it += 1
            continue
        # one decode step for every active request
        for r in list(active):
            remaining[r.rid] -= 1
            total_tokens += 1
            if remaining[r.rid] <= 0:
                r.finished = it + 1
                active.remove(r)
                completed += 1
        it += 1
    makespan = it
    avg_latency = sum(r.finished - r.arrival for r in requests) / n
    return BatchStats(makespan,
                      round(total_tokens / makespan, 2),
                      round(avg_latency, 2),
                      peak)


if __name__ == "__main__":  # pragma: no cover
    print(phase_profile(num_params=7_000_000_000, batch_tokens=1024))
    print(phase_profile(num_params=7_000_000_000, batch_tokens=1))
    print("W4 vs BF16 GB:", quant_weight_memory_gb(num_params=7e9, bits=4),
          quant_weight_memory_gb(num_params=7e9, bits=16))
    print("spec speedup (alpha=0.8,gamma=4):",
          speculative_speedup(acceptance_rate=0.8, gamma=4))
