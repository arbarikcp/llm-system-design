"""Cost engineering: build-vs-buy, TCO, cost-per-token, capacity for growth
(Part XI: Chapters 41-42).

Pure functions for the dollar math behind LLM systems: the self-host-vs-API
break-even, total cost of ownership, cost per token by GPU, and GPU counts under
projected traffic growth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .estimation import monthly_inference_cost, gpus_for_throughput


def self_host_monthly_cost(*, num_gpus: int, gpu_hourly_usd: float,
                           ops_overhead: float = 0.30,
                           hours_per_month: float = 730.0) -> float:
    """Monthly self-hosting cost: GPU-hours plus an ops/overhead multiplier."""
    base = num_gpus * gpu_hourly_usd * hours_per_month
    return round(base * (1 + ops_overhead), 2)


def cost_per_token_usd(*, gpu_hourly_usd: float, tokens_per_sec: float,
                       utilization: float = 0.7) -> float:
    """Effective $/token for a self-hosted GPU at a given decode rate."""
    effective_tps = tokens_per_sec * utilization
    if effective_tps <= 0:
        return float("inf")
    return gpu_hourly_usd / 3600.0 / effective_tps


def breakeven_requests_per_day(*, self_host_monthly_usd: float,
                               usd_per_1k_input: float, usd_per_1k_output: float,
                               avg_input_tokens: int, avg_output_tokens: int,
                               cache_hit_rate: float = 0.0) -> int:
    """Daily request volume at which self-hosting equals API cost.

    Below this volume the API is cheaper; above it, self-hosting wins (ignoring
    the engineering effort captured separately in TCO).
    """
    per_req = (avg_input_tokens / 1000.0) * usd_per_1k_input + (
        avg_output_tokens / 1000.0) * usd_per_1k_output
    billable = per_req * (1 - cache_hit_rate)
    if billable <= 0:
        return 0
    return math.ceil(self_host_monthly_usd / (30.0 * billable))


@dataclass
class TCO:
    hardware_usd: float
    ops_usd: float
    engineering_usd: float

    @property
    def total(self) -> float:
        return round(self.hardware_usd + self.ops_usd + self.engineering_usd, 2)


def tco(*, num_gpus: int, gpu_hourly_usd: float, months: int,
        engineers: float, engineer_monthly_usd: float,
        ops_overhead: float = 0.30) -> TCO:
    """Total cost of ownership over a horizon: hardware + ops + engineering."""
    hardware = num_gpus * gpu_hourly_usd * 730.0 * months
    ops = hardware * ops_overhead
    engineering = engineers * engineer_monthly_usd * months
    return TCO(round(hardware, 2), round(ops, 2), round(engineering, 2))


def gpus_for_growth(*, current_tokens_per_sec: float, monthly_growth: float,
                    months: int, per_gpu_tokens_per_sec: float,
                    utilization: float = 0.7) -> int:
    """GPU count needed after compounding monthly traffic growth.

    Useful for procurement lead-time planning: size for the demand you will have
    when the hardware actually arrives, not today's.
    """
    projected = current_tokens_per_sec * ((1 + monthly_growth) ** months)
    return gpus_for_throughput(
        target_output_tokens_per_sec=int(math.ceil(projected)),
        per_replica_tokens_per_sec=int(per_gpu_tokens_per_sec),
        utilization=utilization)
