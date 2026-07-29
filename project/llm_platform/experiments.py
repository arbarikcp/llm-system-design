"""Online experimentation: A/B tests, shadow, and canary (Chapter 28).

Deterministic variant assignment by hashing the randomization unit, metric
aggregation with significance tests (two-proportion z-test and Welch's t-test),
guardrail metrics with stopping rules, a shadow comparator, and a canary
controller that ramps traffic and auto-rolls-back on regression.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def assign_variant(unit_id: str, variants: Sequence[str],
                   weights: Optional[Sequence[float]] = None) -> str:
    """Deterministically bucket a unit (user/session/request) into a variant."""
    weights = list(weights) if weights else [1.0] * len(variants)
    total = sum(weights)
    h = int(hashlib.sha256(unit_id.encode()).hexdigest(), 16) % 10_000
    point = (h / 10_000) * total
    cum = 0.0
    for v, w in zip(variants, weights):
        cum += w
        if point < cum:
            return v
    return variants[-1]


def two_proportion_ztest(succ_a: int, n_a: int, succ_b: int, n_b: int) -> Tuple[float, float]:
    """Z-test for a difference in success rates (e.g. task-success). Returns (z, p)."""
    if n_a == 0 or n_b == 0:
        return (0.0, 1.0)
    p_a, p_b = succ_a / n_a, succ_b / n_b
    p_pool = (succ_a + succ_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return (0.0, 1.0)
    z = (p_b - p_a) / se
    p = 2 * (1 - _normal_cdf(abs(z)))
    return (round(z, 4), round(p, 4))


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Welch's t-test for a difference in means (continuous metrics). Returns (t, p)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (0.0, 1.0)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return (0.0, 1.0)
    t = (mb - ma) / se
    # Normal approximation to the t-distribution p-value (good for large n).
    p = 2 * (1 - _normal_cdf(abs(t)))
    return (round(t, 4), round(p, 4))


@dataclass
class ABTest:
    variants: List[str]
    weights: Optional[List[float]] = None
    _values: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        self._values = {v: [] for v in self.variants}

    def record(self, unit_id: str, value: float) -> str:
        v = assign_variant(unit_id, self.variants, self.weights)
        self._values[v].append(value)
        return v

    def mean(self, variant: str) -> float:
        vals = self._values[variant]
        return sum(vals) / len(vals) if vals else 0.0

    def compare(self, control: str, treatment: str) -> Tuple[float, float]:
        return welch_t_test(self._values[control], self._values[treatment])

    def guardrail_breached(self, variant: str, *, metric_floor: float) -> bool:
        return self.mean(variant) < metric_floor


@dataclass
class ShadowComparator:
    """Compare a shadow model's outputs to production without serving them."""
    agreements: int = 0
    total: int = 0
    shadow_cost: float = 0.0

    def compare(self, prod_output: str, shadow_output: str, *,
                shadow_request_cost: float = 0.0) -> None:
        self.total += 1
        self.shadow_cost += shadow_request_cost
        if prod_output.strip().lower() == shadow_output.strip().lower():
            self.agreements += 1

    def agreement_rate(self) -> float:
        return self.agreements / self.total if self.total else 0.0


class CanaryController:
    """Ramp traffic through stages, holding or rolling back on regression."""

    def __init__(self, stages: Sequence[float] = (0.01, 0.05, 0.25, 1.0),
                 *, rollback_delta: float = 0.05) -> None:
        self.stages = list(stages)
        self.rollback_delta = rollback_delta
        self.stage_index = 0

    @property
    def traffic(self) -> float:
        return self.stages[self.stage_index]

    def evaluate(self, control_metric: float, canary_metric: float) -> str:
        """Return 'rollback', 'advance', or 'complete' based on the canary's metric."""
        if canary_metric < control_metric - self.rollback_delta:
            return "rollback"
        if self.stage_index >= len(self.stages) - 1:
            return "complete"
        self.stage_index += 1
        return "advance"
