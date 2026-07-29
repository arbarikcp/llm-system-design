"""Monitoring and observability (Chapter 29).

Latency percentiles (p50/p95/p99), throughput and error rates, cost attribution
by dimension (model/user/feature), output-distribution drift via the Population
Stability Index, and trace collection for multi-step agent workflows.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in [0,100])."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(p / 100 * len(s)) - 1))
    return s[k]


@dataclass
class RequestMetric:
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    feature: str = "default"
    user_id: str = "anon"
    cost_usd: float = 0.0
    error: Optional[str] = None
    ts: float = field(default_factory=time.time)


@dataclass
class MetricsCollector:
    records: List[RequestMetric] = field(default_factory=list)

    def record(self, m: RequestMetric) -> None:
        self.records.append(m)

    def _ok(self) -> List[RequestMetric]:
        return [r for r in self.records if r.error is None]

    def latency_percentiles(self) -> Dict[str, float]:
        lat = [r.latency_ms for r in self._ok()]
        return {"p50": round(percentile(lat, 50), 2),
                "p95": round(percentile(lat, 95), 2),
                "p99": round(percentile(lat, 99), 2)}

    def error_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.error) / len(self.records)

    def throughput_rps(self) -> float:
        if len(self.records) < 2:
            return 0.0
        span = max(r.ts for r in self.records) - min(r.ts for r in self.records)
        return round(len(self.records) / span, 3) if span > 0 else 0.0

    def tokens_per_sec(self) -> float:
        if len(self.records) < 2:
            return 0.0
        span = max(r.ts for r in self.records) - min(r.ts for r in self.records)
        toks = sum(r.output_tokens for r in self.records)
        return round(toks / span, 2) if span > 0 else 0.0

    def cost_by(self, dimension: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for r in self.records:
            key = getattr(r, dimension, "unknown")
            out[key] = round(out.get(key, 0.0) + r.cost_usd, 6)
        return out

    def cost_anomaly(self, *, window: int = 20, factor: float = 3.0) -> bool:
        """Flag if recent average cost/request spikes vs the prior baseline."""
        if len(self.records) < 2 * window:
            return False
        recent = self.records[-window:]
        base = self.records[-2 * window:-window]
        r_avg = sum(x.cost_usd for x in recent) / window
        b_avg = sum(x.cost_usd for x in base) / window
        return b_avg > 0 and r_avg > factor * b_avg


def population_stability_index(expected: Sequence[float], actual: Sequence[float],
                               *, bins: int = 10) -> float:
    """PSI between a reference (expected) and current (actual) distribution.

    Rule of thumb: <0.1 no shift, 0.1-0.25 moderate, >0.25 significant drift.
    """
    if not expected or not actual:
        return 0.0
    lo, hi = min(expected), max(expected)
    if hi == lo:
        return 0.0
    width = (hi - lo) / bins

    def hist(xs):
        counts = [0] * bins
        for x in xs:
            idx = min(bins - 1, max(0, int((x - lo) / width)))
            counts[idx] += 1
        return [c / len(xs) for c in counts]

    e, a = hist(expected), hist(actual)
    psi = 0.0
    for ei, ai in zip(e, a):
        ei = max(ei, 1e-6)
        ai = max(ai, 1e-6)
        psi += (ai - ei) * math.log(ai / ei)
    return round(psi, 4)


# --- tracing for multi-step agent workflows ------------------------------
@dataclass
class Span:
    name: str
    start: float
    end: Optional[float] = None
    attributes: Dict[str, str] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return ((self.end or self.start) - self.start) * 1000.0


@dataclass
class Trace:
    trace_id: str
    spans: List[Span] = field(default_factory=list)

    def start_span(self, name: str, **attributes) -> Span:
        sp = Span(name, time.perf_counter(), attributes=attributes)
        self.spans.append(sp)
        return sp

    def end_span(self, span: Span) -> None:
        span.end = time.perf_counter()

    def total_ms(self) -> float:
        return round(sum(s.duration_ms for s in self.spans), 3)

    def step_count(self) -> int:
        return len(self.spans)
