"""Offline evaluation harness (Part VIII).

Runs a suite of cases through the pipeline and aggregates correctness, cost,
and latency. This is the gate you put in CI so a prompt or model change cannot
silently regress quality or blow the cost budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .pipeline import LLMPipeline
from .schemas import ChatRequest


@dataclass
class EvalCase:
    query: str
    # A predicate over the answer; defaults to "non-empty".
    check: Optional[Callable[[str], bool]] = None
    expect_substring: Optional[str] = None
    use_rag: bool = True

    def passed(self, answer: str) -> bool:
        if self.expect_substring is not None:
            return self.expect_substring.lower() in answer.lower()
        if self.check is not None:
            return self.check(answer)
        return bool(answer.strip())


@dataclass
class EvalReport:
    total: int = 0
    passed: int = 0
    total_cost_usd: float = 0.0
    p50_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    escalation_rate: float = 0.0
    cache_hit_rate: float = 0.0
    rows: List[dict] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class EvalHarness:
    def __init__(self, pipeline: LLMPipeline) -> None:
        self.pipeline = pipeline

    def run(self, cases: List[EvalCase]) -> EvalReport:
        rep = EvalReport(total=len(cases))
        latencies: List[float] = []
        escalations = caches = 0
        for c in cases:
            resp = self.pipeline.handle(
                ChatRequest(query=c.query, use_rag=c.use_rag))
            ok = c.passed(resp.answer)
            rep.passed += int(ok)
            rep.total_cost_usd += resp.telemetry.cost_usd
            latencies.append(resp.telemetry.latency_ms)
            escalations += int(resp.telemetry.escalated)
            caches += int(resp.telemetry.cache != "miss")
            rep.rows.append({
                "query": c.query, "passed": ok,
                "model": resp.telemetry.model or resp.telemetry.cache,
                "cost_usd": resp.telemetry.cost_usd,
                "latency_ms": round(resp.telemetry.latency_ms, 2),
            })
        if latencies:
            ordered = sorted(latencies)
            rep.p50_latency_ms = round(ordered[len(ordered) // 2], 2)
            rep.avg_latency_ms = round(sum(latencies) / len(latencies), 2)
        n = rep.total or 1
        rep.escalation_rate = round(escalations / n, 3)
        rep.cache_hit_rate = round(caches / n, 3)
        rep.total_cost_usd = round(rep.total_cost_usd, 6)
        return rep
