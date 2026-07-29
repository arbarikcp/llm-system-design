"""Reliability and fault tolerance (Chapter 31).

A circuit breaker for failing dependencies, retry-with-backoff, redundancy
availability math (k-of-n), a graceful-degradation fallback chain, and an RTO
estimator. These are the primitives that keep an LLM system serving when GPUs,
upstreams, or whole regions fail.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


class CircuitBreaker:
    """Closed -> Open -> Half-Open breaker around a flaky dependency.

    After ``fail_threshold`` consecutive failures it opens (fails fast) for
    ``reset_timeout`` seconds, then allows a trial request (half-open); success
    closes it, failure re-opens it.
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, *, fail_threshold: int = 5, reset_timeout: float = 30.0) -> None:
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.state = self.CLOSED
        self._failures = 0
        self._opened_at = 0.0

    def allow(self, *, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.monotonic()
        if self.state == self.OPEN:
            if now - self._opened_at >= self.reset_timeout:
                self.state = self.HALF_OPEN
                return True            # allow one trial request
            return False
        return True                    # closed or half-open

    def record_success(self) -> None:
        self._failures = 0
        self.state = self.CLOSED

    def record_failure(self, *, now: Optional[float] = None) -> None:
        now = now if now is not None else time.monotonic()
        self._failures += 1
        if self.state == self.HALF_OPEN or self._failures >= self.fail_threshold:
            self.state = self.OPEN
            self._opened_at = now


def retry_with_backoff(fn: Callable[[], object], *, max_retries: int = 3,
                       base: float = 0.0, factor: float = 2.0) -> Tuple[bool, object, int]:
    """Call fn, retrying transient exceptions with exponential backoff.

    Returns (ok, result_or_error, attempts). base=0 keeps tests instant.
    """
    last_error: object = None
    for attempt in range(1, max_retries + 2):
        try:
            return (True, fn(), attempt)
        except Exception as exc:           # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt <= max_retries:
                time.sleep(base * (factor ** (attempt - 1)))
                continue
            return (False, last_error, attempt)
    return (False, last_error, max_retries + 1)


def system_availability(*, replica_availability: float, n: int, required: int) -> float:
    """P(at least ``required`` of ``n`` replicas are up), each up with prob p.

    This is how N+k redundancy is justified: n = required + k spare replicas.
    """
    p = replica_availability
    prob = 0.0
    for i in range(required, n + 1):
        prob += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return min(1.0, prob)


def replicas_for_target_availability(*, replica_availability: float,
                                     required: int, target: float = 0.9999,
                                     max_extra: int = 20) -> int:
    """Smallest n >= required so that k-of-n availability meets the target."""
    for n in range(required, required + max_extra + 1):
        if system_availability(replica_availability=replica_availability,
                               n=n, required=required) >= target:
            return n
    return required + max_extra


@dataclass
class FallbackChain:
    """Try handlers in order until one succeeds (graceful degradation)."""
    handlers: List[Callable[[str], str]] = field(default_factory=list)

    def run(self, request: str) -> Tuple[str, int]:
        last_error = ""
        for idx, handler in enumerate(self.handlers):
            try:
                return (handler(request), idx)   # (result, which handler served it)
            except Exception as exc:             # noqa: BLE001
                last_error = str(exc)
        raise RuntimeError(f"all fallbacks failed: {last_error}")


def rto_estimate(*, detect_s: float, failover_s: float, warmup_s: float) -> float:
    """Recovery Time Objective: detection + failover + warm-up."""
    return round(detect_s + failover_s + warmup_s, 2)
