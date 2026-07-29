"""API gateway request management (Chapter 12).

The control surface in front of inference: token-bucket rate limiting,
token-budget enforcement (LLM rate limits are measured in tokens, not just
requests), tiered limits, and a priority queue so premium traffic is served
first under load.
"""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class TokenBucket:
    """Classic token-bucket limiter. ``capacity`` is the burst size,
    ``refill_per_sec`` the steady-state rate."""
    capacity: float
    refill_per_sec: float
    _tokens: float = field(default=None)  # type: ignore
    _last: Optional[float] = field(default=None)

    def __post_init__(self):
        if self._tokens is None:
            self._tokens = self.capacity

    def _refill(self, now: float) -> None:
        if self._last is None:           # first touch: anchor the clock, no refill
            self._last = now
            return
        elapsed = max(0.0, now - self._last)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)
        self._last = now

    def allow(self, cost: float = 1.0, *, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.monotonic()
        self._refill(now)
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False


# Per-tier limits: (tokens-per-minute capacity, refill tokens/sec).
TIER_LIMITS: Dict[str, Tuple[float, float]] = {
    "free":  (20_000, 20_000 / 60),
    "pro":   (200_000, 200_000 / 60),
    "scale": (2_000_000, 2_000_000 / 60),
}


class RateLimiter:
    """Token-aware, per-user, per-tier rate limiter.

    LLM cost is dominated by tokens, so we charge the bucket the *estimated
    total tokens* (input + expected output), not one unit per request. This is
    how production LLM gateways prevent a single large-context request from
    starving everyone else.
    """

    def __init__(self, tier_limits=TIER_LIMITS) -> None:
        self.tier_limits = tier_limits
        self._buckets: Dict[str, TokenBucket] = {}

    def _bucket(self, user_id: str, tier: str) -> TokenBucket:
        key = f"{tier}:{user_id}"
        if key not in self._buckets:
            cap, refill = self.tier_limits.get(tier, self.tier_limits["free"])
            self._buckets[key] = TokenBucket(cap, refill)
        return self._buckets[key]

    def check(self, *, user_id: str, tier: str, est_tokens: int,
              now: Optional[float] = None) -> bool:
        return self._bucket(user_id, tier).allow(est_tokens, now=now)


@dataclass(order=True)
class _PItem:
    sort_key: Tuple[int, int]
    payload: object = field(compare=False)


class PriorityRequestQueue:
    """Min-heap priority queue. Lower ``priority`` value = served first;
    FIFO within a priority. Premium tiers map to a lower number."""

    _TIER_PRIORITY = {"scale": 0, "pro": 1, "free": 2}

    def __init__(self) -> None:
        self._heap: list = []
        self._counter = itertools.count()

    def push(self, payload, *, tier: str = "free") -> None:
        pri = self._TIER_PRIORITY.get(tier, 2)
        heapq.heappush(self._heap, _PItem((pri, next(self._counter)), payload))

    def pop(self):
        if not self._heap:
            raise IndexError("pop from empty queue")
        return heapq.heappop(self._heap).payload

    def __len__(self) -> int:
        return len(self._heap)


if __name__ == "__main__":  # pragma: no cover
    rl = RateLimiter()
    t = time.monotonic()
    ok = [rl.check(user_id="u1", tier="free", est_tokens=8000, now=t) for _ in range(4)]
    print("free-tier allow sequence (20k/min cap):", ok)
    q = PriorityRequestQueue()
    q.push("free-job", tier="free"); q.push("vip-job", tier="scale")
    print("served first:", q.pop())
