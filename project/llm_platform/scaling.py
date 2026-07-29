"""Horizontal scaling: autoscaling and token-aware load balancing (Chapter 30).

Stateless LLM replicas scale on queue depth (not CPU), and traffic must be
balanced by in-flight *tokens* rather than request count, because requests vary
enormously in cost. Includes cold-start pre-warm sizing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ScaleDecision:
    desired_replicas: int
    reason: str


class QueueDepthAutoscaler:
    """Scale replicas to keep queue depth per replica near a target.

    Queue depth (and in-flight work), not CPU/GPU utilization, is the right LLM
    autoscaling signal: a memory/queue-bound system can be at modest utilization
    while requests pile up (Chapter 1).
    """

    def __init__(self, *, target_load_per_replica: float = 8.0,
                 min_replicas: int = 1, max_replicas: int = 256,
                 scale_step: int = 4) -> None:
        self.target = target_load_per_replica
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scale_step = scale_step

    def desired(self, *, current_replicas: int, queue_depth: int,
                in_flight: int) -> ScaleDecision:
        load = queue_depth + in_flight
        target_replicas = math.ceil(load / self.target) if self.target else current_replicas
        target_replicas = max(self.min_replicas, min(self.max_replicas, target_replicas))
        # Limit how fast we add capacity (cold starts are slow); shed faster than we add.
        if target_replicas > current_replicas:
            desired = min(target_replicas, current_replicas + self.scale_step)
            reason = "scale_up"
        elif target_replicas < current_replicas:
            desired = target_replicas
            reason = "scale_down"
        else:
            desired, reason = current_replicas, "hold"
        return ScaleDecision(desired, reason)


class TokenAwareLoadBalancer:
    """Route to the replica with the fewest in-flight tokens, with optional
    session affinity for conversation continuity."""

    def __init__(self, replicas: List[str]) -> None:
        if not replicas:
            raise ValueError("need at least one replica")
        self.load: Dict[str, int] = {r: 0 for r in replicas}
        self._affinity: Dict[str, str] = {}

    def route(self, est_tokens: int, *, session_id: Optional[str] = None) -> str:
        if session_id and self._affinity.get(session_id) in self.load:
            replica = self._affinity[session_id]
        else:
            replica = min(self.load, key=self.load.get)   # least in-flight tokens
            if session_id:
                self._affinity[session_id] = replica
        self.load[replica] += est_tokens
        return replica

    def complete(self, replica: str, est_tokens: int) -> None:
        self.load[replica] = max(0, self.load[replica] - est_tokens)

    def add_replica(self, replica: str) -> None:
        self.load.setdefault(replica, 0)

    def remove_replica(self, replica: str) -> None:
        self.load.pop(replica, None)
        self._affinity = {s: r for s, r in self._affinity.items() if r != replica}


def prewarm_pool_size(*, peak_rps: float, cold_start_s: float,
                      per_replica_rps: float, safety: float = 1.5) -> int:
    """Warm replicas to keep ready so a traffic ramp is covered during cold start.

    During a ``cold_start_s`` window, up to peak_rps*cold_start_s requests can
    arrive before new replicas are ready; pre-warm enough to serve them.
    """
    burst = peak_rps * cold_start_s
    return max(1, math.ceil(burst / per_replica_rps * safety))
