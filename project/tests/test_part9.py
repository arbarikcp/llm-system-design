"""Tests for the Part IX scaling, reliability, and caching modules."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from llm_platform.scaling import (
    QueueDepthAutoscaler, TokenAwareLoadBalancer, prewarm_pool_size,
)
from llm_platform.reliability import (
    CircuitBreaker, retry_with_backoff, system_availability,
    replicas_for_target_availability, FallbackChain, rto_estimate,
)
from llm_platform.caching import (
    LRUCache, EmbeddingCache, prefix_cache_savings, MultiLevelCache,
)


# --- scaling -------------------------------------------------------------
def test_autoscaler_scales_up_and_down():
    asg = QueueDepthAutoscaler(target_load_per_replica=8, scale_step=4)
    up = asg.desired(current_replicas=2, queue_depth=80, in_flight=20)
    assert up.reason == "scale_up" and up.desired_replicas > 2
    down = asg.desired(current_replicas=10, queue_depth=0, in_flight=4)
    assert down.reason == "scale_down" and down.desired_replicas < 10
    hold = asg.desired(current_replicas=2, queue_depth=8, in_flight=8)
    assert hold.reason == "hold"


def test_autoscaler_respects_min_max_and_step():
    asg = QueueDepthAutoscaler(target_load_per_replica=8, min_replicas=2,
                               max_replicas=20, scale_step=3)
    up = asg.desired(current_replicas=2, queue_depth=10_000, in_flight=0)
    assert up.desired_replicas == 5            # limited by scale_step (2+3)
    down = asg.desired(current_replicas=2, queue_depth=0, in_flight=0)
    assert down.desired_replicas == 2          # min floor


def test_token_aware_balancer_picks_least_loaded():
    lb = TokenAwareLoadBalancer(["r1", "r2"])
    lb.route(1000)                              # r1 now loaded
    chosen = lb.route(10)                       # should pick the lighter r2
    assert chosen == "r2"


def test_balancer_session_affinity():
    lb = TokenAwareLoadBalancer(["r1", "r2"])
    first = lb.route(10, session_id="s1")
    again = lb.route(10, session_id="s1")
    assert first == again                       # sticky to the same replica


def test_prewarm_pool_size():
    n = prewarm_pool_size(peak_rps=100, cold_start_s=30, per_replica_rps=50)
    assert n >= 1


# --- reliability ---------------------------------------------------------
def test_circuit_breaker_opens_and_recovers():
    cb = CircuitBreaker(fail_threshold=3, reset_timeout=10)
    t = 1000.0
    for _ in range(3):
        cb.record_failure(now=t)
    assert cb.state == "open"
    assert cb.allow(now=t + 1) is False         # still open
    assert cb.allow(now=t + 11) is True         # half-open trial
    cb.record_success()
    assert cb.state == "closed"


def test_retry_with_backoff():
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"
    ok, value, attempts = retry_with_backoff(flaky, max_retries=3, base=0.0)
    assert ok and value == "ok" and attempts == 3


def test_availability_math():
    # 3-of-4 replicas each 99% available beats a single replica
    a4 = system_availability(replica_availability=0.99, n=4, required=3)
    a1 = system_availability(replica_availability=0.99, n=1, required=1)
    assert a4 > a1
    n = replicas_for_target_availability(replica_availability=0.99, required=3, target=0.9999)
    assert n >= 3


def test_fallback_chain():
    def primary(x): raise RuntimeError("down")
    def secondary(x): return f"cached:{x}"
    chain = FallbackChain([primary, secondary])
    result, idx = chain.run("q")
    assert result == "cached:q" and idx == 1


def test_rto_estimate():
    assert rto_estimate(detect_s=5, failover_s=10, warmup_s=30) == 45.0


# --- caching -------------------------------------------------------------
def test_lru_eviction_and_hit_rate():
    c = LRUCache(capacity=2)
    c.put("a", 1); c.put("b", 2)
    assert c.get("a") == 1                       # 'a' now most-recent
    c.put("c", 3)                                # evicts 'b' (LRU)
    assert c.get("b") is None
    assert c.get("a") == 1
    assert 0 < c.hit_rate() < 1


def test_lru_ttl_expiry():
    c = LRUCache(capacity=10, ttl_seconds=100)
    c.put("k", "v", now=1000.0)
    assert c.get("k", now=1050.0) == "v"
    assert c.get("k", now=2000.0) is None        # expired


def test_embedding_cache_reuses():
    ec = EmbeddingCache(capacity=10)
    v1 = ec.embed("hello world")
    v2 = ec.embed("hello world")
    assert v1 == v2 and ec.cache.hits >= 1


def test_prefix_cache_savings():
    assert prefix_cache_savings(shared_prefix_tokens=500, num_requests=100) == 49_500
    assert prefix_cache_savings(shared_prefix_tokens=500, num_requests=1) == 0


def test_multilevel_cache_levels():
    mlc = MultiLevelCache()
    assert mlc.get("what is rag") is None        # miss
    mlc.put("what is rag", "answer", ["docs/rag.md"])
    level, ans, cites = mlc.get("what is rag")   # exact hit
    assert level == "exact" and ans == "answer"
    assert mlc.hit_rate() > 0
