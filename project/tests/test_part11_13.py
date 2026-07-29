"""Tests for the Part XI-XIII modules: cost, registry, compound, reasoning, gateway."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from llm_platform.cost import (
    self_host_monthly_cost, cost_per_token_usd, breakeven_requests_per_day,
    tco, gpus_for_growth,
)
from llm_platform.registry import ModelRegistry, ModelVersion
from llm_platform.compound import FlowGraph, Node
from llm_platform.reasoning import (
    best_of_n, self_consistency, best_of_n_expected_pass, thinking_budget_tokens,
)
from llm_platform.llm_gateway import MultiProviderGateway, Provider


# --- cost ----------------------------------------------------------------
def test_self_host_cost_and_overhead():
    base = self_host_monthly_cost(num_gpus=4, gpu_hourly_usd=2.0, ops_overhead=0.0)
    withovh = self_host_monthly_cost(num_gpus=4, gpu_hourly_usd=2.0, ops_overhead=0.3)
    assert withovh > base


def test_cost_per_token_decreases_with_throughput():
    a = cost_per_token_usd(gpu_hourly_usd=2.0, tokens_per_sec=40)
    b = cost_per_token_usd(gpu_hourly_usd=2.0, tokens_per_sec=80)
    assert b < a


def test_breakeven_is_positive():
    n = breakeven_requests_per_day(
        self_host_monthly_usd=10_000, usd_per_1k_input=0.003, usd_per_1k_output=0.012,
        avg_input_tokens=800, avg_output_tokens=256)
    assert n > 0


def test_tco_sums_components():
    t = tco(num_gpus=8, gpu_hourly_usd=2.0, months=12,
            engineers=2, engineer_monthly_usd=15_000)
    assert t.total == round(t.hardware_usd + t.ops_usd + t.engineering_usd, 2)
    assert t.engineering_usd == 2 * 15_000 * 12


def test_gpus_for_growth_grows():
    base = gpus_for_growth(current_tokens_per_sec=1000, monthly_growth=0.0,
                           months=12, per_gpu_tokens_per_sec=40)
    grown = gpus_for_growth(current_tokens_per_sec=1000, monthly_growth=0.15,
                            months=12, per_gpu_tokens_per_sec=40)
    assert grown > base


# --- registry ------------------------------------------------------------
def test_registry_promote_and_rollback():
    reg = ModelRegistry()
    reg.register(ModelVersion("asst", "1.0", base_model="base", training_data="ds-v1"))
    reg.register(ModelVersion("asst", "2.0", base_model="base", training_data="ds-v2"))
    reg.promote("asst", "1.0")
    assert reg.production("asst").version == "1.0"
    reg.promote("asst", "2.0")
    assert reg.production("asst").version == "2.0"
    assert reg.get("asst", "1.0").stage == "archived"
    reg.rollback("asst")
    assert reg.production("asst").version == "1.0"


def test_registry_lineage():
    reg = ModelRegistry()
    reg.register(ModelVersion("m", "1.0", base_model="llama", training_data="snap-7"))
    lin = reg.lineage("m", "1.0")
    assert lin["base_model"] == "llama" and lin["training_data"] == "snap-7"


# --- compound flow -------------------------------------------------------
def test_flow_runs_in_dependency_order():
    flow = FlowGraph()
    flow.add(Node("a", lambda ctx: 2))
    flow.add(Node("b", lambda ctx: ctx["a"] * 3, deps=["a"]))
    flow.add(Node("c", lambda ctx: ctx["a"] + ctx["b"], deps=["a", "b"]))
    ctx = flow.run()
    assert ctx["a"] == 2 and ctx["b"] == 6 and ctx["c"] == 8


def test_flow_conditional_skip():
    flow = FlowGraph()
    flow.add(Node("route", lambda ctx: "cheap"))
    flow.add(Node("expensive", lambda ctx: "ran", deps=["route"],
                  when=lambda ctx: ctx["route"] == "hard"))
    ctx = flow.run()
    assert "expensive" in ctx["_skipped"]


def test_flow_retry_then_error():
    calls = {"n": 0}
    def flaky(ctx):
        calls["n"] += 1
        raise RuntimeError("boom")
    flow = FlowGraph().add(Node("x", flaky, retries=2))
    ctx = flow.run()
    assert calls["n"] == 3 and "x" in ctx["_errors"]


def test_flow_detects_cycle():
    flow = FlowGraph()
    flow.add(Node("a", lambda c: 1, deps=["b"]))
    flow.add(Node("b", lambda c: 1, deps=["a"]))
    with pytest.raises(ValueError):
        flow.run()


# --- reasoning -----------------------------------------------------------
def test_best_of_n_picks_top():
    best, score = best_of_n(["aa", "b", "cccc"], scorer=len)
    assert best == "cccc" and score == 4


def test_self_consistency_majority():
    ans, frac = self_consistency(["4", "4", "5", "4"])
    assert ans == "4" and abs(frac - 0.75) < 1e-9


def test_best_of_n_diminishing_returns():
    p1 = best_of_n_expected_pass(0.5, 1)
    p4 = best_of_n_expected_pass(0.5, 4)
    assert p1 == 0.5 and p4 > p1 and p4 < 1.0


def test_thinking_budget_scales_with_difficulty():
    assert thinking_budget_tokens(0.0) < thinking_budget_tokens(1.0)


# --- gateway -------------------------------------------------------------
def _gw():
    gw = MultiProviderGateway()
    gw.register(Provider("openai", {"chat"}, usd_per_1k=0.01, p50_latency_ms=900))
    gw.register(Provider("anthropic", {"chat"}, usd_per_1k=0.008, p50_latency_ms=1200))
    gw.register(Provider("local", {"chat"}, usd_per_1k=0.001, p50_latency_ms=400))
    return gw


def test_gateway_routes_by_cost_and_latency():
    gw = _gw()
    assert gw.route("chat", optimize="cost").name == "local"
    assert gw.route("chat", optimize="latency").name == "local"


def test_gateway_failover_on_provider_error():
    gw = MultiProviderGateway()
    gw.register(Provider("down", {"chat"}, 0.001, 100,
                         call=lambda r: (_ for _ in ()).throw(RuntimeError("503"))))
    gw.register(Provider("up", {"chat"}, 0.01, 500, call=lambda r: "ok"))
    res = gw.call("chat", "hi", optimize="cost")
    assert res["provider"] == "up" and res["attempts"] == 2


def test_gateway_unhealthy_excluded():
    gw = _gw()
    gw.mark("local", healthy=False)
    assert gw.route("chat", optimize="cost").name == "anthropic"
