"""Tests for the Part VIII evaluation, experimentation, and observability modules."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_platform.metrics import (
    exact_match, contains, token_f1, rouge_l, pass_at_k, rubric_score, pairwise_judge,
)
from llm_platform.eval_offline import (
    EvalSample, EvalDataset, OfflineEvalRunner, bootstrap_ci, detect_regression,
)
from llm_platform.experiments import (
    assign_variant, two_proportion_ztest, welch_t_test, ABTest,
    ShadowComparator, CanaryController,
)
from llm_platform.observability import (
    percentile, RequestMetric, MetricsCollector,
    population_stability_index, Trace,
)


# --- metrics -------------------------------------------------------------
def test_text_metrics():
    assert exact_match("Paris", " paris ") == 1.0
    assert contains("the capital is Paris", "paris") == 1.0
    assert token_f1("the cat sat", "the cat sat") == 1.0
    assert 0 < token_f1("the cat", "the cat sat") < 1
    assert rouge_l("a b c d", "a b c d") == 1.0


def test_pass_at_k():
    assert pass_at_k(5, 0, 1) == 0.0          # no correct samples
    assert pass_at_k(5, 5, 1) == 1.0          # all correct
    assert 0 < pass_at_k(10, 1, 5) < 1


def test_llm_judge_proxies():
    assert rubric_score("", reference="x") == 1
    assert rubric_score("the kv cache stores tensors", reference="the kv cache stores tensors") >= 4
    assert pairwise_judge("the capital is Paris", "bananas", reference="the capital is Paris") == "A"


# --- offline eval --------------------------------------------------------
def _dataset():
    return EvalDataset("qa", "v1", [
        EvalSample("s1", "capital of France?", "Paris", "geography"),
        EvalSample("s2", "2+2?", "4", "math"),
    ])


def test_offline_runner_overall_and_by_capability():
    ds = _dataset()
    runner = OfflineEvalRunner({"em": exact_match, "f1": token_f1})
    answers = {"capital of France?": "Paris", "2+2?": "5"}
    report = runner.run(ds, lambda p: answers[p])
    assert report.overall["em"] == 0.5
    assert "geography" in report.by_capability and "math" in report.by_capability
    assert report.by_capability["geography"]["em"] == 1.0


def test_contamination_check():
    ds = _dataset()
    contaminated = ds.contamination_check(["the answer is Paris of course"])
    assert "s1" in contaminated


def test_dataset_fingerprint_changes():
    a = _dataset().fingerprint()
    ds2 = _dataset(); ds2.samples[0].reference = "London"
    assert a != ds2.fingerprint()


def test_bootstrap_ci_and_regression():
    base = [1.0] * 50
    cand_same = [1.0] * 50
    cand_worse = [0.0] * 50
    lo, hi = bootstrap_ci([1.0, 1.0, 0.0, 1.0])
    assert lo <= hi
    assert detect_regression(base, cand_worse) is True
    assert detect_regression(base, cand_same) is False


# --- experiments ---------------------------------------------------------
def test_variant_assignment_deterministic_and_split():
    assert assign_variant("u1", ["A", "B"]) == assign_variant("u1", ["A", "B"])
    counts = {"A": 0, "B": 0}
    for i in range(2000):
        counts[assign_variant(f"user{i}", ["A", "B"])] += 1
    assert 800 < counts["A"] < 1200            # roughly balanced 50/50


def test_two_proportion_ztest_detects_difference():
    z, p = two_proportion_ztest(500, 1000, 700, 1000)
    assert p < 0.01                            # 50% vs 70% is significant


def test_welch_and_abtest():
    ab = ABTest(["control", "treat"])
    for i in range(1000):
        v = assign_variant(f"u{i}", ["control", "treat"])
        ab.record(f"u{i}", 1.0 if v == "treat" else 0.5)
    assert ab.mean("treat") > ab.mean("control")
    assert ab.guardrail_breached("control", metric_floor=0.9) is True


def test_shadow_comparator():
    sc = ShadowComparator()
    sc.compare("Paris", "Paris", shadow_request_cost=0.001)
    sc.compare("Paris", "London", shadow_request_cost=0.001)
    assert sc.agreement_rate() == 0.5
    assert sc.shadow_cost > 0


def test_canary_advances_and_rolls_back():
    c = CanaryController(stages=[0.05, 0.5, 1.0], rollback_delta=0.05)
    assert c.evaluate(control_metric=0.8, canary_metric=0.81) == "advance"
    assert c.traffic == 0.5
    assert c.evaluate(control_metric=0.8, canary_metric=0.5) == "rollback"


# --- observability -------------------------------------------------------
def test_percentiles_and_error_rate():
    mc = MetricsCollector()
    for i in range(100):
        mc.record(RequestMetric(latency_ms=float(i + 1), model="m", cost_usd=0.001))
    mc.record(RequestMetric(latency_ms=10, error="timeout"))
    pct = mc.latency_percentiles()
    assert pct["p50"] <= pct["p95"] <= pct["p99"]
    assert 0 < mc.error_rate() < 0.05


def test_cost_by_dimension_and_anomaly():
    mc = MetricsCollector()
    for _ in range(20):
        mc.record(RequestMetric(latency_ms=5, model="small", cost_usd=0.001))
    for _ in range(20):
        mc.record(RequestMetric(latency_ms=5, model="large", cost_usd=0.05))
    costs = mc.cost_by("model")
    assert costs["large"] > costs["small"]
    assert mc.cost_anomaly() is True


def test_psi_drift():
    ref = [float(i % 10) for i in range(1000)]
    same = [float(i % 10) for i in range(1000)]
    shifted = [float(5 + i % 5) for i in range(1000)]
    assert population_stability_index(ref, same) < 0.1
    assert population_stability_index(ref, shifted) > 0.25


def test_trace_spans():
    tr = Trace("t1")
    sp = tr.start_span("retrieve", k="4")
    time.sleep(0.001)
    tr.end_span(sp)
    sp2 = tr.start_span("generate")
    tr.end_span(sp2)
    assert tr.step_count() == 2
    assert tr.total_ms() >= 0
