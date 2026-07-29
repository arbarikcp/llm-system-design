"""Offline evaluation pipeline (Chapter 27).

Versioned evaluation datasets with contamination checking, a runner that scores
predictions with pluggable metrics (overall and per-capability), bootstrap
confidence intervals, and regression detection against a baseline.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .metrics import _toks


@dataclass
class EvalSample:
    sample_id: str
    prompt: str
    reference: str
    capability: str = "general"


@dataclass
class EvalDataset:
    name: str
    version: str
    samples: List[EvalSample] = field(default_factory=list)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for s in self.samples:
            h.update(f"{s.sample_id}|{s.prompt}|{s.reference}".encode())
        return h.hexdigest()[:16]

    def contamination_check(self, training_texts: List[str]) -> List[str]:
        """Return ids of eval samples whose reference appears in training data.

        Test-set contamination inflates scores and invalidates the eval, so this
        runs as a release gate (Chapter 6).
        """
        norm_train = {" ".join(_toks(t)) for t in training_texts}
        contaminated = []
        for s in self.samples:
            ref = " ".join(_toks(s.reference))
            if any(ref and ref in t for t in norm_train):
                contaminated.append(s.sample_id)
        return contaminated


@dataclass
class EvalReport:
    dataset: str
    version: str
    overall: Dict[str, float]
    by_capability: Dict[str, Dict[str, float]]
    per_sample: List[dict] = field(default_factory=list)


class OfflineEvalRunner:
    def __init__(self, metric_fns: Dict[str, Callable[[str, str], float]]) -> None:
        self.metric_fns = metric_fns

    def run(self, dataset: EvalDataset,
            predict_fn: Callable[[str], str]) -> EvalReport:
        per_sample: List[dict] = []
        agg: Dict[str, List[float]] = {m: [] for m in self.metric_fns}
        cap_agg: Dict[str, Dict[str, List[float]]] = {}

        for s in dataset.samples:
            pred = predict_fn(s.prompt)
            row = {"id": s.sample_id, "capability": s.capability, "prediction": pred}
            for name, fn in self.metric_fns.items():
                score = fn(pred, s.reference)
                row[name] = score
                agg[name].append(score)
                cap_agg.setdefault(s.capability, {m: [] for m in self.metric_fns})
                cap_agg[s.capability][name].append(score)
            per_sample.append(row)

        overall = {m: round(_mean(v), 4) for m, v in agg.items()}
        by_cap = {cap: {m: round(_mean(v), 4) for m, v in metrics.items()}
                  for cap, metrics in cap_agg.items()}
        return EvalReport(dataset.name, dataset.version, overall, by_cap, per_sample)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(scores: List[float], *, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap confidence interval for a metric's mean."""
    if not scores:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    m = len(scores)
    for _ in range(n_boot):
        sample = [scores[rng.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (round(lo, 4), round(hi, 4))


def detect_regression(baseline: List[float], candidate: List[float], *,
                      min_delta: float = 0.0, alpha: float = 0.05) -> bool:
    """True if the candidate is worse than baseline beyond noise.

    Uses the bootstrap CI of the difference; a regression is flagged when the
    upper bound of (candidate - baseline) is below ``-min_delta``.
    """
    if not baseline or not candidate:
        return False
    rng = random.Random(0)
    diffs = []
    nb, nc = len(baseline), len(candidate)
    for _ in range(1000):
        b = sum(baseline[rng.randrange(nb)] for _ in range(nb)) / nb
        c = sum(candidate[rng.randrange(nc)] for _ in range(nc)) / nc
        diffs.append(c - b)
    diffs.sort()
    upper = diffs[int((1 - alpha / 2) * 1000) - 1]
    return upper < -min_delta
