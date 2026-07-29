"""Automated evaluation metrics (Chapter 27).

Reference-based metrics (exact match, token-F1, ROUGE-L), code/sampling metrics
(pass@k with the unbiased estimator), and LLM-as-judge stand-ins (rubric scoring
and pairwise comparison). The judge functions are deterministic proxies; in
production they call a strong model with a scoring rubric.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional, Sequence

_WORD = re.compile(r"[a-z0-9]+")


def _toks(s: str) -> List[str]:
    return _WORD.findall(s.lower())


def exact_match(pred: str, ref: str) -> float:
    return float(pred.strip().lower() == ref.strip().lower())


def contains(pred: str, substring: str) -> float:
    return float(substring.lower() in pred.lower())


def token_f1(pred: str, ref: str) -> float:
    p, r = Counter(_toks(pred)), Counter(_toks(ref))
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    a, b = _toks(pred), _toks(ref)
    if not a or not b:
        return 0.0
    lcs = _lcs_len(a, b)
    precision, recall = lcs / len(a), lcs / len(b)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    n, c = num_samples, num_correct
    if n - c < k:
        return 1.0
    prob_all_fail = 1.0
    for i in range(k):
        prob_all_fail *= (n - c - i) / (n - i)
    return 1.0 - prob_all_fail


# --- LLM-as-judge proxies ------------------------------------------------
def rubric_score(answer: str, *, reference: str = "", max_score: int = 5) -> int:
    """Deterministic 1..max_score rubric proxy (an LLM judge in production)."""
    if not answer.strip():
        return 1
    if reference:
        sim = max(token_f1(answer, reference), rouge_l(answer, reference))
        return max(1, min(max_score, 1 + round(sim * (max_score - 1))))
    # Reference-free: reward non-trivial, well-formed length.
    n = len(_toks(answer))
    return max(1, min(max_score, 1 + (n >= 5) + (n >= 20) + (n >= 50)))


def pairwise_judge(answer_a: str, answer_b: str, *, reference: str) -> str:
    """Return 'A', 'B', or 'tie' for which answer better matches the reference."""
    sa = max(token_f1(answer_a, reference), rouge_l(answer_a, reference))
    sb = max(token_f1(answer_b, reference), rouge_l(answer_b, reference))
    if abs(sa - sb) < 1e-6:
        return "tie"
    return "A" if sa > sb else "B"
