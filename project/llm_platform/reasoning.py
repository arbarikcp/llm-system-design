"""Test-time compute: best-of-N, self-consistency, thinking budgets (Chapter 47).

Spending more compute at inference (sampling multiple paths, voting, verifying)
to raise answer quality without retraining. Pure, testable helpers.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def best_of_n(samples: Sequence[T], scorer: Callable[[T], float]) -> Tuple[T, float]:
    """Pick the highest-scoring sample (a reward model scores in production)."""
    if not samples:
        raise ValueError("need at least one sample")
    best = max(samples, key=scorer)
    return best, scorer(best)


def self_consistency(answers: Sequence[T]) -> Tuple[T, float]:
    """Majority vote over sampled answers; returns (answer, agreement fraction)."""
    if not answers:
        raise ValueError("need at least one answer")
    counts = Counter(answers)
    answer, votes = counts.most_common(1)[0]
    return answer, votes / len(answers)


def best_of_n_expected_pass(p_correct: float, n: int) -> float:
    """Probability at least one of n independent samples is correct: 1-(1-p)^n.

    Shows diminishing returns: the marginal gain of each extra sample shrinks.
    """
    p = max(0.0, min(1.0, p_correct))
    return 1.0 - (1.0 - p) ** n


def thinking_budget_tokens(difficulty: float, *, min_tokens: int = 256,
                           max_tokens: int = 8192) -> int:
    """Allocate a reasoning/thinking token budget by estimated difficulty (0..1).

    Hard requests get more test-time compute; easy ones get little, so budget is
    spent where it changes the answer.
    """
    d = max(0.0, min(1.0, difficulty))
    return int(min_tokens + d * (max_tokens - min_tokens))
