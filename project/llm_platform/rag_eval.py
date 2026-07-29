"""RAG evaluation metrics (Chapter 16).

Two halves: retrieval metrics (did we fetch the right chunks?) and generation
metrics (did the answer stay grounded in them?). All pure and testable so they
can run as a CI gate on every RAG change.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence

from .embeddings import tokenize


# --- Retrieval metrics ---------------------------------------------------
def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(set(retrieved[:k]) & rel) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & set(relevant)) / len(top)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Mean reciprocal rank of the first relevant hit."""
    rel = set(relevant)
    for i, doc in enumerate(retrieved):
        if doc in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevance: Dict[str, int], k: int) -> float:
    """Normalized discounted cumulative gain with graded relevance."""
    dcg = sum((2 ** relevance.get(doc, 0) - 1) / math.log2(i + 2)
              for i, doc in enumerate(retrieved[:k]))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# --- Generation metrics (lexical proxies for LLM-judge scores) -----------
_SENT = re.compile(r"(?<=[.!?])\s+")


def faithfulness(answer: str, context: str) -> float:
    """Fraction of the answer's content words supported by the context.

    A proxy for grounding: a real system uses an LLM judge or NLI model to check
    each claim against the context, but this lexical version captures the signal
    and is dependency-free for testing.
    """
    ctx = set(tokenize(context))
    content = [w for w in tokenize(answer) if len(w) > 3]
    if not content:
        return 1.0
    return sum(1 for w in content if w in ctx) / len(content)


def answer_relevance(answer: str, query: str) -> float:
    """How much the answer addresses the query terms."""
    q = set(tokenize(query))
    a = set(tokenize(answer))
    return len(q & a) / len(q) if q else 0.0


def context_relevance(query: str, context: str) -> float:
    """How much of the query is covered by the retrieved context."""
    q = set(tokenize(query))
    c = set(tokenize(context))
    return len(q & c) / len(q) if q else 0.0


def hallucination_rate(answer: str, context: str, threshold: float = 0.5) -> float:
    """1 - faithfulness, surfaced as a first-class alarm metric."""
    return round(max(0.0, 1.0 - faithfulness(answer, context)), 3)
