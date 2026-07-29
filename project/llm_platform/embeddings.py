"""Dependency-free embedding + similarity utilities.

In production you call a real embedding model (e.g. a bi-encoder). For a
runnable, offline book project we use a deterministic hashing embedder that
maps tokens into a fixed-dimension bag-of-features vector. It is good enough
to demonstrate semantic caching and RAG retrieval mechanics end-to-end.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List, Sequence

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


class HashingEmbedder:
    """Hashing vectorizer with L2 normalisation.

    Swap this for ``SentenceTransformer`` or a managed embedding API by
    implementing the same ``embed`` method -- the rest of the platform is
    written against the interface, not the implementation.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    # Inputs are pre-normalised, so the dot product is already cosine sim.
    return max(-1.0, min(1.0, dot))
