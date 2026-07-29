"""Multi-level caching architecture (Chapter 32).

LLM caching is layered: L2 prefix cache (shared-prefix KV reuse), L3 response
cache (exact + semantic), and an embedding cache. This module provides an LRU
cache with TTL, prefix-cache savings math, and a MultiLevelCache that checks an
exact layer then a semantic layer, tracking per-level hit rates.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .cache import ResponseCache, prompt_hash
from .embeddings import HashingEmbedder


@dataclass
class LRUCache:
    """Capacity-bounded LRU cache with optional per-entry TTL and hit stats."""
    capacity: int = 1024
    ttl_seconds: Optional[float] = None
    _store: "OrderedDict[str, Tuple[Any, float]]" = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def _alive(self, expires_at: float, now: float) -> bool:
        return self.ttl_seconds is None or expires_at > now

    def get(self, key: str, *, now: Optional[float] = None) -> Optional[Any]:
        now = now if now is not None else time.time()
        entry = self._store.get(key)
        if entry is None or not self._alive(entry[1], now):
            if entry is not None:
                del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)        # mark most-recently-used
        self.hits += 1
        return entry[0]

    def put(self, key: str, value: Any, *, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        expires = now + self.ttl_seconds if self.ttl_seconds else float("inf")
        self._store[key] = (value, expires)
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)  # evict least-recently-used

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._store)


class EmbeddingCache:
    """Cache query embeddings to avoid recomputing them for hot queries."""

    def __init__(self, *, capacity: int = 4096, ttl_seconds: float = 3600,
                 embedder: Optional[HashingEmbedder] = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.cache = LRUCache(capacity=capacity, ttl_seconds=ttl_seconds)

    def embed(self, text: str) -> List[float]:
        key = prompt_hash(text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        vec = self.embedder.embed(text)
        self.cache.put(key, vec)
        return vec


def prefix_cache_savings(*, shared_prefix_tokens: int, num_requests: int) -> int:
    """Prompt tokens saved by computing a shared prefix's KV once and reusing it.

    Without prefix caching every request prefills the shared prefix; with it, the
    prefix is computed once and reused, saving (num_requests - 1) * prefix tokens.
    """
    return max(0, (num_requests - 1) * shared_prefix_tokens)


@dataclass
class MultiLevelCache:
    """L-exact (LRU) in front of an L-semantic (ResponseCache) layer."""
    exact: LRUCache = field(default_factory=lambda: LRUCache(capacity=4096))
    semantic: ResponseCache = field(default_factory=lambda: ResponseCache(semantic_threshold=0.92))
    level_hits: Dict[str, int] = field(default_factory=lambda: {"exact": 0, "semantic": 0, "miss": 0})

    def get(self, query: str) -> Optional[Tuple[str, str, List[str]]]:
        """Return (level, answer, citations) or None."""
        exact = self.exact.get(prompt_hash(query))
        if exact is not None:
            self.level_hits["exact"] += 1
            return ("exact", exact[0], exact[1])
        sem = self.semantic.get(query)
        if sem is not None:
            kind, answer, citations = sem
            self.level_hits["semantic"] += 1
            return ("semantic", answer, citations)
        self.level_hits["miss"] += 1
        return None

    def put(self, query: str, answer: str, citations: List[str]) -> None:
        self.exact.put(prompt_hash(query), (answer, list(citations)))
        self.semantic.put(query, answer, citations)

    def hit_rate(self) -> float:
        total = sum(self.level_hits.values())
        hits = self.level_hits["exact"] + self.level_hits["semantic"]
        return hits / total if total else 0.0
