"""Two-tier response cache (Part IX: Caching Architecture).

Tier 1 -- exact cache: hash of the normalized prompt, O(1) lookup, TTL'd.
Tier 2 -- semantic cache: embed the query and return a stored answer if a
prior query is similar enough. Semantic caching is what turns "10 ways to ask
the same FAQ" into one model call.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .embeddings import HashingEmbedder, cosine_similarity


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def prompt_hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode()).hexdigest()


@dataclass
class _Entry:
    answer: str
    citations: List[str]
    embedding: List[float]
    expires_at: float


@dataclass
class ResponseCache:
    ttl_seconds: int = 900
    semantic_threshold: float = 0.92
    embedder: HashingEmbedder = field(default_factory=HashingEmbedder)
    _exact: Dict[str, _Entry] = field(default_factory=dict)

    def _alive(self, e: _Entry) -> bool:
        return e.expires_at > time.time()

    def get(self, query: str) -> Optional[Tuple[str, str, List[str]]]:
        """Return (kind, answer, citations) or None.

        ``kind`` is "exact" or "semantic" for telemetry.
        """
        key = prompt_hash(query)
        e = self._exact.get(key)
        if e and self._alive(e):
            return "exact", e.answer, e.citations

        # Semantic fallback: linear scan is fine for a demo; back this with an
        # ANN index (FAISS/pgvector) keyed on the query embedding at scale.
        q = self.embedder.embed(query)
        best_score, best = 0.0, None
        for entry in self._exact.values():
            if not self._alive(entry):
                continue
            s = cosine_similarity(q, entry.embedding)
            if s > best_score:
                best_score, best = s, entry
        if best and best_score >= self.semantic_threshold:
            return "semantic", best.answer, best.citations
        return None

    def put(self, query: str, answer: str, citations: List[str]) -> None:
        self._exact[prompt_hash(query)] = _Entry(
            answer=answer,
            citations=list(citations),
            embedding=self.embedder.embed(query),
            expires_at=time.time() + self.ttl_seconds,
        )

    def __len__(self) -> int:
        return sum(1 for e in self._exact.values() if self._alive(e))
