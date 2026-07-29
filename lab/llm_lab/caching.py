"""Semantic response cache with LangChain embeddings (Chapter 32).

Returns a stored answer when a new query is embedding-similar to a previous one,
collapsing paraphrases into a single model call. Uses the same embeddings backend
(Ollama nomic-embed-text or offline hash) as retrieval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config import get_embeddings


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class SemanticCache:
    threshold: float = 0.9
    backend: Optional[str] = None
    _entries: List[Tuple[List[float], str, str]] = field(default_factory=list)  # (emb, query, answer)

    def __post_init__(self):
        self.embeddings = get_embeddings(backend=self.backend)

    def lookup(self, query: str) -> Optional[dict]:
        if not self._entries:
            return None
        q = self.embeddings.embed_query(query)
        best_score, best = 0.0, None
        for emb, orig, answer in self._entries:
            s = _cosine(q, emb)
            if s > best_score:
                best_score, best = s, (orig, answer)
        if best and best_score >= self.threshold:
            return {"hit": True, "score": round(best_score, 4),
                    "matched_query": best[0], "answer": best[1]}
        return {"hit": False, "score": round(best_score, 4)}

    def put(self, query: str, answer: str) -> None:
        self._entries.append((self.embeddings.embed_query(query), query, answer))

    def __len__(self) -> int:
        return len(self._entries)
