"""Hybrid retrieval: BM25 + dense, fused with RRF, then reranked (Chapter 14).

Dense (semantic) retrieval and sparse (lexical, BM25) retrieval fail in
different ways; fusing them with Reciprocal Rank Fusion (RRF) is a strong,
tuning-free default. A cross-encoder reranker then reorders the shortlist for
precision. Implemented with the standard library so it runs anywhere.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

from .embeddings import HashingEmbedder, cosine_similarity, tokenize


class BM25:
    """Okapi BM25 lexical scorer."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = corpus_tokens
        self.N = len(corpus_tokens)
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.df: Counter = Counter()
        for d in corpus_tokens:
            self.df.update(set(d))
        self.tf = [Counter(d) for d in corpus_tokens]

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: Sequence[str], idx: int) -> float:
        s, dl = 0.0, self.doc_len[idx]
        for t in query_tokens:
            tf = self.tf[idx].get(t, 0)
            if not tf:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s += self.idf(t) * (tf * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, k: int = 10) -> List[tuple]:
        q = tokenize(query)
        scored = [(i, self.score(q, i)) for i in range(self.N)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scored if s > 0][:k]


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[int]:
    """Fuse multiple ranked id-lists. RRF score = sum 1/(k + rank).

    RRF needs no score normalization across retrievers and is robust to their
    different score scales -- which is exactly why it is the default fusion.
    """
    scores: Counter = Counter()
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] += 1.0 / (k + rank + 1)
    return [doc for doc, _ in scores.most_common()]


class CrossEncoderReranker:
    """Stand-in for a trained cross-encoder reranker.

    A real reranker jointly encodes (query, passage) and scores relevance. Here
    we blend lexical overlap with embedding similarity as a deterministic proxy,
    behind the same ``rerank`` interface so you can swap in BGE/Cohere Rerank.
    """

    def __init__(self, embedder: HashingEmbedder = None):
        self.embedder = embedder or HashingEmbedder()

    def score(self, query: str, text: str) -> float:
        q, d = set(tokenize(query)), set(tokenize(text))
        lexical = len(q & d) / len(q) if q else 0.0
        sem = cosine_similarity(self.embedder.embed(query), self.embedder.embed(text))
        return 0.5 * lexical + 0.5 * max(0.0, sem)

    def rerank(self, query: str, docs: List[dict], top_k: int = None) -> List[dict]:
        ranked = sorted(docs, key=lambda d: self.score(query, d["text"]), reverse=True)
        return ranked[:top_k] if top_k else ranked


class HybridRetriever:
    """Dense + BM25 retrieval fused with RRF over a fixed document set."""

    def __init__(self, docs: List[dict], embedder: HashingEmbedder = None):
        # docs: list of {"doc_id", "text", ...}
        self.docs = docs
        self.embedder = embedder or HashingEmbedder()
        self.tokens = [tokenize(d["text"]) for d in docs]
        self.bm25 = BM25(self.tokens)
        self.embs = [self.embedder.embed(d["text"]) for d in docs]

    def dense_search(self, query: str, k: int) -> List[int]:
        q = self.embedder.embed(query)
        order = sorted(range(len(self.docs)),
                       key=lambda i: cosine_similarity(q, self.embs[i]), reverse=True)
        return order[:k]

    def search(self, query: str, k: int = 5, rrf_k: int = 60) -> List[dict]:
        dense = self.dense_search(query, k * 2)
        sparse = [i for i, _ in self.bm25.search(query, k * 2)]
        fused = reciprocal_rank_fusion([dense, sparse], k=rrf_k)[:k]
        return [self.docs[i] for i in fused]
