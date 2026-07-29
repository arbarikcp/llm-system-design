"""End-to-end RAG: offline indexing and the online query engine (Chapter 13).

`RAGIndex` is the offline pipeline -- parse, chunk, embed, index. `RAGQueryEngine`
is the online pipeline -- (optionally reformulate) query, hybrid-retrieve, rerank,
pack context, generate with citations. It composes the chunking, hybrid-retrieval,
inference, and guardrail modules into the canonical RAG flow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .chunking import Chunk, recursive_split, structure_aware_chunks, fixed_size_chunks
from .embeddings import HashingEmbedder
from .retrieval_hybrid import HybridRetriever, CrossEncoderReranker
from .inference import InferenceBackend, MockBackend, Completion
from .config import MODEL_REGISTRY, DEFAULT_MODEL
from .estimation import estimate_tokens

_CHUNKERS = {
    "fixed": fixed_size_chunks,
    "recursive": recursive_split,
    "structure": structure_aware_chunks,
}


@dataclass
class RAGIndex:
    """Offline indexing pipeline: documents -> chunks -> hybrid index."""
    embedder: HashingEmbedder = field(default_factory=HashingEmbedder)
    _chunks: List[Chunk] = field(default_factory=list)
    _retriever: Optional[HybridRetriever] = None

    def add_document(self, source: str, text: str, *, strategy: str = "recursive",
                     **kwargs) -> int:
        chunker = _CHUNKERS[strategy]
        if strategy == "structure":
            new = chunker(text, source=source)
        else:
            new = chunker(text, source=source, **kwargs)
        self._chunks.extend(new)
        self._retriever = None          # invalidate; rebuild lazily
        return len(new)

    def build(self) -> "RAGIndex":
        docs = [{"doc_id": c.doc_id, "text": c.text, "metadata": c.metadata}
                for c in self._chunks]
        self._retriever = HybridRetriever(docs, embedder=self.embedder)
        return self

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self.build()
        return self._retriever

    def __len__(self) -> int:
        return len(self._chunks)


@dataclass
class RAGResult:
    answer: str
    citations: List[str]
    contexts: List[dict]
    confidence: float


class RAGQueryEngine:
    """Online query pipeline over a built RAGIndex."""

    SYSTEM = ("Answer using ONLY the provided context. Cite chunk ids in [brackets]. "
              "If the context is insufficient, say so.")

    def __init__(self, index: RAGIndex, *, reranker: Optional[CrossEncoderReranker] = None,
                 backend: Optional[InferenceBackend] = None, model: str = DEFAULT_MODEL):
        self.index = index
        self.reranker = reranker or CrossEncoderReranker(index.embedder)
        self.backend = backend or MockBackend()
        self.model = MODEL_REGISTRY[model]

    def retrieve(self, query: str, *, k: int = 8, top_n: int = 4) -> List[dict]:
        """Hybrid-retrieve a shortlist, then rerank down to the top_n."""
        shortlist = self.index.retriever.search(query, k=k)
        return self.reranker.rerank(query, shortlist, top_k=top_n)

    def _build_prompt(self, query: str, contexts: List[dict]) -> str:
        ctx = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in contexts)
        return f"{self.SYSTEM}\n\n# Context\n{ctx}\n\n# Question\n{query}\n\n# Answer\n"

    def answer(self, query: str, *, k: int = 8, top_n: int = 4,
               max_tokens: int = 256) -> RAGResult:
        contexts = self.retrieve(query, k=k, top_n=top_n)
        prompt = self._build_prompt(query, contexts)
        comp: Completion = self.backend.generate(
            prompt, model=self.model, max_tokens=max_tokens, temperature=0.1)
        citations = [c["doc_id"] for c in contexts]
        return RAGResult(comp.text, citations, contexts, round(comp.confidence, 3))


def hyde_reformulate(query: str) -> str:
    """HyDE-style query expansion (illustrative).

    Hypothetical Document Embeddings: generate a hypothetical answer and embed
    *that* to retrieve, since a fake answer is often closer to real passages than
    the raw question. Here we expand the query deterministically as a stand-in.
    """
    return f"{query} A detailed factual explanation: {query.rstrip('?')}."
