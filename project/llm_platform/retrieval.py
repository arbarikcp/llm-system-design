"""In-memory vector store and retriever (RAG core, Part IV).

Demonstrates the retrieval half of a RAG pipeline: index documents, embed a
query, return top-k by cosine similarity with a relevance floor. The
interface (``add``, ``search``) matches what you would back with FAISS,
pgvector, or a managed vector DB in production.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .embeddings import HashingEmbedder, cosine_similarity


@dataclass
class Document:
    doc_id: str
    text: str
    source: str = ""
    embedding: Optional[List[float]] = None


@dataclass
class ScoredDoc:
    document: Document
    score: float


@dataclass
class InMemoryVectorStore:
    embedder: HashingEmbedder = field(default_factory=HashingEmbedder)
    _docs: List[Document] = field(default_factory=list)

    def add(self, doc: Document) -> None:
        if doc.embedding is None:
            doc.embedding = self.embedder.embed(doc.text)
        self._docs.append(doc)

    def add_texts(self, items: List[dict]) -> None:
        for it in items:
            self.add(Document(doc_id=it["doc_id"], text=it["text"],
                              source=it.get("source", "")))

    def search(self, query: str, k: int = 4, min_score: float = 0.15) -> List[ScoredDoc]:
        q = self.embedder.embed(query)
        scored = [
            ScoredDoc(d, cosine_similarity(q, d.embedding or []))
            for d in self._docs
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return [s for s in scored if s.score >= min_score][:k]

    def __len__(self) -> int:
        return len(self._docs)


class Retriever:
    """Thin façade that returns context blocks + citations for the prompt."""

    def __init__(self, store: InMemoryVectorStore) -> None:
        self.store = store

    def retrieve(self, query: str, k: int = 4):
        hits = self.store.search(query, k=k)
        context = "\n\n".join(
            f"[{h.document.doc_id}] {h.document.text}" for h in hits
        )
        citations = [h.document.source or h.document.doc_id for h in hits]
        return context, citations, hits
