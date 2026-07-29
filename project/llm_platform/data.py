"""Tiny seed knowledge base so the demo and tests retrieve real context."""
from __future__ import annotations

from .retrieval import InMemoryVectorStore

KNOWLEDGE = [
    {"doc_id": "kv-cache", "source": "docs/serving.md",
     "text": "The KV cache stores key and value tensors for every token so "
             "attention is not recomputed each step. Its size grows linearly "
             "with sequence length and dominates serving memory."},
    {"doc_id": "gqa", "source": "docs/attention.md",
     "text": "Grouped-query attention shares key/value heads across query "
             "heads, shrinking the KV cache and raising serving throughput "
             "with little quality loss."},
    {"doc_id": "cascade", "source": "docs/routing.md",
     "text": "Model cascading runs a cheap model first and escalates to a "
             "stronger model only when confidence is low, cutting cost while "
             "protecting quality on hard queries."},
    {"doc_id": "rag", "source": "docs/rag.md",
     "text": "Retrieval-augmented generation grounds answers in retrieved "
             "documents, reducing hallucination and letting you update "
             "knowledge without retraining the model."},
    {"doc_id": "semantic-cache", "source": "docs/caching.md",
     "text": "A semantic cache returns a stored answer when a new query is "
             "embedding-similar to a previous one, collapsing many "
             "paraphrases of the same question into a single model call."},
]


def seed_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add_texts(KNOWLEDGE)
    return store
