"""Tests for the Part IV RAG modules: chunking, hybrid retrieval, eval, rag."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_platform.chunking import (
    fixed_size_chunks, recursive_split, structure_aware_chunks,
)
from llm_platform.retrieval_hybrid import (
    BM25, reciprocal_rank_fusion, CrossEncoderReranker, HybridRetriever,
)
from llm_platform.rag_eval import (
    recall_at_k, precision_at_k, mrr, ndcg_at_k,
    faithfulness, answer_relevance, context_relevance, hallucination_rate,
)
from llm_platform.rag import RAGIndex, RAGQueryEngine, hyde_reformulate


CORPUS = [
    {"doc_id": "d0", "text": "The KV cache stores key and value tensors for every token."},
    {"doc_id": "d1", "text": "Grouped-query attention shrinks the KV cache and raises throughput."},
    {"doc_id": "d2", "text": "Reciprocal rank fusion combines dense and sparse retrieval rankings."},
    {"doc_id": "d3", "text": "BM25 is a lexical scoring function based on term frequency and inverse document frequency."},
]


# --- chunking ------------------------------------------------------------
def test_fixed_size_overlap():
    text = "abcdefghij" * 10           # 100 chars
    chunks = fixed_size_chunks(text, size=40, overlap=10)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 40 for c in chunks)
    assert chunks[0].doc_id.endswith("#0")


def test_recursive_respects_max():
    text = "Para one sentence. Another sentence here.\n\nSecond paragraph is longer " * 5
    chunks = recursive_split(text, max_chars=80)
    assert all(len(c.text) <= 80 or " " not in c.text for c in chunks)
    assert len(chunks) > 1


def test_structure_aware_sections():
    md = "# Intro\nHello world.\n\n## Details\nDeep content here.\n## More\nE ven more."
    chunks = structure_aware_chunks(md, source="doc")
    sections = {c.metadata["section"] for c in chunks}
    assert {"Intro", "Details", "More"} <= sections


# --- BM25 / RRF / rerank -------------------------------------------------
def test_bm25_ranks_lexical_match_first():
    bm = BM25([["the", "kv", "cache"], ["bm25", "lexical", "scoring"]])
    hits = bm.search("bm25 scoring", k=2)
    assert hits and hits[0][0] == 1


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert fused[0] in (1, 2)        # docs ranked high by both float to the top


def test_cross_encoder_reranks_relevant_first():
    rr = CrossEncoderReranker()
    docs = [{"doc_id": "d0", "text": "unrelated text about weather"},
            {"doc_id": "d1", "text": "BM25 term frequency scoring function"}]
    out = rr.rerank("BM25 scoring", docs, top_k=2)
    assert out[0]["doc_id"] == "d1"


def test_hybrid_retriever_finds_relevant():
    hr = HybridRetriever(CORPUS)
    res = hr.search("what shrinks the KV cache", k=2)
    ids = [d["doc_id"] for d in res]
    assert "d1" in ids


# --- metrics -------------------------------------------------------------
def test_recall_precision_mrr():
    retrieved = ["d2", "d0", "d1"]
    relevant = ["d1"]
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3
    assert mrr(retrieved, relevant) == 1 / 3


def test_ndcg_perfect_order_is_one():
    retrieved = ["a", "b", "c"]
    relevance = {"a": 3, "b": 2, "c": 1}
    assert abs(ndcg_at_k(retrieved, relevance, 3) - 1.0) < 1e-9


def test_faithfulness_and_hallucination():
    ctx = "the kv cache stores key and value tensors"
    grounded = faithfulness("the cache stores tensors", ctx)
    ungrounded = faithfulness("bananas grow in tropical orchards", ctx)
    assert grounded > ungrounded
    assert hallucination_rate("bananas grow in tropical orchards", ctx) > 0.5


def test_answer_and_context_relevance():
    assert answer_relevance("the kv cache stores tensors", "what does the kv cache store") >= 0.5
    assert context_relevance("kv cache", "the kv cache stores tensors") == 1.0


# --- end-to-end RAG ------------------------------------------------------
def test_rag_index_and_answer():
    idx = RAGIndex()
    idx.add_document("serving.md",
                     "The KV cache stores key/value tensors. "
                     "Grouped-query attention shrinks it and raises throughput.",
                     strategy="recursive", max_chars=60)
    idx.add_document("retrieval.md",
                     "Hybrid retrieval fuses BM25 and dense search with RRF.",
                     strategy="recursive", max_chars=60)
    idx.build()
    assert len(idx) >= 2
    engine = RAGQueryEngine(idx)
    result = engine.answer("what shrinks the KV cache?", k=4, top_n=2)
    assert result.answer
    assert len(result.citations) >= 1
    assert 0.0 <= result.confidence <= 1.0


def test_hyde_expands_query():
    assert len(hyde_reformulate("what is RRF?")) > len("what is RRF?")
