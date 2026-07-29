"""Offline evaluation harness with an LLM-as-judge (Chapters 27-29).

Runs the seed eval cases through the RAG pipeline, checks a keyword expectation,
scores faithfulness against the retrieved context, and asks the chat model to
judge each answer -- aggregating pass rate, cost, and latency for a CI-style gate.
"""
from __future__ import annotations

import re
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import get_chat_model
from .rag import RAGPipeline
from .seed import SEED_EVAL_CASES

_JUDGE = ChatPromptTemplate.from_messages([
    ("system", "You are a strict grader. Reply with a single integer 1-5 for how well the "
               "ANSWER addresses the QUESTION using the CONTEXT. Only the number."),
    ("human", "QUESTION: {q}\nCONTEXT: {c}\nANSWER: {a}\nScore (1-5):"),
])


def _faithfulness(answer: str, context: str) -> float:
    words = [w for w in re.findall(r"[a-z0-9]+", answer.lower()) if len(w) > 3]
    ctx = set(re.findall(r"[a-z0-9]+", context.lower()))
    return round(sum(1 for w in words if w in ctx) / len(words), 3) if words else 1.0


def _extract_score(text: str) -> int:
    m = re.search(r"[1-5]", text or "")
    return int(m.group(0)) if m else 3


def run_eval(cases: Optional[List[dict]] = None, *, backend: Optional[str] = None,
             use_judge: bool = True) -> dict:
    cases = cases or SEED_EVAL_CASES
    rag = RAGPipeline(backend=backend, k=3)
    judge_model = get_chat_model(temperature=0.0, backend=backend)
    judge_chain = _JUDGE | judge_model | StrOutputParser()

    rows, passed, total_cost, latencies, judge_scores = [], 0, 0.0, [], []
    for c in cases:
        res = rag.answer(c["query"])
        answer = res["answer"]
        context = "\n".join(x["text"] for x in res["contexts"])
        ok = c.get("expect", "").lower() in answer.lower()
        passed += int(ok)
        total_cost += res["telemetry"]["cost_usd"]
        latencies.append(res["telemetry"]["latency_ms"])
        faith = _faithfulness(answer, context)
        score = _extract_score(judge_chain.invoke(
            {"q": c["query"], "c": context[:600], "a": answer})) if use_judge else None
        if score is not None:
            judge_scores.append(score)
        rows.append({"query": c["query"], "passed": ok, "faithfulness": faith,
                     "judge_score": score, "citations": res["citations"],
                     "latency_ms": res["telemetry"]["latency_ms"]})

    n = len(cases) or 1
    return {
        "pass_rate": round(passed / n, 3),
        "avg_faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 3),
        "avg_judge_score": round(sum(judge_scores) / len(judge_scores), 2) if judge_scores else None,
        "total_cost_usd": round(total_cost, 6),
        "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0.0,
        "rows": rows,
    }
