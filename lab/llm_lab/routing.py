"""Model routing + cascade as an LCEL chain (Chapters 2-3).

Classifies a query by difficulty, routes it to a cheap or strong tier, and runs
a LangChain ``prompt | model | parser`` chain -- reporting tokens, latency, and
estimated cost so the trade-offs are visible in the UI.
"""
from __future__ import annotations

import re
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import get_chat_model, resolve_backend, OLLAMA_LLM
from .telemetry import Telemetry, estimate_tokens, request_cost_usd, timed

_HARD = re.compile(r"\b(prove|derive|analy[sz]e|debug|architect|why|step by step|reason|compare|design)\b", re.I)
_SIMPLE = re.compile(r"\b(define|what is|list|translate|summari[sz]e|yes or no|who|when)\b", re.I)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a concise, helpful assistant. Answer the user directly."),
    ("human", "{query}"),
])


def classify(query: str) -> tuple[str, str]:
    """Return (tier, reason): 'small' for easy queries, 'large' for hard ones."""
    if _HARD.search(query):
        return "large", "reasoning_intent"
    if _SIMPLE.search(query) and len(query) < 200:
        return "small", "simple_intent"
    return "small", "default"


def route_and_answer(query: str, *, backend: Optional[str] = None) -> dict:
    tier, reason = classify(query)
    temperature = 0.2 if tier == "small" else 0.4
    model = get_chat_model(temperature=temperature, backend=backend)
    chain = _PROMPT | model | StrOutputParser()      # LCEL pipeline

    tel = Telemetry(tier=tier)
    with timed(tel):
        answer = chain.invoke({"query": query})
    tel.input_tokens = estimate_tokens(query)
    tel.output_tokens = estimate_tokens(answer)
    price_tier = OLLAMA_LLM if resolve_backend(backend) == "ollama" else tier
    tel.cost_usd = request_cost_usd(price_tier, tel.input_tokens, tel.output_tokens)
    return {"query": query, "tier": tier, "reason": reason,
            "answer": answer, "telemetry": tel.as_dict()}
