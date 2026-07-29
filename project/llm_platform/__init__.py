"""llm_platform -- a minimal, runnable LLM serving + RAG platform.

This package is the single coherent project that runs through the
"LLM System Design" ebook. Each chapter explains one module:

    estimation.py   -> Ch.1  System design thinking / capacity planning
    pipeline.py     -> Ch.2  End-to-end request lifecycle
    app.py          -> Ch.2  Sync / async / streaming delivery
    router.py       -> Ch.3  Model strategy: routing + cascading
    retrieval.py    -> Part IV  RAG retrieval
    cache.py        -> Part IX  Caching architecture
    guardrails.py   -> Part VI  Safety pipelines
    eval.py         -> Part VIII Evaluation

The core (router, cache, retrieval, guardrails, estimation, eval, pipeline)
depends only on the Python standard library so it runs anywhere. ``app.py``
adds an optional FastAPI surface, and the inference backends can optionally
call a real OpenAI-compatible endpoint via httpx.
"""

from .config import ModelSpec, MODEL_REGISTRY, Settings
from .schemas import ChatRequest, ChatResponse, Telemetry
from .pipeline import LLMPipeline

__all__ = [
    "ModelSpec",
    "MODEL_REGISTRY",
    "Settings",
    "ChatRequest",
    "ChatResponse",
    "Telemetry",
    "LLMPipeline",
]

__version__ = "0.1.0"
