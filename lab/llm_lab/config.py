"""Backend selection: real Ollama models or a deterministic offline fallback.

The lab targets two backends behind one interface so it runs anywhere:

* ``ollama``  -> ChatOllama(llama3.1) + OllamaEmbeddings(nomic-embed-text).
* ``offline`` -> OfflineChatModel + HashEmbeddings (deterministic, no server).

Selection order: the ``LAB_BACKEND`` env var (``ollama`` | ``offline`` | ``auto``),
defaulting to ``auto`` which uses Ollama only if its server is reachable.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import SimpleChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

OLLAMA_LLM = os.getenv("LAB_LLM", "llama3.1")
OLLAMA_EMBED = os.getenv("LAB_EMBED", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# --------------------------------------------------------------------------- #
# Deterministic offline chat model (a real LangChain chat model, no server).
# --------------------------------------------------------------------------- #
class OfflineChatModel(SimpleChatModel):
    """A deterministic, context-aware chat model for offline demos.

    It is a genuine LangChain ``SimpleChatModel``: it reads the final prompt,
    extracts any ``# Context`` block, and returns a grounded-looking answer so
    the UI shows sensible output with no LLM server. Swap it for ChatOllama to
    get real generations -- the rest of the lab is unchanged.
    """

    @property
    def _llm_type(self) -> str:
        return "offline-heuristic"

    def _call(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> str:
        prompt = messages[-1].content if messages else ""
        if not isinstance(prompt, str):
            prompt = str(prompt)
        # Pull the question and any provided context out of the assembled prompt.
        question = _after(prompt, "# Question") or _after(prompt, "Question:") or prompt
        context = _between(prompt, "# Context", "# Question")
        if context:
            snippet = " ".join(context.split())[:220]
            return (f"Based on the retrieved context, here is a grounded answer to "
                    f"“{question.strip()[:80]}”: {snippet} "
                    f"[offline-heuristic model]")
        return (f"[offline-heuristic] Answer to: {question.strip()[:120]}. "
                f"Set LAB_BACKEND=ollama to generate with llama3.1.")


def _after(text: str, marker: str) -> str:
    idx = text.find(marker)
    return text[idx + len(marker):].strip() if idx >= 0 else ""


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i + len(start): j if j >= 0 else None].strip()


# --------------------------------------------------------------------------- #
# Deterministic offline embeddings (a real LangChain Embeddings, no server).
# --------------------------------------------------------------------------- #
class HashEmbeddings(Embeddings):
    """Hashing bag-of-words embedder with L2 normalisation.

    Deterministic and dependency-free -- good enough to demonstrate retrieval and
    semantic caching offline. Swap for OllamaEmbeddings(nomic-embed-text) for real
    semantic quality.
    """

    _WORD = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        import hashlib
        import math
        vec = [0.0] * self.dim
        for tok in self._WORD.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


# --------------------------------------------------------------------------- #
# Backend detection and factories.
# --------------------------------------------------------------------------- #
def _ollama_reachable(timeout: float = 0.8) -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def resolve_backend(explicit: Optional[str] = None) -> str:
    """Return 'ollama' or 'offline' from the explicit choice / env / reachability."""
    choice = (explicit or os.getenv("LAB_BACKEND", "auto")).lower()
    if choice == "ollama":
        return "ollama"
    if choice == "offline":
        return "offline"
    return "ollama" if _ollama_reachable() else "offline"


def backend_status(explicit: Optional[str] = None) -> dict:
    backend = resolve_backend(explicit)
    return {
        "backend": backend,
        "llm": OLLAMA_LLM if backend == "ollama" else "offline-heuristic",
        "embeddings": OLLAMA_EMBED if backend == "ollama" else "hash-embeddings",
        "ollama_reachable": _ollama_reachable(),
        "base_url": OLLAMA_BASE_URL,
    }


def get_chat_model(*, temperature: float = 0.1, backend: Optional[str] = None) -> BaseChatModel:
    if resolve_backend(backend) == "ollama":
        from langchain_ollama import ChatOllama  # lazy: avoids import cost/errors offline
        return ChatOllama(model=OLLAMA_LLM, temperature=temperature, base_url=OLLAMA_BASE_URL)
    return OfflineChatModel()


def get_embeddings(*, backend: Optional[str] = None) -> Embeddings:
    if resolve_backend(backend) == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=OLLAMA_EMBED, base_url=OLLAMA_BASE_URL)
    return HashEmbeddings()
