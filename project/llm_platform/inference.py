"""Inference backends behind a single interface.

``MockBackend`` is deterministic and offline so the whole platform runs with
zero credentials. ``OpenAICompatBackend`` calls any OpenAI-compatible
``/chat/completions`` endpoint (vLLM, TGI, or a vendor) when configured.

Both return a ``Completion`` carrying the text, token counts, and a
self-reported confidence used by the cascade router (Chapter 3).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

from .config import ModelSpec
from .estimation import estimate_tokens


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    confidence: float
    model: str


class InferenceBackend:
    def generate(self, prompt: str, *, model: ModelSpec, max_tokens: int,
                 temperature: float) -> Completion:
        raise NotImplementedError

    def stream(self, prompt: str, *, model: ModelSpec, max_tokens: int,
               temperature: float) -> Iterator[str]:
        raise NotImplementedError


class MockBackend(InferenceBackend):
    """Deterministic, offline stand-in for a real LLM.

    The "confidence" is derived from how much retrieved context the prompt
    contains -- richer context => higher confidence -- which lets us exercise
    the cascade-escalation logic without a real model.
    """

    def _answer(self, prompt: str, model: ModelSpec) -> str:
        digest = hashlib.sha1(prompt.encode()).hexdigest()[:8]
        return (f"[{model.name}] Based on the provided context, here is a "
                f"grounded answer (trace={digest}).")

    def _confidence(self, prompt: str, model: ModelSpec) -> float:
        has_ctx = "[" in prompt and "]" in prompt   # retrieved blocks present
        base = 0.45 + 0.35 * model.quality
        return min(0.99, base + (0.15 if has_ctx else -0.10))

    def generate(self, prompt, *, model, max_tokens, temperature) -> Completion:
        time.sleep(min(0.02, model.p50_latency_ms / 100000.0))  # tiny, scaled
        text = self._answer(prompt, model)
        return Completion(
            text=text,
            input_tokens=estimate_tokens(prompt),
            output_tokens=min(max_tokens, estimate_tokens(text)),
            confidence=self._confidence(prompt, model),
            model=model.name,
        )

    def stream(self, prompt, *, model, max_tokens, temperature) -> Iterator[str]:
        for tok in self._answer(prompt, model).split(" "):
            yield tok + " "


class OpenAICompatBackend(InferenceBackend):
    """Calls a real /chat/completions endpoint via httpx (lazy import)."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate(self, prompt, *, model, max_tokens, temperature) -> Completion:
        import httpx  # imported lazily so core stays dependency-free
        payload = {
            "model": model.name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{self.base_url}/chat/completions",
                            json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data.get("usage", {})
        conf = self._confidence_from_logprobs(choice)
        return Completion(
            text=text,
            input_tokens=usage.get("prompt_tokens", estimate_tokens(prompt)),
            output_tokens=usage.get("completion_tokens", estimate_tokens(text)),
            confidence=conf,
            model=model.name,
        )

    @staticmethod
    def _confidence_from_logprobs(choice: dict) -> float:
        import math
        lp = (choice.get("logprobs") or {}).get("content") or []
        if not lp:
            return 0.7
        avg = sum(t.get("logprob", -1.0) for t in lp) / len(lp)
        return max(0.0, min(1.0, math.exp(avg)))


def build_backend(settings) -> InferenceBackend:
    if settings.use_real_backend():
        return OpenAICompatBackend(settings.openai_base_url, settings.openai_api_key)
    return MockBackend()
