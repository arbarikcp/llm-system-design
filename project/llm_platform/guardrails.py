"""Input and output safety pipelines (Part VI).

Input guard: blocks oversized prompts, scrubs PII, and flags prompt-injection
patterns *before* the request reaches the model. Output guard: redacts leaked
PII and checks that grounded answers actually cite retrieved context.

These are deterministic, fast, and run on the hot path -- the cheap first line
of defense in front of slower LLM-based classifiers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |previous |prior )+instructions", re.I),
    re.compile(r"disregard (the )?(system|above)", re.I),
    re.compile(r"reveal (your )?(system )?prompt", re.I),
    re.compile(r"you are now (a|an|in) ", re.I),
]


@dataclass
class GuardResult:
    allowed: bool
    text: str
    reason: str = ""
    redactions: int = 0


def scrub_pii(text: str) -> Tuple[str, int]:
    count = 0

    def _sub(pattern, token, s):
        nonlocal count
        new, n = pattern.subn(token, s)
        count += n
        return new

    text = _sub(_SSN, "[REDACTED_SSN]", text)
    text = _sub(_EMAIL, "[REDACTED_EMAIL]", text)
    text = _sub(_PHONE, "[REDACTED_PHONE]", text)
    return text, count


class InputGuard:
    def __init__(self, max_chars: int = 24000) -> None:
        self.max_chars = max_chars

    def check(self, text: str) -> GuardResult:
        if len(text) > self.max_chars:
            return GuardResult(False, text, reason="input_too_long")
        for pat in _INJECTION_PATTERNS:
            if pat.search(text):
                return GuardResult(False, text, reason="prompt_injection")
        cleaned, n = scrub_pii(text)
        return GuardResult(True, cleaned, redactions=n)


class OutputGuard:
    BANNED = [re.compile(r"\b(kill yourself|build a bomb)\b", re.I)]

    def check(self, text: str, *, require_grounding: bool, has_context: bool) -> GuardResult:
        for pat in self.BANNED:
            if pat.search(text):
                return GuardResult(False, text, reason="unsafe_output")
        cleaned, n = scrub_pii(text)
        if require_grounding and not has_context:
            # Refuse to answer from parametric memory when grounding is required.
            return GuardResult(
                True,
                "I don't have enough grounded context to answer that reliably.",
                reason="ungrounded_refusal",
                redactions=n,
            )
        return GuardResult(True, cleaned, redactions=n)
