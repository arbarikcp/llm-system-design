"""Input and output guardrails (Chapters 21-23).

Fast, deterministic checks that run on the hot path: input screening for prompt
injection, jailbreaks, and PII; output screening for toxicity, PII leakage, and
grounding. A red-team runner scores catch-rate over the seed attack set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .seed import SEED_REDTEAM

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_INJECTION = [
    re.compile(r"ignore (all |previous |prior )+instructions", re.I),
    re.compile(r"disregard (the )?(system|above|previous)", re.I),
    re.compile(r"reveal (your )?(system )?(prompt|instructions)", re.I),
]
_JAILBREAK = [
    re.compile(r"\b(DAN|do anything now)\b", re.I),
    re.compile(r"developer mode", re.I),
    re.compile(r"pretend (you|to be).*(no|without) (rules|restrictions|guidelines)", re.I),
]
_TOXIC = {"kill", "bomb", "attack", "hate", "destroy", "weapon", "exploit", "malware"}


@dataclass
class Verdict:
    allowed: bool
    action: str            # allow | block | redact | refuse
    reason: str = ""
    text: str = ""
    categories: Tuple[str, ...] = ()


def scrub_pii(text: str) -> Tuple[str, int]:
    n = 0
    for pat, tok in ((_SSN, "[SSN]"), (_EMAIL, "[EMAIL]"), (_PHONE, "[PHONE]")):
        text, c = pat.subn(tok, text)
        n += c
    return text, n


def toxicity_score(text: str) -> float:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    if not toks:
        return 0.0
    return min(1.0, sum(1 for t in toks if t in _TOXIC) / max(5, len(toks)) * 5)


def check_input(text: str, *, max_chars: int = 8000) -> Verdict:
    if not text.strip():
        return Verdict(False, "block", "empty_input")
    if len(text) > max_chars:
        return Verdict(False, "block", "input_too_long")
    if any(p.search(text) for p in _INJECTION):
        return Verdict(False, "block", "prompt_injection", categories=("injection",))
    if any(p.search(text) for p in _JAILBREAK):
        return Verdict(False, "block", "jailbreak", categories=("jailbreak",))
    if toxicity_score(text) >= 0.5:
        return Verdict(False, "block", "toxic_input", categories=("toxicity",))
    cleaned, n = scrub_pii(text)
    return Verdict(True, "redact" if n else "allow",
                   "pii_redacted" if n else "", text=cleaned,
                   categories=("pii",) if n else ())


def check_output(answer: str, *, context: str = "", require_grounding: bool = False) -> Verdict:
    if toxicity_score(answer) >= 0.5:
        return Verdict(False, "block", "unsafe_output",
                       text="I can't help with that.", categories=("toxicity",))
    cleaned, n = scrub_pii(answer)
    if require_grounding and context:
        words = [w for w in re.findall(r"[a-z0-9]+", cleaned.lower()) if len(w) > 3]
        ctx = set(re.findall(r"[a-z0-9]+", context.lower()))
        supported = sum(1 for w in words if w in ctx) / len(words) if words else 1.0
        if supported < 0.4:
            return Verdict(True, "refuse", "ungrounded",
                           text="I don't have enough grounded context to answer.",
                           categories=("grounding",))
    return Verdict(True, "redact" if n else "allow",
                   "pii_redacted" if n else "", text=cleaned,
                   categories=("pii",) if n else ())


def run_redteam(cases: Optional[List[dict]] = None) -> dict:
    cases = cases or SEED_REDTEAM
    attacks = [c for c in cases if c["should_block"]]
    rows, caught, fp = [], 0, 0
    for c in cases:
        v = check_input(c["prompt"])
        blocked = v.action in ("block", "refuse")
        if c["should_block"] and blocked:
            caught += 1
        if (not c["should_block"]) and blocked:
            fp += 1
        rows.append({"prompt": c["prompt"][:50], "category": c["category"],
                     "should_block": c["should_block"], "blocked": blocked,
                     "reason": v.reason})
    return {"catch_rate": round(caught / len(attacks), 3) if attacks else 1.0,
            "false_positives": fp, "rows": rows}
