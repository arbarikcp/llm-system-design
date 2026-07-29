"""Safety and guardrails subsystem (Part VI: Chapters 21-23).

Three layers:
  - InputSafetyPipeline  (Ch.21): screen the prompt and any retrieved/tool
    content BEFORE it reaches the model -- token budget, direct and indirect
    prompt injection, jailbreaks, toxicity, topic restriction, PII redaction.
  - OutputSafetyPipeline (Ch.22): screen the model's response -- toxicity, PII
    leakage, grounding/factuality, refusal handling, disclaimer/citation rules.
  - SafetyMonitor + RedTeamSuite (Ch.23): track refusal/jailbreak/toxicity
    metrics, detect anomalies, and run adversarial regression tests.

Deterministic and dependency-free so it runs as a CI gate. The pattern matching
here stands in for the trained classifiers (LlamaGuard, Perspective API,
fine-tuned moderation models) a production system would use.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .guardrails import scrub_pii
from .estimation import estimate_tokens
from .embeddings import tokenize
from .rag_eval import faithfulness


# --- shared types --------------------------------------------------------
@dataclass
class Verdict:
    allowed: bool
    action: str                       # "allow" | "block" | "redact" | "refuse"
    reason: str = ""
    score: float = 0.0
    categories: Tuple[str, ...] = ()
    text: str = ""                    # possibly modified (redacted/refusal) text


_INJECTION = [
    re.compile(r"ignore (all |previous |prior )+instructions", re.I),
    re.compile(r"disregard (the )?(system|above|previous)", re.I),
    re.compile(r"reveal (your )?(system )?(prompt|instructions)", re.I),
    re.compile(r"print (your )?(system )?prompt", re.I),
]
_JAILBREAK = [
    re.compile(r"\b(DAN|do anything now)\b", re.I),
    re.compile(r"developer mode", re.I),
    re.compile(r"pretend (you|to be) .*(no|without) (rules|restrictions|guidelines)", re.I),
    re.compile(r"you have no (restrictions|filters|guidelines)", re.I),
    re.compile(r"roleplay as .* (unfiltered|uncensored)", re.I),
]
_TOXIC_LEXICON = {
    "kill", "bomb", "attack", "hate", "destroy", "weapon", "exploit", "malware",
}


def toxicity_score(text: str) -> float:
    """Lexical toxicity proxy in [0,1] (a trained classifier in production)."""
    toks = tokenize(text)
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in _TOXIC_LEXICON)
    return min(1.0, hits / max(5, len(toks)) * 5)


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION)


def detect_jailbreak(text: str) -> bool:
    return any(p.search(text) for p in _JAILBREAK)


# --- input safety (Ch.21) -----------------------------------------------
class InputSafetyPipeline:
    def __init__(self, *, max_tokens: int = 6000, toxicity_threshold: float = 0.5,
                 blocked_topics: Sequence[str] = ()) -> None:
        self.max_tokens = max_tokens
        self.toxicity_threshold = toxicity_threshold
        self.blocked_topics = {t.lower() for t in blocked_topics}

    def check(self, text: str, *, retrieved_context: Optional[str] = None) -> Verdict:
        # 1) token budget / malformed
        if not text or not text.strip():
            return Verdict(False, "block", "empty_input")
        if estimate_tokens(text) > self.max_tokens:
            return Verdict(False, "block", "input_too_long", categories=("validation",))
        # 2) direct prompt injection
        if detect_injection(text):
            return Verdict(False, "block", "prompt_injection", categories=("injection",))
        # 3) jailbreak attempt
        if detect_jailbreak(text):
            return Verdict(False, "block", "jailbreak", categories=("jailbreak",))
        # 4) toxicity / content moderation
        tox = toxicity_score(text)
        if tox >= self.toxicity_threshold:
            return Verdict(False, "block", "toxic_input", score=tox, categories=("toxicity",))
        # 5) topic restriction
        toks = set(tokenize(text))
        hit = self.blocked_topics & toks
        if hit:
            return Verdict(False, "block", f"restricted_topic:{sorted(hit)[0]}",
                           categories=("topic",))
        # 6) indirect injection via retrieved content / tool results
        if retrieved_context and detect_injection(retrieved_context):
            return Verdict(False, "block", "indirect_injection",
                           categories=("injection", "indirect"))
        # 7) PII redaction (allow, but scrub)
        cleaned, n = scrub_pii(text)
        if n:
            return Verdict(True, "redact", "pii_redacted", score=float(n),
                           categories=("pii",), text=cleaned)
        return Verdict(True, "allow", text=text)


# --- output safety (Ch.22) ----------------------------------------------
SAFE_REFUSAL = ("I can't help with that. If you believe this is an error, please "
                "rephrase or contact support.")
UNGROUNDED_REFUSAL = ("I don't have enough grounded information to answer that "
                      "reliably.")


class OutputSafetyPipeline:
    def __init__(self, *, toxicity_threshold: float = 0.5,
                 require_citations: bool = False,
                 disclaimer: Optional[str] = None) -> None:
        self.toxicity_threshold = toxicity_threshold
        self.require_citations = require_citations
        self.disclaimer = disclaimer

    def check(self, answer: str, *, context: str = "",
              require_grounding: bool = False) -> Verdict:
        # 1) toxicity / harmfulness
        tox = toxicity_score(answer)
        if tox >= self.toxicity_threshold:
            return Verdict(False, "block", "unsafe_output", score=tox,
                           categories=("toxicity",), text=SAFE_REFUSAL)
        # 2) PII leakage in the output -> redact
        cleaned, n = scrub_pii(answer)
        categories: Tuple[str, ...] = ("pii",) if n else ()
        # 3) factuality / grounding against retrieved context
        if require_grounding:
            f = faithfulness(cleaned, context)
            if f < 0.5:
                return Verdict(True, "refuse", "ungrounded", score=round(f, 3),
                               categories=categories + ("grounding",),
                               text=UNGROUNDED_REFUSAL)
        # 4) citation enforcement
        if self.require_citations and "[" not in cleaned:
            return Verdict(True, "refuse", "missing_citation",
                           categories=categories + ("citation",),
                           text="I can only answer with cited sources; none were found.")
        # 5) disclaimer / caveat injection
        if self.disclaimer:
            cleaned = f"{cleaned}\n\n{self.disclaimer}"
        action = "redact" if n else "allow"
        return Verdict(True, action, "pii_redacted" if n else "", score=float(n),
                       categories=categories, text=cleaned)


# --- monitoring + incident response (Ch.23) ------------------------------
@dataclass
class SafetyEvent:
    stage: str                        # "input" | "output"
    action: str                       # allow|block|redact|refuse
    category: str = ""
    toxicity: float = 0.0
    jailbreak_attempt: bool = False
    jailbreak_blocked: bool = False


@dataclass
class SafetyMonitor:
    events: List[SafetyEvent] = field(default_factory=list)

    def record(self, event: SafetyEvent) -> None:
        self.events.append(event)

    def block_rate(self) -> float:
        if not self.events:
            return 0.0
        blocked = sum(1 for e in self.events if e.action in ("block", "refuse"))
        return blocked / len(self.events)

    def refusal_rate_by_category(self) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for e in self.events:
            if e.action in ("block", "refuse") and e.category:
                counts[e.category] = counts.get(e.category, 0) + 1
        total = len(self.events) or 1
        return {k: round(v / total, 3) for k, v in counts.items()}

    def jailbreak_success_rate(self) -> float:
        attempts = [e for e in self.events if e.jailbreak_attempt]
        if not attempts:
            return 0.0
        succeeded = sum(1 for e in attempts if not e.jailbreak_blocked)
        return succeeded / len(attempts)

    def toxicity_trend(self, window: int = 50) -> float:
        recent = [e.toxicity for e in self.events[-window:]]
        return round(statistics.mean(recent), 4) if recent else 0.0

    def detect_anomaly(self, *, baseline_window: int = 100,
                       recent_window: int = 20, z: float = 3.0) -> bool:
        """Flag if the recent block rate deviates sharply from the baseline."""
        if len(self.events) < baseline_window + recent_window:
            return False
        def rate(evs):
            b = sum(1 for e in evs if e.action in ("block", "refuse"))
            return b / len(evs) if evs else 0.0
        base = self.events[-(baseline_window + recent_window):-recent_window]
        recent = self.events[-recent_window:]
        flags = [1 if e.action in ("block", "refuse") else 0 for e in base]
        mu = statistics.mean(flags) if flags else 0.0
        sigma = statistics.pstdev(flags) if len(flags) > 1 else 0.0
        if sigma == 0:
            return rate(recent) > mu + 0.2
        return abs(rate(recent) - mu) > z * sigma


# --- red-teaming (Ch.23) -------------------------------------------------
@dataclass
class RedTeamCase:
    prompt: str
    should_block: bool
    category: str = ""


@dataclass
class RedTeamReport:
    total: int
    attacks: int
    caught: int
    benign: int
    false_positives: int
    rows: List[dict] = field(default_factory=list)

    @property
    def catch_rate(self) -> float:
        return self.caught / self.attacks if self.attacks else 1.0

    @property
    def false_positive_rate(self) -> float:
        return self.false_positives / self.benign if self.benign else 0.0


class RedTeamSuite:
    """Run adversarial + benign cases through the input pipeline and score it."""

    def __init__(self, pipeline: InputSafetyPipeline) -> None:
        self.pipeline = pipeline

    def run(self, cases: List[RedTeamCase]) -> RedTeamReport:
        attacks = sum(1 for c in cases if c.should_block)
        benign = len(cases) - attacks
        caught = fp = 0
        rows = []
        for c in cases:
            v = self.pipeline.check(c.prompt)
            blocked = v.action in ("block", "refuse")
            if c.should_block and blocked:
                caught += 1
            if (not c.should_block) and blocked:
                fp += 1
            rows.append({"prompt": c.prompt[:48], "category": c.category,
                         "should_block": c.should_block, "blocked": blocked,
                         "reason": v.reason})
        return RedTeamReport(len(cases), attacks, caught, benign, fp, rows)
