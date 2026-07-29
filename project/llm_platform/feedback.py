"""Feedback collection and the data flywheel (Chapter 25).

Turns production signals -- explicit ratings, implicit behaviors, and comparative
preferences -- into curated training data, closing the collect -> curate ->
fine-tune -> deploy loop. Includes quality filtering and an active-learning
selector that spends scarce annotation budget on the most informative examples.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Implicit behaviors mapped to a polarity.
_IMPLICIT_POLARITY = {
    "copy": 1, "accept": 1, "thumbs_up": 1,
    "regenerate": -1, "edit": -1, "abandon": -1, "thumbs_down": -1,
}


@dataclass
class Feedback:
    request_id: str
    kind: str                      # "explicit" | "implicit" | "comparative"
    signal: str                    # e.g. "up"/"down"/"regenerate"/"edit"/"copy"
    query: str = ""
    response: str = ""
    chosen: str = ""               # comparative: preferred response
    rejected: str = ""             # comparative: dispreferred response
    user_id: str = "anon"
    ts: float = field(default_factory=time.time)

    def polarity(self) -> int:
        if self.kind == "explicit":
            return 1 if self.signal in ("up", "thumbs_up", "positive") else -1
        if self.kind == "implicit":
            return _IMPLICIT_POLARITY.get(self.signal, 0)
        return 0


@dataclass
class FeedbackStore:
    items: List[Feedback] = field(default_factory=list)

    def add(self, fb: Feedback) -> None:
        self.items.append(fb)

    def satisfaction_rate(self) -> float:
        rated = [f for f in self.items if f.polarity() != 0]
        if not rated:
            return 0.0
        return sum(1 for f in rated if f.polarity() > 0) / len(rated)

    def aggregate(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.items:
            counts[f.signal] = counts.get(f.signal, 0) + 1
        return counts

    # --- curation: production -> training data ---------------------------
    def to_sft_examples(self, *, min_len: int = 1) -> List[dict]:
        """Positive-signal (query, response) pairs become SFT examples."""
        out = []
        for f in self.items:
            if f.polarity() > 0 and f.query and len(f.response) >= min_len:
                out.append({"prompt": f.query, "completion": f.response})
        return out

    def to_preference_pairs(self) -> List[dict]:
        """Comparative feedback becomes (prompt, chosen, rejected) DPO pairs."""
        return [{"prompt": f.query, "chosen": f.chosen, "rejected": f.rejected}
                for f in self.items
                if f.kind == "comparative" and f.chosen and f.rejected]


def quality_filter(examples: List[dict], *, min_chars: int = 5,
                   max_chars: int = 8000) -> List[dict]:
    """Drop empty, too-short, too-long, or duplicate examples."""
    seen = set()
    clean = []
    for ex in examples:
        key = (ex.get("prompt", ""), ex.get("completion", ex.get("chosen", "")))
        body = key[1]
        if not body or len(body) < min_chars or len(body) > max_chars:
            continue
        if key in seen:
            continue
        seen.add(key)
        clean.append(ex)
    return clean


@dataclass
class Candidate:
    item_id: str
    text: str
    confidence: float              # model's confidence in its own output


class ActiveLearner:
    """Select the most informative items to annotate (uncertainty sampling)."""

    def select(self, candidates: List[Candidate], k: int = 10) -> List[Candidate]:
        # Lowest-confidence items are the most informative to label.
        return sorted(candidates, key=lambda c: c.confidence)[:k]
