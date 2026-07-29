"""Agent memory system (Chapter 20).

Three memory types with explicit read/write/update/delete paths:
  - ShortTermMemory: the conversation, with a sliding window + summarization.
  - LongTermMemory: a vector store recalled by semantic similarity, with
    recency/relevance scoring and decay-based forgetting.
  - WorkingMemory: a scratchpad for plan state and intermediate variables.
``AgentMemory`` composes them into one interface.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .embeddings import HashingEmbedder, cosine_similarity


class ShortTermMemory:
    """Conversation memory: keep a recent window, summarize what falls off."""

    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max_turns
        self.turns: List[Tuple[str, str]] = []
        self.summary: str = ""

    def add(self, role: str, text: str) -> None:
        self.turns.append((role, text))
        if len(self.turns) > self.max_turns:
            overflow = self.turns[:-self.max_turns]
            self.turns = self.turns[-self.max_turns:]
            digest = " ".join(t for _, t in overflow)
            self.summary = (self.summary + " " + digest).strip()[:600]

    def context(self) -> str:
        head = f"[summary] {self.summary}\n" if self.summary else ""
        return head + "\n".join(f"{r}: {t}" for r, t in self.turns)


@dataclass
class MemoryItem:
    text: str
    embedding: List[float]
    created_at: float
    last_used: float
    uses: int = 0
    tags: Tuple[str, ...] = ()


class LongTermMemory:
    """Vector-store memory with recency-weighted recall and decay forgetting."""

    def __init__(self, embedder: Optional[HashingEmbedder] = None,
                 half_life_s: float = 7 * 24 * 3600) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.half_life_s = half_life_s
        self._items: List[MemoryItem] = []

    def write(self, text: str, *, tags: Tuple[str, ...] = (),
              now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        self._items.append(MemoryItem(text, self.embedder.embed(text), now, now, 0, tags))

    def _recency(self, item: MemoryItem, now: float) -> float:
        age = max(0.0, now - item.last_used)
        return 0.5 ** (age / self.half_life_s)        # exponential decay

    def recall(self, query: str, k: int = 3, *, alpha: float = 0.7,
               now: Optional[float] = None) -> List[str]:
        """Rank by alpha*semantic + (1-alpha)*recency, then mark items used."""
        now = now if now is not None else time.time()
        q = self.embedder.embed(query)
        scored = []
        for it in self._items:
            sem = cosine_similarity(q, it.embedding)
            score = alpha * sem + (1 - alpha) * self._recency(it, now)
            scored.append((score, it))
        scored.sort(key=lambda s: s[0], reverse=True)
        top = [it for _, it in scored[:k]]
        for it in top:
            it.uses += 1
            it.last_used = now
        return [it.text for it in top]

    def update(self, old_text: str, new_text: str) -> bool:
        for it in self._items:
            if it.text == old_text:
                it.text = new_text
                it.embedding = self.embedder.embed(new_text)
                return True
        return False

    def forget(self, *, max_age_s: float, now: Optional[float] = None) -> int:
        """Delete stale, unused memories (the forget/delete path)."""
        now = now if now is not None else time.time()
        before = len(self._items)
        self._items = [it for it in self._items
                       if (now - it.last_used) < max_age_s or it.uses > 0]
        return before - len(self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class WorkingMemory:
    """Scratchpad: plan, progress, and intermediate variables."""
    scratch: List[str] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    plan: List[str] = field(default_factory=list)
    done: set = field(default_factory=set)

    def note(self, text: str) -> None:
        self.scratch.append(text)

    def set_plan(self, steps: List[str]) -> None:
        self.plan = list(steps)

    def complete(self, step: str) -> None:
        self.done.add(step)

    def progress(self) -> str:
        return f"{len(self.done)}/{len(self.plan)} steps complete"


class AgentMemory:
    """Unified read/write interface over the three memory types."""

    def __init__(self, embedder: Optional[HashingEmbedder] = None) -> None:
        self.short = ShortTermMemory()
        self.long = LongTermMemory(embedder)
        self.working = WorkingMemory()

    def observe(self, role: str, text: str) -> None:
        self.short.add(role, text)

    def remember(self, fact: str, **kw) -> None:
        self.long.write(fact, **kw)

    def recall(self, query: str, k: int = 3) -> List[str]:
        return self.long.recall(query, k=k)

    def assemble_context(self, query: str, k: int = 3) -> str:
        recalled = self.recall(query, k=k)
        mem = "\n".join(f"- {m}" for m in recalled)
        return f"{self.short.context()}\n\n[recalled memories]\n{mem}".strip()
