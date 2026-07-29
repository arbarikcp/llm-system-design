"""Multi-agent orchestration: supervisor/worker + blackboard (Chapter 18).

A supervisor decomposes a task into subtasks and routes each to the
best-matching specialized worker. Workers share state through a blackboard,
which is the cleanest way to propagate context across agent boundaries without
ad-hoc message plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from .embeddings import tokenize


@dataclass
class Blackboard:
    """Shared workspace agents read from and write to."""
    entries: Dict[str, str] = field(default_factory=dict)
    log: List[Tuple[str, str]] = field(default_factory=list)

    def write(self, key: str, value: str, *, by: str = "system") -> None:
        self.entries[key] = value
        self.log.append((by, key))

    def read(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.entries.get(key, default)


@dataclass
class Worker:
    name: str
    role: str
    skills: Set[str]                          # keywords this worker handles
    handler: Callable[[str, Blackboard], str]

    def match(self, subtask: str) -> int:
        return len(self.skills & set(tokenize(subtask)))


class Supervisor:
    """Routes subtasks to specialized workers and aggregates their results."""

    def __init__(self, workers: List[Worker]) -> None:
        if not workers:
            raise ValueError("need at least one worker")
        self.workers = workers

    def decompose(self, task: str) -> List[str]:
        """Split a compound task into subtasks (LLM does this in production)."""
        parts = [p.strip() for p in task.replace(";", " then ").split(" then ")]
        return [p for p in parts if p]

    def route(self, subtask: str) -> Worker:
        return max(self.workers, key=lambda w: (w.match(subtask), -len(w.skills)))

    def run(self, task: str) -> Tuple[List[Tuple[str, str]], Blackboard]:
        bb = Blackboard()
        bb.write("task", task)
        results: List[Tuple[str, str]] = []
        for subtask in self.decompose(task):
            worker = self.route(subtask)
            output = worker.handler(subtask, bb)
            bb.write(f"result:{worker.name}:{subtask[:24]}", output, by=worker.name)
            results.append((worker.name, output))
        return results, bb


# --- Convenience workers for the demo ------------------------------------
def make_default_team() -> List[Worker]:
    def researcher(subtask, bb):
        return f"researched: key facts about '{subtask}'"

    def coder(subtask, bb):
        facts = [v for k, v in bb.entries.items() if k.startswith("result:")]
        return f"wrote code for '{subtask}' using {len(facts)} prior result(s)"

    def reviewer(subtask, bb):
        return f"reviewed '{subtask}': looks correct"

    return [
        Worker("researcher", "research", {"research", "find", "facts", "look", "investigate"}, researcher),
        Worker("coder", "engineering", {"code", "implement", "write", "build", "function"}, coder),
        Worker("reviewer", "qa", {"review", "check", "verify", "test", "critique"}, reviewer),
    ]
