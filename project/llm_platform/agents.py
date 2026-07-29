"""Single-agent architecture: the ReAct loop (Chapter 17).

A ReAct agent interleaves Thought -> Action -> Observation until it can answer.
The "brain" is a pluggable ``policy``; in production it is the LLM emitting tool
calls. We ship a deterministic ``RuleBasedPolicy`` so the loop, tool execution,
error recovery, and termination are all runnable and testable offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .tools import ToolRegistry, ToolExecutor

_ARITH = re.compile(r"[-+]?\d[\d\s]*(?:[-+*/][\d\s().]+)+")


@dataclass
class Step:
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


@dataclass
class AgentResult:
    answer: str
    steps: List[Step]
    success: bool


class Policy:
    """Decides the next step given the task and the scratchpad of past steps."""
    def decide(self, task: str, scratch: List[Step]) -> Dict[str, Any]:
        raise NotImplementedError


class RuleBasedPolicy(Policy):
    """A deterministic stand-in for an LLM tool-calling policy.

    Real policies prompt the model with the tool specs and parse its emitted
    function call. The decision *shape* here is identical, so a real
    ``LLMPolicy`` is a drop-in replacement.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def decide(self, task: str, scratch: List[Step]) -> Dict[str, Any]:
        used = {s.action for s in scratch}
        m = _ARITH.search(task)
        if m and self.registry.get("calculator") and "calculator" not in used:
            return {"type": "action", "thought": "This needs arithmetic; use the calculator.",
                    "action": "calculator", "action_input": {"expression": m.group(0).strip()}}
        if self.registry.get("kb_search") and "kb_search" not in used and "?" in task:
            return {"type": "action", "thought": "Look up supporting facts.",
                    "action": "kb_search", "action_input": {"query": task}}
        # Synthesize a final answer from observations.
        obs = [s.observation for s in scratch if s.observation]
        answer = obs[-1] if obs else "I could not find enough information."
        return {"type": "final", "thought": "I have enough to answer.", "answer": str(answer)}


class ReActAgent:
    def __init__(self, registry: ToolRegistry, *, policy: Optional[Policy] = None,
                 executor: Optional[ToolExecutor] = None, max_steps: int = 6) -> None:
        self.registry = registry
        self.executor = executor or ToolExecutor(registry)
        self.policy = policy or RuleBasedPolicy(registry)
        self.max_steps = max_steps

    def run(self, task: str) -> AgentResult:
        steps: List[Step] = []
        for _ in range(self.max_steps):
            decision = self.policy.decide(task, steps)
            if decision["type"] == "final":
                steps.append(Step(decision["thought"]))
                return AgentResult(decision["answer"], steps, True)

            result = self.executor.execute(decision["action"], decision["action_input"])
            # Graceful error recovery: feed the error back as an observation so
            # the policy can choose a different action next turn.
            observation = str(result.output) if result.ok else f"ERROR: {result.error}"
            steps.append(Step(decision["thought"], decision["action"],
                              decision["action_input"], observation))
        return AgentResult("Stopped: reached max steps without finishing.", steps, False)
