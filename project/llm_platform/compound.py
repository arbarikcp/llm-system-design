"""Compound AI systems: a DAG flow engine for multi-step LLM pipelines
(Chapter 46).

A flow is a directed acyclic graph of nodes (LLM calls, tools, routers,
verifiers). The engine resolves dependencies, runs nodes in topological order,
retries failures, and supports conditional skipping -- the runtime under
"flow engineering" and frameworks like DSPy/LangGraph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Node:
    name: str
    fn: Callable[[Dict[str, Any]], Any]   # receives the context, returns its output
    deps: List[str] = field(default_factory=list)
    retries: int = 0
    # Optional predicate over the context; if it returns False the node is skipped.
    when: Optional[Callable[[Dict[str, Any]], bool]] = None


class FlowGraph:
    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}

    def add(self, node: Node) -> "FlowGraph":
        self._nodes[node.name] = node
        return self

    def _topo_order(self) -> List[str]:
        order: List[str] = []
        temp, perm = set(), set()

        def visit(n: str):
            if n in perm:
                return
            if n in temp:
                raise ValueError(f"cycle detected at '{n}'")
            temp.add(n)
            for d in self._nodes[n].deps:
                if d not in self._nodes:
                    raise KeyError(f"node '{n}' depends on unknown '{d}'")
                visit(d)
            temp.discard(n); perm.add(n); order.append(n)

        for name in self._nodes:
            visit(name)
        return order

    def run(self, initial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx: Dict[str, Any] = dict(initial or {})
        ctx.setdefault("_skipped", [])
        ctx.setdefault("_errors", {})
        for name in self._topo_order():
            node = self._nodes[name]
            if node.when is not None and not node.when(ctx):
                ctx["_skipped"].append(name)        # conditional branch skip
                continue
            last_error = None
            for attempt in range(node.retries + 1):
                try:
                    ctx[name] = node.fn(ctx)        # store output under the node name
                    break
                except Exception as exc:            # retry at the node level
                    last_error = f"{type(exc).__name__}: {exc}"
            else:
                ctx["_errors"][name] = last_error
        return ctx
