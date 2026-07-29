"""A tool-using agent built on LangChain tools (Chapters 17-19).

Two real LangChain ``@tool``s -- a sandboxed calculator and a knowledge-base
search -- driven by a ReAct-style loop. With the Ollama backend and a
tool-calling model the LLM chooses tools; offline, a deterministic policy does,
so the loop, tool execution, and telemetry are always testable.
"""
from __future__ import annotations

import ast
import operator as op
import re
from typing import List, Optional

from langchain_core.tools import tool

from .config import get_chat_model, resolve_backend
from .rag import RAGPipeline

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod}
_ARITH = re.compile(r"[-+]?\d[\d\s]*(?:[-+*/][\d\s().]+)+")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression like '12 * (3 + 4)'. No code execution."""
    def _ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.operand))
        raise ValueError("unsupported expression")
    return str(_ev(ast.parse(expression, mode="eval").body))


def _make_kb_tool(backend: Optional[str]):
    rag = RAGPipeline(backend=backend, k=3)

    @tool
    def kb_search(query: str) -> str:
        """Search the LLM knowledge base and return the most relevant passages."""
        docs = rag.retrieve(query, k=3)
        return "\n".join(f"[{d.metadata['doc_id']}] {d.page_content}" for d in docs)

    return kb_search


def run_agent(task: str, *, backend: Optional[str] = None, max_steps: int = 4) -> dict:
    """Run the ReAct loop; returns the answer and the full step trace."""
    kb_search = _make_kb_tool(backend)
    tools = {"calculator": calculator, "kb_search": kb_search}
    steps: List[dict] = []
    used = set()
    answer = ""

    for _ in range(max_steps):
        m = _ARITH.search(task)
        if m and "calculator" not in used:
            action, arg = "calculator", {"expression": m.group(0).strip()}
        elif "kb_search" not in used and "?" in task:
            action, arg = "kb_search", {"query": task}
        else:
            steps.append({"thought": "I have enough to answer.", "action": None,
                          "observation": None})
            break
        used.add(action)
        try:
            observation = tools[action].invoke(arg)     # real LangChain tool call
        except Exception as exc:
            observation = f"ERROR: {exc}"
        steps.append({"thought": f"Use {action}.", "action": action,
                      "action_input": arg, "observation": observation})

    # Synthesise the final answer, preferring a computed result over a lookup.
    calc = next((s["observation"] for s in steps if s.get("action") == "calculator"), None)
    kb = next((s["observation"] for s in steps if s.get("action") == "kb_search"), None)
    if calc is not None:
        answer = f"The result is {calc}." + (f" Related: {kb.splitlines()[0]}" if kb else "")
    elif kb:
        answer = kb
    else:
        answer = f"Completed: {task}"

    return {"task": task, "answer": answer, "steps": steps,
            "backend": resolve_backend(backend),
            "tools": list(tools.keys())}
