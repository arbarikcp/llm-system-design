"""Tool / function-calling infrastructure (Chapter 19).

A tool registry with JSON-schema-style definitions, an executor that *validates
before executing*, retries transient failures with exponential backoff, caches
deterministic tools, and returns structured errors the agent can read. This is
the layer that turns "the model emitted a function call" into a safe, observable
side effect.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_JSON_TYPES = {
    "string": str, "number": (int, float), "integer": int,
    "boolean": bool, "object": dict, "array": list,
}


def _type_ok(value: Any, json_type: str) -> bool:
    py = _JSON_TYPES.get(json_type)
    if py is None:
        return True
    if json_type == "integer" and isinstance(value, bool):
        return False           # bool is a subclass of int; reject for integer
    return isinstance(value, py)


@dataclass
class Tool:
    """A callable the agent can invoke, with a self-describing schema."""
    name: str
    description: str
    parameters: Dict[str, dict]          # {param: {"type", "required", "description"}}
    func: Callable[..., Any]
    version: str = "1.0"
    deterministic: bool = False          # cacheable if True
    timeout_s: float = 10.0

    def validate(self, args: Dict[str, Any]) -> List[str]:
        """Return a list of validation errors (empty == valid)."""
        errors: List[str] = []
        for pname, spec in self.parameters.items():
            if spec.get("required") and pname not in args:
                errors.append(f"missing required parameter '{pname}'")
        for pname, value in args.items():
            spec = self.parameters.get(pname)
            if spec is None:
                errors.append(f"unknown parameter '{pname}'")
            elif "type" in spec and not _type_ok(value, spec["type"]):
                errors.append(f"parameter '{pname}' must be {spec['type']}")
        return errors

    def spec(self) -> dict:
        """The definition handed to the LLM for tool selection."""
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters, "version": self.version}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list(self) -> List[Tool]:
        return list(self._tools.values())

    def specs(self) -> List[dict]:
        return [t.spec() for t in self._tools.values()]


@dataclass
class ToolResult:
    ok: bool
    output: Any = None
    error: str = ""
    cached: bool = False
    attempts: int = 1


class ToolExecutor:
    """Validates, executes, retries, and caches tool calls."""

    def __init__(self, registry: ToolRegistry, *, max_retries: int = 2,
                 base_backoff: float = 0.0) -> None:
        self.registry = registry
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self._cache: Dict[str, Any] = {}

    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        tool = self.registry.get(name)
        if tool is None:
            return ToolResult(False, error=f"unknown tool '{name}'")

        # Validate BEFORE executing -- never run a malformed call.
        errors = tool.validate(args)
        if errors:
            return ToolResult(False, error="; ".join(errors))

        cache_key = None
        if tool.deterministic:
            cache_key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            if cache_key in self._cache:
                return ToolResult(True, self._cache[cache_key], cached=True)

        last_error = ""
        for attempt in range(1, self.max_retries + 2):
            try:
                output = tool.func(**args)
                if cache_key is not None:
                    self._cache[cache_key] = output
                return ToolResult(True, output, attempts=attempt)
            except Exception as exc:               # noqa: BLE001 - report to the agent
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt <= self.max_retries:
                    time.sleep(self.base_backoff * (2 ** (attempt - 1)))  # backoff
                    continue
                return ToolResult(False, error=last_error, attempts=attempt)
        return ToolResult(False, error=last_error)


# --- A couple of real tools used by the agent demo -----------------------
import ast
import operator as _op

_OPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
    ast.Div: _op.truediv, ast.Pow: _op.pow, ast.USub: _op.neg, ast.Mod: _op.mod,
}


def safe_calculator(expression: str) -> float:
    """Evaluate an arithmetic expression without eval() (no code execution)."""
    def _ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.operand))
        raise ValueError("unsupported expression")
    return _ev(ast.parse(expression, mode="eval").body)


def make_calculator_tool() -> Tool:
    return Tool(
        name="calculator",
        description="Evaluate an arithmetic expression, e.g. '2 * (3 + 4)'.",
        parameters={"expression": {"type": "string", "required": True,
                                   "description": "Arithmetic expression"}},
        func=safe_calculator, deterministic=True,
    )
