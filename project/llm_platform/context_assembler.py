"""Prompt templating and budget-aware context assembly (Chapter 24).

The context window is a scarce, shared resource. The assembler packs context
sources -- system prompt, user profile, conversation history, retrieved docs,
tool definitions -- into the token budget by priority, truncating or dropping the
least important when space runs out. Prompt templates are versioned so they can be
A/B tested like any other change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .estimation import estimate_tokens, CHARS_PER_TOKEN


# --- Prompt templates ----------------------------------------------------
class PromptTemplate:
    """A versioned template with ``{name}`` placeholders.

    Rendering substitutes values literally (not via str.format), so a value that
    contains braces cannot break or inject into the template -- a small but real
    safety property when user content is interpolated.
    """
    _VAR = re.compile(r"\{(\w+)\}")

    def __init__(self, name: str, template: str, version: str = "1.0") -> None:
        self.name = name
        self.template = template
        self.version = version

    def variables(self) -> set:
        return set(self._VAR.findall(self.template))

    def render(self, **values) -> str:
        missing = self.variables() - set(values)
        if missing:
            raise KeyError(f"missing template variables: {sorted(missing)}")
        return self._VAR.sub(lambda m: str(values[m.group(1)]), self.template)


class PromptLibrary:
    """Registry of templates keyed by (name, version) with A/B selection."""

    def __init__(self) -> None:
        self._t: Dict[str, Dict[str, PromptTemplate]] = {}

    def register(self, template: PromptTemplate) -> None:
        self._t.setdefault(template.name, {})[template.version] = template

    def get(self, name: str, version: str = "1.0") -> PromptTemplate:
        return self._t[name][version]

    def choose(self, name: str, *, user_id: str, weights: Dict[str, float]) -> PromptTemplate:
        """Deterministically bucket a user into a template version for A/B tests."""
        versions = sorted(weights)
        total = sum(weights.values()) or 1.0
        h = (hash(f"{name}:{user_id}") % 10_000) / 10_000.0 * total
        cum = 0.0
        for v in versions:
            cum += weights[v]
            if h < cum:
                return self.get(name, v)
        return self.get(name, versions[-1])


# --- Budget-aware context assembly --------------------------------------
@dataclass
class ContextBlock:
    name: str
    content: str
    priority: int = 0          # higher = kept first among optional blocks
    required: bool = False     # always included (truncated if it must be)


@dataclass
class AssembledContext:
    text: str
    used_tokens: int
    included: List[str]
    dropped: List[str]
    truncated: List[str]


class ContextAssembler:
    def __init__(self, max_tokens: int = 4000) -> None:
        self.max_tokens = max_tokens

    def _truncate(self, content: str, budget_tokens: int) -> str:
        max_chars = max(0, int(budget_tokens * CHARS_PER_TOKEN))
        return content[:max_chars]

    def assemble(self, blocks: List[ContextBlock]) -> AssembledContext:
        required = [b for b in blocks if b.required]
        optional = sorted((b for b in blocks if not b.required),
                          key=lambda b: b.priority, reverse=True)

        used = 0
        parts: List[ContextBlock] = []
        included: List[str] = []
        dropped: List[str] = []
        truncated: List[str] = []

        # Required blocks first: include, truncating to remaining budget if needed.
        for b in required:
            t = estimate_tokens(b.content)
            remaining = self.max_tokens - used
            if t <= remaining:
                parts.append(b); used += t; included.append(b.name)
            elif remaining > 0:
                cut = ContextBlock(b.name, self._truncate(b.content, remaining), b.priority, True)
                parts.append(cut); used += estimate_tokens(cut.content)
                included.append(b.name); truncated.append(b.name)
            else:
                dropped.append(b.name)

        # Optional blocks by priority while budget remains.
        for b in optional:
            t = estimate_tokens(b.content)
            if used + t <= self.max_tokens:
                parts.append(b); used += t; included.append(b.name)
            else:
                dropped.append(b.name)

        text = "\n\n".join(f"### {b.name}\n{b.content}" for b in parts)
        return AssembledContext(text, used, included, dropped, truncated)
