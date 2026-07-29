"""Request/response/telemetry contracts.

Plain dataclasses keep the core dependency-free. ``app.py`` adapts these to
Pydantic models at the HTTP boundary when FastAPI is installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ChatRequest:
    query: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    use_rag: bool = True
    stream: bool = False
    max_tokens: int = 256
    temperature: float = 0.2
    # Force a specific model (bypasses the router) -- handy for evals/A-B.
    force_model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Telemetry:
    """Per-request observability record (Chapter 2 / Part VIII)."""
    model: str = ""
    route_reason: str = ""
    cache: str = "miss"               # "exact" | "semantic" | "miss"
    retrieved_docs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    escalated: bool = False
    blocked: bool = False
    block_reason: str = ""
    confidence: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatResponse:
    answer: str
    citations: List[str] = field(default_factory=list)
    telemetry: Telemetry = field(default_factory=Telemetry)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "telemetry": self.telemetry.as_dict(),
        }
