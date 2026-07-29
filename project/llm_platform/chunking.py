"""Document chunking strategies (Chapter 15).

Chunking is the highest-leverage knob in a RAG system: too large and retrieval
returns irrelevant filler, too small and it fragments the answer. These are pure,
testable implementations of the strategies you actually choose between.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        src = self.metadata.get("source", "doc")
        return f"{src}#{self.index}"


def fixed_size_chunks(text: str, *, size: int = 500, overlap: int = 50,
                      source: str = "doc") -> List[Chunk]:
    """Fixed character windows with overlap.

    Overlap preserves context across boundaries so a fact split between two
    chunks is still retrievable. Simple and fast; ignores document structure.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    chunks, i, idx = [], 0, 0
    while i < len(text):
        piece = text[i:i + size].strip()
        if piece:
            chunks.append(Chunk(piece, idx, {"source": source, "strategy": "fixed"}))
            idx += 1
        i += step
    return chunks


def recursive_split(text: str, *, max_chars: int = 500,
                    separators: Optional[List[str]] = None,
                    source: str = "doc") -> List[Chunk]:
    """Recursive character splitting on a separator hierarchy.

    Tries to split on paragraph breaks first, then lines, then sentences, then
    words, so chunks fall on natural boundaries while respecting a size cap.
    This is the sensible default for most prose.
    """
    separators = separators or ["\n\n", "\n", ". ", " "]

    def _split(t: str, seps: List[str]) -> List[str]:
        if len(t) <= max_chars or not seps:
            return [t]
        sep, rest = seps[0], seps[1:]
        out, cur = [], ""
        for part in t.split(sep):
            candidate = f"{cur}{sep}{part}" if cur else part
            if len(candidate) <= max_chars:
                cur = candidate
            else:
                if cur:
                    out.append(cur)
                if len(part) > max_chars:
                    out.extend(_split(part, rest))
                    cur = ""
                else:
                    cur = part
        if cur:
            out.append(cur)
        return out

    pieces = _split(text, separators)
    return [Chunk(p.strip(), i, {"source": source, "strategy": "recursive"})
            for i, p in enumerate(pieces) if p.strip()]


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def structure_aware_chunks(markdown: str, *, source: str = "doc") -> List[Chunk]:
    """Split on Markdown headings, attaching the section title as metadata.

    Structure-aware chunking keeps a section's content together and records its
    heading, which both improves retrieval relevance and lets you cite the
    section a passage came from.
    """
    chunks: List[Chunk] = []
    heading, body, idx = "", [], 0

    def flush():
        nonlocal body, idx
        text = "\n".join(body).strip()
        if text:
            chunks.append(Chunk(text, idx, {"source": source, "section": heading,
                                            "strategy": "structure"}))
            idx += 1
        body = []

    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            heading = m.group(2).strip()
        else:
            body.append(line)
    flush()
    return chunks
