"""Conversation storage and session management (Chapter 26).

Conversations are persisted message logs that can branch (fork) for "edit and
regenerate" flows. Sessions are the runtime envelope -- created, persisted,
recovered across devices, and expired by TTL. Inference stays stateless
(Chapter 2); all conversation state lives here.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Message:
    role: str
    content: str
    ts: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class Conversation:
    conv_id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    parent_id: Optional[str] = None       # set when this is a fork
    forked_at: Optional[int] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def append(self, role: str, content: str) -> Message:
        m = Message(role, content)
        self.messages.append(m)
        return m


class ConversationStore:
    """In-memory store; swap for a database keeping the same interface."""

    def __init__(self) -> None:
        self._convs: Dict[str, Conversation] = {}

    def create(self, user_id: str, **metadata) -> Conversation:
        cid = uuid.uuid4().hex[:12]
        conv = Conversation(cid, user_id, metadata=metadata)
        self._convs[cid] = conv
        return conv

    def get(self, conv_id: str) -> Optional[Conversation]:
        return self._convs.get(conv_id)

    def list_for_user(self, user_id: str) -> List[Conversation]:
        return [c for c in self._convs.values() if c.user_id == user_id]

    def fork(self, conv_id: str, at_index: int) -> Conversation:
        """Branch a conversation at a message index (edit-and-regenerate)."""
        parent = self._convs[conv_id]
        cid = uuid.uuid4().hex[:12]
        branch = Conversation(
            cid, parent.user_id,
            messages=[Message(m.role, m.content, m.ts, m.msg_id)
                      for m in parent.messages[:at_index]],
            parent_id=conv_id, forked_at=at_index,
            metadata=dict(parent.metadata),
        )
        self._convs[cid] = branch
        return branch


# --- conversation memory helpers ----------------------------------------
def sliding_window(messages: List[Message], max_turns: int) -> List[Message]:
    return messages[-max_turns:]


def summarize_history(messages: List[Message], keep_recent: int = 4) -> str:
    """Compress older turns into a one-line summary, keep recent verbatim.

    A real system uses an LLM to summarize; this concatenation is a deterministic
    stand-in with the same shape (summary + recent window).
    """
    if len(messages) <= keep_recent:
        return ""
    old = messages[:-keep_recent]
    return "[summary] " + " ".join(m.content for m in old)[:400]


def extract_kv(messages: List[Message]) -> Dict[str, str]:
    """Pull simple 'my X is Y' style facts from a conversation (KV extraction)."""
    import re
    facts: Dict[str, str] = {}
    # Value is a single token so multiple "my X is Y" facts in one sentence
    # (joined by "and") are captured separately rather than greedily merged.
    pat = re.compile(r"\bmy (\w+) is ([\w@.\-]+)", re.I)
    for m in messages:
        if m.role != "user":
            continue
        for key, val in pat.findall(m.content):
            facts[key.lower()] = val.strip()
    return facts


# --- session management --------------------------------------------------
@dataclass
class Session:
    session_id: str
    user_id: str
    conv_id: str
    created_at: float
    last_seen: float


class SessionManager:
    """Stateful session envelope with TTL expiry and cross-device continuity."""

    def __init__(self, store: ConversationStore, *, ttl_seconds: int = 3600) -> None:
        self.store = store
        self.ttl = ttl_seconds
        self._sessions: Dict[str, Session] = {}

    def start(self, user_id: str, *, now: Optional[float] = None) -> Session:
        now = now if now is not None else time.time()
        conv = self.store.create(user_id)
        sid = uuid.uuid4().hex[:12]
        s = Session(sid, user_id, conv.conv_id, now, now)
        self._sessions[sid] = s
        return s

    def touch(self, session_id: str, *, now: Optional[float] = None) -> Optional[Session]:
        now = now if now is not None else time.time()
        s = self._sessions.get(session_id)
        if s and (now - s.last_seen) <= self.ttl:
            s.last_seen = now
            return s
        if s:                                    # expired
            del self._sessions[session_id]
        return None

    def resume_for_user(self, user_id: str) -> Optional[Session]:
        """Cross-device continuity: find the user's most recent live session."""
        live = [s for s in self._sessions.values() if s.user_id == user_id]
        return max(live, key=lambda s: s.last_seen) if live else None

    def expire(self, *, now: Optional[float] = None) -> int:
        now = now if now is not None else time.time()
        stale = [sid for sid, s in self._sessions.items() if (now - s.last_seen) > self.ttl]
        for sid in stale:
            del self._sessions[sid]
        return len(stale)
