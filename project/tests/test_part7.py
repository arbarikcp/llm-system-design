"""Tests for the Part VII data-platform modules."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from llm_platform.context_assembler import (
    PromptTemplate, PromptLibrary, ContextBlock, ContextAssembler,
)
from llm_platform.feedback import (
    Feedback, FeedbackStore, quality_filter, ActiveLearner, Candidate,
)
from llm_platform.sessions import (
    ConversationStore, SessionManager, Message,
    sliding_window, summarize_history, extract_kv,
)


# --- context assembly ----------------------------------------------------
def test_template_render_and_missing():
    t = PromptTemplate("greet", "Hello {name}, welcome to {place}.")
    assert t.render(name="Alice", place="Paris") == "Hello Alice, welcome to Paris."
    with pytest.raises(KeyError):
        t.render(name="Alice")


def test_template_value_with_braces_is_safe():
    t = PromptTemplate("x", "Value: {v}")
    assert t.render(v="{not_a_var}") == "Value: {not_a_var}"


def test_library_ab_is_deterministic():
    lib = PromptLibrary()
    lib.register(PromptTemplate("sys", "A", version="a"))
    lib.register(PromptTemplate("sys", "B", version="b"))
    pick1 = lib.choose("sys", user_id="u1", weights={"a": 0.5, "b": 0.5})
    pick2 = lib.choose("sys", user_id="u1", weights={"a": 0.5, "b": 0.5})
    assert pick1.version == pick2.version          # same user -> same bucket


def test_assembler_keeps_required_drops_low_priority():
    asm = ContextAssembler(max_tokens=60)          # ~240 chars
    blocks = [
        ContextBlock("system", "S" * 40, required=True),
        ContextBlock("docs", "D" * 120, priority=10),
        ContextBlock("history", "H" * 120, priority=5),
        ContextBlock("profile", "P" * 120, priority=1),
    ]
    out = asm.assemble(blocks)
    assert "system" in out.included                 # required always in
    assert out.dropped                              # low priority dropped
    assert out.used_tokens <= 60


def test_assembler_truncates_oversized_required():
    asm = ContextAssembler(max_tokens=20)
    out = asm.assemble([ContextBlock("system", "X" * 1000, required=True)])
    assert "system" in out.included and "system" in out.truncated
    assert out.used_tokens <= 22


# --- feedback ------------------------------------------------------------
def test_feedback_polarity():
    assert Feedback("r", "explicit", "up").polarity() == 1
    assert Feedback("r", "explicit", "down").polarity() == -1
    assert Feedback("r", "implicit", "regenerate").polarity() == -1
    assert Feedback("r", "implicit", "copy").polarity() == 1


def test_feedback_store_satisfaction_and_sft():
    st = FeedbackStore()
    st.add(Feedback("1", "explicit", "up", query="q1", response="good answer"))
    st.add(Feedback("2", "explicit", "down", query="q2", response="bad"))
    st.add(Feedback("3", "implicit", "copy", query="q3", response="copied answer"))
    assert abs(st.satisfaction_rate() - 2 / 3) < 1e-9
    sft = st.to_sft_examples()
    assert len(sft) == 2 and all("prompt" in e for e in sft)


def test_preference_pairs():
    st = FeedbackStore()
    st.add(Feedback("1", "comparative", "pref", query="q", chosen="A", rejected="B"))
    pairs = st.to_preference_pairs()
    assert pairs == [{"prompt": "q", "chosen": "A", "rejected": "B"}]


def test_quality_filter_dedup_and_length():
    ex = [{"prompt": "p", "completion": "ok answer"},
          {"prompt": "p", "completion": "ok answer"},   # dup
          {"prompt": "q", "completion": "x"},           # too short
          {"prompt": "r", "completion": ""}]            # empty
    assert len(quality_filter(ex)) == 1


def test_active_learning_picks_uncertain():
    al = ActiveLearner()
    cands = [Candidate("a", "", 0.9), Candidate("b", "", 0.2), Candidate("c", "", 0.5)]
    picked = al.select(cands, k=2)
    assert [c.item_id for c in picked] == ["b", "c"]


# --- sessions ------------------------------------------------------------
def test_conversation_append_and_fork():
    store = ConversationStore()
    conv = store.create("u1")
    conv.append("user", "hello")
    conv.append("assistant", "hi there")
    conv.append("user", "tell me a joke")
    branch = store.fork(conv.conv_id, at_index=2)
    assert branch.parent_id == conv.conv_id
    assert len(branch.messages) == 2               # forked before the 3rd message
    branch.append("user", "different follow-up")
    assert len(conv.messages) == 3                 # original unchanged


def test_session_ttl_and_resume():
    store = ConversationStore()
    mgr = SessionManager(store, ttl_seconds=100)
    s = mgr.start("u1", now=1000.0)
    assert mgr.touch(s.session_id, now=1050.0) is not None
    assert mgr.touch(s.session_id, now=2000.0) is None     # expired
    s2 = mgr.start("u1", now=3000.0)
    assert mgr.resume_for_user("u1").session_id == s2.session_id


def test_conversation_memory_helpers():
    msgs = [Message("user", f"turn {i}") for i in range(6)]
    assert len(sliding_window(msgs, 3)) == 3
    assert summarize_history(msgs, keep_recent=2).startswith("[summary]")
    kv = extract_kv([Message("user", "my name is Alice and my city is Berlin")])
    assert kv.get("name") == "Alice" or kv.get("city") == "Berlin"
