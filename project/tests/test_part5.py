"""Tests for the Part V agent modules: tools, agents, multi_agent, memory."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from llm_platform.tools import (
    Tool, ToolRegistry, ToolExecutor, ToolResult, make_calculator_tool, safe_calculator,
)
from llm_platform.agents import ReActAgent, RuleBasedPolicy
from llm_platform.multi_agent import Supervisor, Blackboard, make_default_team
from llm_platform.memory import (
    ShortTermMemory, LongTermMemory, WorkingMemory, AgentMemory,
)


# --- tools ---------------------------------------------------------------
def test_tool_validation_blocks_bad_call():
    t = make_calculator_tool()
    assert t.validate({"expression": "2+2"}) == []
    assert t.validate({}) != []                          # missing required
    assert t.validate({"expression": 5}) != []           # wrong type
    assert t.validate({"bogus": "x"}) != []              # unknown param


def test_executor_validates_before_running():
    reg = ToolRegistry(); reg.register(make_calculator_tool())
    ex = ToolExecutor(reg)
    bad = ex.execute("calculator", {})                   # invalid -> not executed
    assert bad.ok is False and "missing" in bad.error
    good = ex.execute("calculator", {"expression": "2 * (3 + 4)"})
    assert good.ok and good.output == 14


def test_executor_caches_deterministic_tools():
    reg = ToolRegistry(); reg.register(make_calculator_tool())
    ex = ToolExecutor(reg)
    first = ex.execute("calculator", {"expression": "6*7"})
    second = ex.execute("calculator", {"expression": "6*7"})
    assert first.output == 42 and second.output == 42
    assert second.cached is True


def test_executor_retries_then_fails():
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        raise RuntimeError("transient")
    reg = ToolRegistry()
    reg.register(Tool("flaky", "always fails", {}, flaky))
    ex = ToolExecutor(reg, max_retries=2, base_backoff=0.0)
    res = ex.execute("flaky", {})
    assert res.ok is False and res.attempts == 3         # 1 + 2 retries
    assert calls["n"] == 3


def test_unknown_tool():
    ex = ToolExecutor(ToolRegistry())
    assert ex.execute("nope", {}).ok is False


def test_safe_calculator_rejects_code():
    with pytest.raises(ValueError):
        safe_calculator("__import__('os').system('ls')")


# --- ReAct agent ---------------------------------------------------------
def test_react_agent_uses_calculator():
    reg = ToolRegistry(); reg.register(make_calculator_tool())
    agent = ReActAgent(reg)
    result = agent.run("What is 12 * (3 + 4)?")
    assert result.success
    assert any(s.action == "calculator" for s in result.steps)
    assert "84" in result.answer


def test_react_agent_terminates():
    reg = ToolRegistry(); reg.register(make_calculator_tool())
    agent = ReActAgent(reg, max_steps=3)
    result = agent.run("Just say hello.")
    assert isinstance(result.answer, str)
    assert len(result.steps) <= 3


# --- multi-agent ---------------------------------------------------------
def test_supervisor_routes_to_specialist():
    sup = Supervisor(make_default_team())
    assert sup.route("research the facts").name == "researcher"
    assert sup.route("implement the function").name == "coder"
    assert sup.route("review and verify").name == "reviewer"


def test_supervisor_decomposes_and_runs():
    sup = Supervisor(make_default_team())
    results, bb = sup.run("research the topic then implement the function then review and verify the result")
    names = [n for n, _ in results]
    assert names == ["researcher", "coder", "reviewer"]
    assert bb.read("task") is not None
    assert len(bb.log) >= 4


# --- memory --------------------------------------------------------------
def test_short_term_window_and_summary():
    stm = ShortTermMemory(max_turns=4)
    for i in range(8):
        stm.add("user", f"message number {i}")
    assert len(stm.turns) == 4
    assert stm.summary                                   # older turns summarized
    assert "message number 7" in stm.context()


def test_long_term_recall_and_forget():
    ltm = LongTermMemory()
    t = 1000.0
    ltm.write("the user prefers dark mode", now=t)
    ltm.write("the capital of France is Paris", now=t)
    hits = ltm.recall("what theme does the user like", k=1, now=t)
    assert "dark mode" in hits[0]
    # an unused, very old memory is forgotten
    ltm.write("ephemeral note", now=t)
    removed = ltm.forget(max_age_s=10, now=t + 10_000)
    assert removed >= 1


def test_long_term_update():
    ltm = LongTermMemory()
    ltm.write("user lives in Berlin")
    assert ltm.update("user lives in Berlin", "user lives in Munich")
    assert "Munich" in ltm.recall("where does the user live", k=1)[0]


def test_working_memory_progress():
    wm = WorkingMemory()
    wm.set_plan(["a", "b", "c"])
    wm.complete("a")
    assert wm.progress() == "1/3 steps complete"


def test_agent_memory_assembles_context():
    mem = AgentMemory()
    mem.observe("user", "I am planning a trip to Japan")
    mem.remember("user is vegetarian")
    ctx = mem.assemble_context("food preferences", k=1)
    assert "vegetarian" in ctx
    assert "Japan" in ctx
