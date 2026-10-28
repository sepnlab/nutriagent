"""Regression tests for ablation variant dispatch (the dispatch issue).

An earlier implementation dispatched the architecture dimension with
``elif "no ReAct" in name.lower()``. Because the literal contains a capital
"R" while the haystack was lower-cased, the branch never matched and the
"no ReAct" variant silently re-ran the full pipeline. These tests pin the
explicit mapping so the bug cannot come back.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import src.evaluation.ablation as abl
from src.agent.base_agent import SpecialistAgent
from src.agent.multi_agent_engine import MultiAgentEngine


class TestArchitectureDispatch:
    CANONICAL = {
        "Single ReAct": "_run_single_react",
        "Multi-Agent (single-pass tool call)": "_run_multi_agent_single_pass",
        "Multi-Agent (no tools)": "_run_multi_agent_no_tools",
        "Multi-Agent (full)": "_run_multi_agent",
    }

    def test_mapping_is_explicit(self):
        for name, fn_name in self.CANONICAL.items():
            runner = abl._resolve_architecture_runner(name)
            assert runner is getattr(abl, fn_name), name

    def test_legacy_alias_maps_to_single_pass(self):
        """The old name is accepted but maps to the true single-pass variant."""
        assert (
            abl._resolve_architecture_runner("Multi-Agent (no ReAct)")
            is abl._run_multi_agent_single_pass
        )

    def test_unknown_variant_raises(self):
        """A typo must not silently fall back to the full pipeline."""
        with pytest.raises(ValueError):
            abl._resolve_architecture_runner("Multi-Agent (no raect)")

    def test_variant_single_uses_mapping(self, monkeypatch):
        """_run_variant_single dispatches through the mapping table."""
        called = {}

        def fake_runner(queries, tools=None, verbose=False, variant_name="", **kwargs):
            called["variant"] = variant_name
            called["n_queries"] = len(queries)
            return ["resp"], [[]]

        monkeypatch.setitem(
            abl._ARCHITECTURE_RUNNERS,
            "Multi-Agent (single-pass tool call)",
            fake_runner,
        )
        name, resp, traces, elapsed, err = abl._run_variant_single(
            {"name": "Multi-Agent (single-pass tool call)"},
            "architecture",
            [{"query": "q"}],
            [],
            False,
        )
        assert err is None
        assert name == "Multi-Agent (single-pass tool call)"
        assert called == {
            "variant": "Multi-Agent (single-pass tool call)",
            "n_queries": 1,
        }
        assert resp == ["resp"]


class _FakeTool:
    name = "feed_ingredient_bank"

    def __init__(self):
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return "Ingredient: maize\nCP: 8.5\n"


class _FakeLLM:
    """First invoke returns one tool call; second returns the final answer."""

    def __init__(self):
        self.calls = 0
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="I will look up maize.",
                tool_calls=[
                    {
                        "name": "feed_ingredient_bank",
                        "args": {"ingredient": "maize"},
                        "id": "call_1",
                    }
                ],
                usage_metadata={
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "total_tokens": 25,
                },
            )
        return AIMessage(
            content="Maize CP is 8.5%.",
            usage_metadata={
                "input_tokens": 40,
                "output_tokens": 10,
                "total_tokens": 50,
            },
        )


class _DummySpecialist(SpecialistAgent):
    agent_name = "dummy"

    def _build_prompt(self):
        return "dummy prompt"


class TestSinglePassSpecialist:
    def _make_agent(self):
        agent = object.__new__(_DummySpecialist)
        agent.tools = [_FakeTool()]
        agent.llm = _FakeLLM()
        agent.max_tokens = 200
        agent.verbose = False
        agent.single_pass = True
        return agent

    def test_one_tool_call_then_answer(self):
        agent = self._make_agent()
        result = agent.run("What is the CP of maize?")

        assert result.error is None
        assert result.response == "Maize CP is 8.5%."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["action"] == "feed_ingredient_bank"
        assert "CP: 8.5" in result.tool_calls[0]["observation"]
        # Both LLM invocations are accounted for.
        assert result.prompt_tokens == 60
        assert result.completion_tokens == 15
        # Exactly one tool execution.
        assert len(agent.tools[0].calls) == 1

    def test_no_tool_returns_direct_answer(self):
        agent = self._make_agent()
        agent.tools = []
        result = agent.run("What is CP?")
        # With no tools the single-pass path is a plain LLM call.
        assert result.response == "I will look up maize."
        assert result.tool_calls == []
        assert agent.llm.bound_tools is None


class TestValidationMode:
    def _engine(self, mode):
        engine = object.__new__(MultiAgentEngine)
        engine.validation_mode = mode
        return engine

    def test_off(self):
        decision = SimpleNamespace(requires_cross_validation=True, target_agents=["a", "b"])
        assert self._engine("off")._should_validate(decision) is False

    def test_always(self):
        decision = SimpleNamespace(requires_cross_validation=False, target_agents=["a"])
        assert self._engine("always")._should_validate(decision) is True

    def test_auto_follows_router_flag(self):
        d1 = SimpleNamespace(requires_cross_validation=True, target_agents=["a"])
        d2 = SimpleNamespace(requires_cross_validation=False, target_agents=["a"])
        d3 = SimpleNamespace(requires_cross_validation=False, target_agents=["a", "b"])
        engine = self._engine("auto")
        assert engine._should_validate(d1) is True
        assert engine._should_validate(d2) is False
        assert engine._should_validate(d3) is True

    def test_shuffled_is_a_validation_mode(self):
        decision = SimpleNamespace(requires_cross_validation=False, target_agents=["a"])
        engine = self._engine("shuffled")
        assert engine._resolve_validation_mode(decision) == "shuffled"
        assert engine._should_validate(decision) is True


class TestFactorialDispatch:
    """E1/E2 mappings must be explicit and typo-safe."""

    ARCH = {
        "Standalone LLM": "_run_standalone_llm",
        "Single ReAct + tools": "_run_single_react",
        "Single ReAct + tools + RAG": "_run_single_react_rag",
        "Single ReAct + tools + RAG + synthesis": "_run_single_react_rag_synthesis",
        "Multi-Agent (validation off)": "_run_multi_agent_validation_off",
        "Multi-Agent (validation on)": "_run_multi_agent_validation_on",
    }
    SYNTH = {
        "Concatenation": "_run_multi_agent_no_synthesis",
        "Validation + concatenation": "_run_multi_agent_validation_concat",
        "Synthesis (validation off)": "_run_multi_agent_validation_off",
        "Synthesis (shuffled warnings)": "_run_multi_agent_shuffled_validation",
        "Synthesis (correct warnings)": "_run_multi_agent_validation_on",
        "Query-only synthesis": "_run_multi_agent_query_only_synthesis",
    }

    def test_architecture_factorial_mapping(self):
        for name, fn_name in self.ARCH.items():
            runner = abl._resolve_factorial_runner(
                name, abl._ARCHITECTURE_FACTORIAL_RUNNERS, "architecture_factorial"
            )
            assert runner is getattr(abl, fn_name), name

    def test_synthesis_factorial_mapping(self):
        for name, fn_name in self.SYNTH.items():
            runner = abl._resolve_factorial_runner(
                name, abl._SYNTHESIS_FACTORIAL_RUNNERS, "synthesis_factorial"
            )
            assert runner is getattr(abl, fn_name), name

    def test_unknown_factorial_variant_raises(self):
        with pytest.raises(ValueError):
            abl._resolve_factorial_runner(
                "Standalone LLm", abl._ARCHITECTURE_FACTORIAL_RUNNERS, "architecture_factorial"
            )

    def test_factorial_dimension_uses_mapping(self, monkeypatch):
        called = {}

        def fake_runner(queries, tools=None, verbose=False, variant_name="", **kwargs):
            called["variant"] = variant_name
            return ["resp"], [[]]

        monkeypatch.setitem(
            abl._ARCHITECTURE_FACTORIAL_RUNNERS, "Standalone LLM", fake_runner
        )
        name, resp, traces, elapsed, err = abl._run_variant_single(
            {"name": "Standalone LLM"},
            "architecture_factorial",
            [{"query": "q"}],
            [],
            False,
        )
        assert err is None
        assert name == "Standalone LLM"
        assert called == {"variant": "Standalone LLM"}
        assert resp == ["resp"]
