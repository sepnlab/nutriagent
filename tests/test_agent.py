"""Agent 模块单元测试（不依赖 LLM API）。"""

from src.agent.base_agent import AgentResult, RouterDecision


class TestAgentResult:
    def test_default_values(self):
        r = AgentResult(agent_name="feed", response="some answer")
        assert r.agent_name == "feed"
        assert r.response == "some answer"
        assert r.tool_calls == []
        assert r.domains == set()
        assert r.confidence == 1.0
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0

    def test_with_tool_calls(self):
        r = AgentResult(
            agent_name="nutrition",
            response="calcium: 0.96%",
            tool_calls=[{"action": "nutrient_requirements", "action_input": "..."}],
            domains={"Nutrient Requirements"},
            prompt_tokens=50,
            completion_tokens=30,
        )
        assert r.agent_name == "nutrition"
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0]["action"] == "nutrient_requirements"
        assert r.domains == {"Nutrient Requirements"}
        assert r.prompt_tokens == 50
        assert r.completion_tokens == 30


class TestRouterDecision:
    def test_default_values(self):
        d = RouterDecision(query="what is calcium?", intent="feed")
        assert d.query == "what is calcium?"
        assert d.intent == "feed"
        assert d.target_agents == []
        assert d.requires_cross_validation is False

    def test_multi_intent(self):
        d = RouterDecision(
            query="calcium and growth",
            intent="multi",
            target_agents=["feed", "nutrition"],
            requires_cross_validation=True,
        )
        assert d.intent == "multi"
        assert d.target_agents == ["feed", "nutrition"]
        assert d.requires_cross_validation is True


class TestRouterAgentKeywordFallback:
    """Test RouterAgent._keyword_fallback with domain keywords (no LLM)."""

    def test_feed_keyword(self):
        from src.agent.router_agent import RouterAgent
        agent = RouterAgent()
        d = agent._keyword_fallback("What is the crude protein of Soybean Meal 48%?")
        assert "feed" in d.target_agents
        assert d.intent in ("feed", "multi")

    def test_nutrition_keyword(self):
        from src.agent.router_agent import RouterAgent
        agent = RouterAgent()
        d = agent._keyword_fallback("What is the calcium requirement for Ross 308 starter?")
        assert "nutrition" in d.target_agents

    def test_performance_keyword(self):
        from src.agent.router_agent import RouterAgent
        agent = RouterAgent()
        d = agent._keyword_fallback("What is the FCR of Cobb 500 at 35 days?")
        assert "performance" in d.target_agents

    def test_definitions_keyword(self):
        from src.agent.router_agent import RouterAgent
        agent = RouterAgent()
        d = agent._keyword_fallback("What is FCR?")
        assert "definitions" in d.target_agents

    def test_chinese_feed(self):
        from src.agent.router_agent import RouterAgent
        agent = RouterAgent()
        d = agent._keyword_fallback("玉米的粗蛋白含量是多少？")
        assert "feed" in d.target_agents

    def test_empty_query(self):
        from src.agent.router_agent import RouterAgent
        agent = RouterAgent()
        d = agent._keyword_fallback("")
        targets = d.target_agents
        assert len(targets) == 4  # broadcast when no keywords match

    def test_multi_domain_fallback(self):
        from src.agent.router_agent import RouterAgent
        agent = RouterAgent()
        # "lysine" → feed (score=1), "requirement" → nutrition (score=1)
        # Both at max_score=1, threshold=0.7 → both qualify
        d = agent._keyword_fallback("What is the lysine requirement for broilers?")
        assert len(d.target_agents) >= 2


class TestSynthesisCrossValidate:
    """Test SynthesisAgent.cross_validate (consistency validator, no LLM)."""

    def test_no_contradiction(self):
        from src.agent.synthesis_agent import SynthesisAgent
        from src.agent.base_agent import AgentResult
        agent = SynthesisAgent()
        results = [
            AgentResult(agent_name="feed", response="Ca: 0.96%, P: 0.45%"),
            AgentResult(agent_name="nutrition", response="Ca: 0.96%, P: 0.45%"),
        ]
        warnings = agent.cross_validate(results)
        # Ca/P = 0.96/0.45 = 2.13 → within normal range, no warnings
        assert len(warnings) == 0

    def test_ca_p_ratio_not_flagged_by_default(self):
        """The generic Ca:P band was removed (it flagged normal layer diets).

        A segment naming both Ca and P is skipped rather than mis-aligned, so
        the legacy "Ca: 0.50%, P: 0.45%" input yields no warning. The opt-in
        Ca:P policy is covered in tests/test_validation.py.
        """
        from src.agent.synthesis_agent import SynthesisAgent
        from src.agent.base_agent import AgentResult
        agent = SynthesisAgent()
        results = [
            AgentResult(agent_name="feed", response="Ca: 0.50%, P: 0.45%"),
            AgentResult(agent_name="nutrition", response="Ca: 0.50%, P: 0.45%"),
        ]
        warnings = agent.cross_validate(results)
        assert warnings == []

    def test_single_result_no_warnings(self):
        from src.agent.synthesis_agent import SynthesisAgent
        from src.agent.base_agent import AgentResult
        agent = SynthesisAgent()
        results = [
            AgentResult(agent_name="feed", response="Crude protein: 23%"),
        ]
        warnings = agent.cross_validate(results)
        assert len(warnings) == 0


class TestDomainExtraction:
    """Test tool domain extraction logic (no LLM)."""

    def test_extract_tool_domains(self):
        from src.agent.hybrid_engine import NutriCHATEngine
        from src.tools import get_all_tools
        engine = NutriCHATEngine(tools=get_all_tools(), mode="single", rag_enabled=False)
        steps = [
            {"action": "feed_ingredient_bank"},
            {"action": "nutrient_requirements"},
        ]
        domains = engine._extract_tool_domains(steps)
        assert "Feed Ingredient Bank" in domains
        assert "Nutrient Requirements" in domains

    def test_empty_steps(self):
        from src.agent.hybrid_engine import NutriCHATEngine
        from src.tools import get_all_tools
        engine = NutriCHATEngine(tools=get_all_tools(), mode="single", rag_enabled=False)
        domains = engine._extract_tool_domains([])
        assert domains == set()

    def test_unknown_action(self):
        from src.agent.hybrid_engine import NutriCHATEngine
        from src.tools import get_all_tools
        engine = NutriCHATEngine(tools=get_all_tools(), mode="single", rag_enabled=False)
        steps = [{"action": "some_unknown_tool"}]
        domains = engine._extract_tool_domains(steps)
        assert domains == set()


class TestCacheKey:
    """Test hybrid engine cache key generation."""

    def test_cache_key_deterministic(self):
        from src.agent.hybrid_engine import NutriCHATEngine
        from src.tools import get_all_tools
        import os
        os.environ["SPECIES"] = "broiler"
        # Need to reload or use a fresh instance
        engine = NutriCHATEngine(tools=get_all_tools(), mode="single", rag_enabled=False)
        key1 = engine._cache_key("What is calcium?")
        key2 = engine._cache_key("What is calcium?")
        assert key1 == key2

    def test_cache_key_differs_for_different_queries(self):
        from src.agent.hybrid_engine import NutriCHATEngine
        from src.tools import get_all_tools
        engine = NutriCHATEngine(tools=get_all_tools(), mode="single", rag_enabled=False)
        key1 = engine._cache_key("What is calcium?")
        key2 = engine._cache_key("What is protein?")
        assert key1 != key2
