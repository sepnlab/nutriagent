"""Faithfulness 模块单元测试（不依赖 LLM API）。"""

import pytest
from src.evaluation.faithfulness import (
    _parse_tool_observations,
    _parse_response_values,
    compute_faithfulness_from_steps,
    compute_faithfulness_batch,
)


class TestParseToolObservations:
    def test_feed_ingredient_format(self):
        """从 Feed Ingredient Bank 输出中提取数值。"""
        obs = """Feed Ingredient: Corn Grain
----------------------------------------
COMPOSITION:
  DM: 89.0
  CP: 8.5
  ME_kcal_kg: 3350
  Ca: 0.02
  P_total: 0.28"""
        steps = [{"action": "feed_ingredient_bank", "observation": obs}]
        pairs = _parse_tool_observations(steps)
        assert ("DM", 89.0) in pairs
        assert ("CP", 8.5) in pairs
        assert ("ME_kcal_kg", 3350.0) in pairs
        assert ("Ca", 0.02) in pairs

    def test_nutrient_requirements_format(self):
        """从 Nutrient Requirements 输出中提取数值。"""
        obs = """Nutrient Requirements: Ross 308 | Starter
--------------------------------------------------
MACRO NUTRIENTS:
  ME_kcal_kg: 3025
  CP_pct: 22.0
  Ca_pct: 0.96
  Avail_P_pct: 0.48"""
        steps = [{"action": "nutrient_requirements", "observation": obs}]
        pairs = _parse_tool_observations(steps)
        assert ("ME_kcal_kg", 3025.0) in pairs
        assert ("CP_pct", 22.0) in pairs
        assert ("Ca_pct", 0.96) in pairs

    def test_performance_format(self):
        """从 Performance Management 输出中提取数值。"""
        obs = """Performance: Ross 308 | Day 35 | Mixed
--------------------------------------------------
  BW_g: 2150
  ADG_g: 62
  FCR: 1.55
  Mortality_pct: 3.5"""
        steps = [{"action": "performance_management", "observation": obs}]
        pairs = _parse_tool_observations(steps)
        assert ("BW_g", 2150.0) in pairs
        assert ("ADG_g", 62.0) in pairs
        assert ("FCR", 1.55) in pairs

    def test_empty_steps(self):
        """空 steps 返回空列表。"""
        assert _parse_tool_observations([]) == []

    def test_no_observation(self):
        """没有 observation 的 step 被跳过。"""
        steps = [{"action": "feed_ingredient_bank", "observation": ""}]
        assert _parse_tool_observations(steps) == []

    def test_unknown_action_skipped(self):
        """action 为 unknown 的 step 被跳过。"""
        obs = "DM: 89.0"
        steps = [{"action": "unknown", "observation": obs}]
        assert _parse_tool_observations(steps) == []


class TestParseResponseValues:
    def test_simple_values(self):
        """提取回答中的数值对。"""
        text = "Corn grain contains 8.5% crude protein and 3350 kcal/kg ME."
        pairs = _parse_response_values(text)
        values = [v for _, v in pairs]
        assert 8.5 in values
        assert 3350.0 in values

    def test_multiple_values(self):
        """提取多个不同的数值。"""
        text = "CP: 22.0%, Ca: 0.96%, ME: 3025 kcal/kg"
        pairs = _parse_response_values(text)
        values = [v for _, v in pairs]
        assert 22.0 in values
        assert 0.96 in values
        assert 3025.0 in values

    def test_no_numbers(self):
        """没有数字的回答返回空列表。"""
        text = "Corn grain is a common feed ingredient."
        assert _parse_response_values(text) == []

    def test_year_value_filtered(self):
        """年份值应被过滤（避免系统性压低 Faithfulness）。"""
        text = "Published in 2025"
        pairs = _parse_response_values(text)
        values = [v for _, v in pairs]
        assert 2025.0 not in values


class TestComputeFaithfulness:
    def test_fully_faithful(self):
        """所有数值都来自工具返回。"""
        response = "Corn grain contains 8.5% CP and 3350 kcal/kg ME."
        steps = [{
            "action": "feed_ingredient_bank",
            "observation": "Feed Ingredient: Corn Grain\n  CP: 8.5\n  ME_kcal_kg: 3350",
        }]
        score = compute_faithfulness_from_steps(response, steps)
        assert score == 1.0

    def test_fully_unfaithful(self):
        """所有数值都不是工具返回的。"""
        response = "Corn grain contains 12.0% CP and 2800 kcal/kg ME."
        steps = [{
            "action": "feed_ingredient_bank",
            "observation": "Feed Ingredient: Corn Grain\n  CP: 8.5\n  ME_kcal_kg: 3350",
        }]
        score = compute_faithfulness_from_steps(response, steps)
        assert score == 0.0

    def test_partially_faithful(self):
        """部分数值匹配。"""
        response = "Corn grain contains 8.5% CP and 2800 kcal/kg ME."
        steps = [{
            "action": "feed_ingredient_bank",
            "observation": "Feed Ingredient: Corn Grain\n  CP: 8.5\n  ME_kcal_kg: 3350",
        }]
        score = compute_faithfulness_from_steps(response, steps)
        assert score == 0.5  # 8.5 matches, 2800 doesn't

    def test_tolerance_match(self):
        """数值在 5% 容差内也算匹配。"""
        response = "CP is about 8.3%."
        steps = [{
            "action": "feed_ingredient_bank",
            "observation": "Feed Ingredient: Corn Grain\n  CP: 8.5",
        }]
        score = compute_faithfulness_from_steps(response, steps)
        assert score == 1.0  # 8.3 within 5% of 8.5

    def test_no_tool_observations(self):
        """没有工具观测值时返回 None。"""
        response = "CP is 8.5%."
        score = compute_faithfulness_from_steps(response, [])
        assert score is None

    def test_no_numeric_claims(self):
        """回答中没有数值时返回 None。"""
        response = "Corn grain is a common feed ingredient."
        steps = [{
            "action": "feed_ingredient_bank",
            "observation": "Feed Ingredient: Corn Grain\n  CP: 8.5",
        }]
        score = compute_faithfulness_from_steps(response, steps)
        assert score is None

    def test_multi_step(self):
        """多个 tool call 的观测值合并。"""
        response = "CP: 22.0%, BW: 2150g"
        steps = [
            {
                "action": "nutrient_requirements",
                "observation": "Nutrient Requirements: Ross 308 | Starter\n  CP_pct: 22.0",
            },
            {
                "action": "performance_management",
                "observation": "Performance: Ross 308 | Day 35\n  BW_g: 2150",
            },
        ]
        score = compute_faithfulness_from_steps(response, steps)
        assert score == 1.0


class TestComputeFaithfulnessBatch:
    def test_batch(self):
        """批处理返回等长的分数列表。"""
        responses = [
            "CP is 8.5%.",
            "CP is 12.0%.",
        ]
        step_traces = [
            [{"action": "feed_ingredient_bank", "observation": "CP: 8.5"}],
            [{"action": "feed_ingredient_bank", "observation": "CP: 8.5"}],
        ]
        scores = compute_faithfulness_batch(responses, step_traces)
        assert scores[0] == 1.0
        assert scores[1] == 0.0
        assert len(scores) == 2


class TestObservationToSentences:
    """Test the tool observation → natural language conversion."""

    def test_key_value_to_sentence(self):
        from src.evaluation.faithfulness import _observation_to_sentences
        obs = "  CP: 8.5\n  ME_kcal_kg: 3350\n  Ca: 0.02"
        sentences = _observation_to_sentences(obs)
        assert len(sentences) == 3
        assert any("CP" in s and "8.5" in s for s in sentences)
        assert any("3350" in s for s in sentences)  # ME kcal kg → 3350
        assert any("Ca" in s and "0.02" in s for s in sentences)

    def test_deduplicates_values(self):
        from src.evaluation.faithfulness import _observation_to_sentences
        obs = "  CP: 8.5\n  CP: 8.5"
        sentences = _observation_to_sentences(obs)
        assert len(sentences) == 1

    def test_short_key_skipped(self):
        from src.evaluation.faithfulness import _observation_to_sentences
        obs = "  X: 1.0\n  Y: 2.0"
        sentences = _observation_to_sentences(obs)
        # "X" and "Y" are < 2 chars, may be filtered
        for s in sentences:
            assert len(s.split()[1]) >= 2  # key after "The" should be >= 2 chars


class TestSplitClaims:
    """Test response → verifiable claim splitting."""

    def test_simple_sentences(self):
        from src.evaluation.faithfulness import _split_claims
        text = "Corn grain is a common feed ingredient. It contains 8.5% crude protein."
        claims = _split_claims(text)
        assert len(claims) == 2
        assert "Corn grain" in claims[0]
        assert "crude protein" in claims[1]

    def test_numeric_heavy_sentence_filtered(self):
        from src.evaluation.faithfulness import _split_claims
        # First sentence has >=50% numeric tokens → filtered
        # (CP:, 8.5,, Ca:, 0.96,, ME:, 3025. → 3/6 are purely numeric)
        text = "CP: 8.5, Ca: 0.96, ME: 3025. Corn is a common feed ingredient."
        claims = _split_claims(text)
        assert len(claims) == 1
        assert "feed ingredient" in claims[0]

    def test_short_sentence_filtered(self):
        from src.evaluation.faithfulness import _split_claims
        text = "Hi. Corn grain contains 8.5% crude protein."
        claims = _split_claims(text)
        assert all(len(c) >= 15 for c in claims)
        assert "crude protein" in claims[0]

    def test_empty_text(self):
        from src.evaluation.faithfulness import _split_claims
        assert _split_claims("") == []

    def test_single_sentence(self):
        from src.evaluation.faithfulness import _split_claims
        text = "Corn grain is a high-energy feed ingredient used in poultry diets."
        claims = _split_claims(text)
        assert len(claims) == 1
