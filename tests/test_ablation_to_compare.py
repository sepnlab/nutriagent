"""消融→对比管道测试（不依赖 LLM API）。"""

import pytest
from src.ablation_to_compare import (
    _composite_score,
    AblationParser,
    _resolve_backbone_config,
    WEIGHTS,
)


class TestCompositeScore:
    """综合评分公式测试。"""

    def test_all_metrics_present(self):
        metrics = {
            "correctness": {"mean": 4.0},
            "scientific_depth": {"mean": 3.0},
            "precision": {"mean": 5.0},
            "hallucination_n5": {"mean": 0.1},
        }
        score = _composite_score(metrics)
        expected = 4.0 * 0.4 + 3.0 * 0.3 + 5.0 * 0.2 + 0.1 * (-0.1)
        assert abs(score - expected) < 0.001

    def test_no_hallucination(self):
        """halluc 为 None 时权重归一化。"""
        metrics = {
            "correctness": {"mean": 4.0},
            "scientific_depth": {"mean": 3.0},
            "precision": {"mean": 5.0},
        }
        score = _composite_score(metrics)
        w_total = 0.4 + 0.3 + 0.2
        expected = (4.0 * 0.4 + 3.0 * 0.3 + 5.0 * 0.2) / w_total
        assert abs(score - expected) < 0.001

    def test_none_means_zero(self):
        """mean 为 None 时视为 0。"""
        metrics = {
            "correctness": {"mean": None},
            "scientific_depth": {"mean": 3.0},
            "precision": {"mean": 5.0},
        }
        score = _composite_score(metrics)
        w_total = 0.4 + 0.3 + 0.2
        expected = (0 * 0.4 + 3.0 * 0.3 + 5.0 * 0.2) / w_total
        assert abs(score - expected) < 0.001

    def test_empty_metrics(self):
        score = _composite_score({})
        assert score == 0.0

    def test_hallucination_n5_overrides_hallucination(self):
        """hallucination_n5 优先于 hallucination。"""
        metrics = {
            "correctness": {"mean": 4.0},
            "scientific_depth": {"mean": 4.0},
            "precision": {"mean": 4.0},
            "hallucination": {"mean": 0.5},
            "hallucination_n5": {"mean": 0.2},
        }
        score = _composite_score(metrics)
        expected = 4.0 * 0.4 + 4.0 * 0.3 + 4.0 * 0.2 + 0.2 * (-0.1)
        assert abs(score - expected) < 0.001


class TestAblationParser:
    """消融报告解析测试。"""

    def _make_report(self):
        m = lambda c, d, p, h=None: {
            "correctness": {"mean": c},
            "scientific_depth": {"mean": d},
            "precision": {"mean": p},
            **({"hallucination_n5": {"mean": h}} if h is not None else {}),
        }
        return {
            "llm_backbone": {
                "DeepSeek-V4-Flash": m(4.0, 3.8, 4.0),
                "Qwen3.5-27B": m(4.2, 4.0, 4.0, 0.05),
            },
            "router": {
                "LLM Router": m(4.0, 4.0, 4.0),
                "Keyword Router": m(3.5, 3.5, 3.5),
            },
            "synthesis": {
                "With Synthesis": m(4.2, 4.0, 4.0, 0.05),
                "Without Synthesis": m(3.8, 3.5, 3.8),
            },
            "rag": {
                "Multi-Agent + RAG": m(4.2, 4.0, 4.0, 0.05),
                "Multi-Agent (no RAG)": m(3.9, 3.7, 3.9),
            },
        }

    def test_pick_best(self):
        parser = AblationParser(self._make_report())
        name, score, metrics = parser.pick_best("llm_backbone")
        # DeepSeek has no hallucination penalty, so it wins despite lower raw scores
        assert name == "DeepSeek-V4-Flash"
        assert score > 0

    def test_pick_best_missing_dim(self):
        parser = AblationParser({})
        with pytest.raises(ValueError, match="not found"):
            parser.pick_best("nonexistent")

    def test_variant_scores_sorted(self):
        parser = AblationParser(self._make_report())
        ranked = parser._variant_scores("llm_backbone")
        assert ranked[0][1] >= ranked[1][1]  # descending

    def test_pick_all_best(self):
        parser = AblationParser(self._make_report())
        picks = parser.pick_all_best()
        assert "llm_backbone" in picks
        assert "router" in picks
        assert "synthesis" in picks
        assert "rag" in picks
        assert picks["llm_backbone"]["name"] == "DeepSeek-V4-Flash"

    def test_summarize(self):
        parser = AblationParser(self._make_report())
        picks = parser.pick_all_best()
        text = parser.summarize(picks)
        assert "DeepSeek-V4-Flash" in text
        assert "最优" in text


class TestResolveBackboneConfig:
    """LLM backbone 配置解析测试。"""

    def test_lookup_from_baselines(self):
        """从 models.yaml baselines 查找已知模型。"""
        result = _resolve_backbone_config("siliconflow/Qwen3.5-27B")
        assert result["provider"] == "siliconflow"
        assert result["model"] == "Qwen/Qwen3.5-27B"

    def test_fallback_deepseek(self):
        result = _resolve_backbone_config("deepseek/deepseek-v4-pro")
        assert result["provider"] == "deepseek"
        assert result["model"] == "deepseek-v4-pro"

    def test_fallback_unknown_provider(self):
        result = _resolve_backbone_config("SomeUnknownModel")
        assert result["provider"] == "siliconflow"  # default
