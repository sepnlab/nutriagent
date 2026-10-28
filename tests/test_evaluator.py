"""Evaluator 模块单元测试（不依赖 LLM API）。"""

import json
import pytest
from src.evaluation.evaluator import (
    parse_numeric,
    check_tolerance,
    evaluate_tolerance,
    compute_flesch,
    EvalReport,
)
from src.evaluation.evaluator_llm import EvaluatorLLM


class TestParseNumeric:
    def test_simple_number(self):
        vals = parse_numeric("The value is 23.5%")
        assert any(abs(v[0] - 23.5) < 0.01 for v in vals)

    def test_range(self):
        vals = parse_numeric("between 20 and 25%")
        ranges = [v for v in vals if v[1]]
        assert any(abs(v[0] - 22.5) < 0.01 for v in ranges)

    def test_multiple_values(self):
        vals = parse_numeric("CP: 23%, Ca: 0.96%")
        values = [v[0] for v in vals]
        assert 23.0 in values
        assert 0.96 in values


class TestCheckTolerance:
    def test_within_2pct(self):
        assert check_tolerance(100, 101, 0.02) is True   # 101 in [98, 102]

    def test_outside_2pct(self):
        assert check_tolerance(100, 103, 0.02) is False  # 103 not in [98, 102]

    def test_within_5pct(self):
        assert check_tolerance(100, 104, 0.05) is True   # 104 in [95, 105]

    def test_exact_match(self):
        assert check_tolerance(23.0, 23.0, 0.02) is True


    def test_symmetric_relative_error(self):
        # Old interval rule accepted g=95 for p=100 at tau=0.05; relative error
        # is 5.26% and must be rejected.
        assert check_tolerance(100, 95, 0.05) is False
        assert check_tolerance(95, 100, 0.05) is True


class TestEvaluateTolerance:
    def test_correct_value(self):
        assert evaluate_tolerance("The result is 23.0%", 23.0, 0.02) == 1.0

    def test_wrong_value(self):
        assert evaluate_tolerance("The result is 10.0%", 23.0, 0.02) == 0.0

    def test_unparseable(self):
        assert evaluate_tolerance("No numbers here", 23.0, 0.02) is None


class TestComputeFlesch:
    def test_easy_text(self):
        score = compute_flesch("The cat sat on the mat. The dog ran in the park.")
        # Flesch can exceed 100 for very short/simple texts
        assert score > 50

    def test_empty_text(self):
        assert compute_flesch("") == 0.0

    def test_short_text(self):
        # Flesch can exceed 100 for very short texts; just verify it computes
        score = compute_flesch("Hi")
        assert isinstance(score, (int, float))


class TestEvalReport:
    def test_empty_report(self):
        r = EvalReport("test")
        s = r.summary()
        assert s["method"] == "test"
        assert s["correctness"]["mean"] is None

    def test_single_score(self):
        r = EvalReport("test")
        r.add(c=4.0, p=3.5, d=5.0)
        s = r.summary()
        assert s["correctness"]["mean"] == 4.0
        assert s["overall_weighted_score"] == pytest.approx((4 + 3.5 + 5) / 3)

    def test_flesch_and_halluc(self):
        r = EvalReport("test")
        r.add(flesch=75.0, halluc=0.05)
        s = r.summary()
        assert s["flesch_reading_ease"]["mean"] == 75.0
        assert s["inconsistency"]["mean"] == 0.05




class TestExtractJson:
    """EvaluatorLLM._extract_json 的 JSON 抽取测试。"""

    def test_code_fence_json(self):
        text = '```json\n{"key": "value"}\n```'
        result = EvaluatorLLM._extract_json(text)
        assert '"key": "value"' in result

    def test_code_fence_bare(self):
        text = '```\n{"a": 1}\n```'
        result = EvaluatorLLM._extract_json(text)
        assert '"a": 1' in result

    def test_bare_object(self):
        text = 'Some text {"correctness": 5, "precision": 4} more text'
        result = EvaluatorLLM._extract_json(text)
        assert '"correctness": 5' in result

    def test_nested_braces(self):
        text = '{"outer": {"inner": "value"}}'
        result = EvaluatorLLM._extract_json(text)
        assert '"outer"' in result

    def test_no_json_fallback(self):
        text = "Just plain text with no JSON"
        result = EvaluatorLLM._extract_json(text)
        assert result == text


class TestInconsistencyRatio:
    """run_evaluation_pipeline 的 inconsistency_ratio 采样逻辑。"""

    _LONG_RESP = "This response is longer than twenty characters."

    def _make_scorer(self):
        class _Mock:
            def compute_score_multi_n(self, resp, query, n_values=None):
                return {5: 0.5}
        return _Mock().compute_score_multi_n

    def _run(self, queries, ratio):
        from src.evaluation.evaluator import run_evaluation_pipeline
        responses = {"m": [self._LONG_RESP for _ in queries]}
        reports = run_evaluation_pipeline(
            queries=queries, nutrichat_responses=[], baseline_responses=responses,
            selfcheck=self._make_scorer(), n_selfcheck_values=[5],
            enable_tolerance=False, enable_flesch=False,
            inconsistency_ratio=ratio,
        )
        s = reports["m"].summary()
        return s.get("inconsistency_n5", {}).get("n", 0)

    def test_ratio_zero_skips_all(self):
        q = [{"id": f"q{i}", "query": f"test {i}"} for i in range(10)]
        assert self._run(q, 0.0) == 0

    def test_ratio_one_runs_all(self):
        q = [{"id": f"q{i}", "query": f"test {i}"} for i in range(10)]
        assert self._run(q, 1.0) == 10

    def test_ratio_half_rounds_correctly(self):
        q = [{"id": f"q{i}", "query": f"test {i}"} for i in range(20)]
        n = self._run(q, 0.5)
        assert 8 <= n <= 12, f"expected ~10, got {n}"
