"""EvaluatorLLM 模块测试（不依赖 LLM API）。"""

import pytest
from src.evaluation.evaluator_llm import EvaluatorLLM


class TestExtractJsonFromMarkdown:
    """Test _extract_json() with various LLM output formats."""

    def test_code_fence_json(self):
        text = "```json\n{\"scores\": [{\"response_id\": \"R1\", \"correctness\": 5}]}\n```"
        result = EvaluatorLLM._extract_json(text)
        assert "\"correctness\": 5" in result

    def test_code_fence_bare(self):
        text = "```\n{\"scores\": [{\"response_id\": \"R1\", \"correctness\": 4}]}\n```"
        result = EvaluatorLLM._extract_json(text)
        assert "\"correctness\": 4" in result

    def test_bare_object(self):
        text = '{"scores": [{"response_id": "R1", "correctness": 3}]}'
        result = EvaluatorLLM._extract_json(text)
        assert "\"correctness\": 3" in result

    def test_text_before_json(self):
        text = 'Here is the result:\n{"scores": [{"response_id": "R1", "correctness": 5}]}\n'
        result = EvaluatorLLM._extract_json(text)
        assert "\"correctness\": 5" in result

    def test_nested_braces(self):
        text = '{"scores": [{"response_id": "R1", "correctness": 5, "justification": "a {b} c"}]}'
        result = EvaluatorLLM._extract_json(text)
        assert "a {b} c" in result

    def test_no_json_returns_original(self):
        text = "Just a plain text response with no JSON."
        result = EvaluatorLLM._extract_json(text)
        assert result == text.strip()


class TestRepairJson:
    """Test _repair_json() for common LLM JSON formatting errors."""

    def test_trailing_comma(self):
        text = '{"scores": [{"correctness": 5,}]}'
        result = EvaluatorLLM._repair_json(text)
        import json
        parsed = json.loads(result)
        assert parsed["scores"][0]["correctness"] == 5

    def test_single_quote_keys(self):
        text = "{'scores': [{'response_id': 'R1', 'correctness': 5}]}"
        result = EvaluatorLLM._repair_json(text)
        import json
        parsed = json.loads(result)
        assert parsed["scores"][0]["correctness"] == 5

    def test_single_quote_values(self):
        text = '{"key": \'value with spaces\'}'
        result = EvaluatorLLM._repair_json(text)
        import json
        parsed = json.loads(result)
        assert parsed["key"] == "value with spaces"

    def test_truncated_json(self):
        text = '{"scores": [{"correctness": 5, "precision": 4}'
        result = EvaluatorLLM._repair_json(text)
        assert result.endswith("}")

    def test_trailing_comma_in_nested(self):
        text = '{"scores": [{"correctness": 5, "precision": 4,}],}'
        result = EvaluatorLLM._repair_json(text)
        import json
        parsed = json.loads(result)
        assert parsed["scores"][0]["correctness"] == 5

    def test_code_fence_removal(self):
        text = '```\n{"key": "value"}\n```'
        result = EvaluatorLLM._repair_json(text)
        assert result == '{"key": "value"}'


class TestModelCapabilityCheck:
    """Test _normalize_model and _model_flavor."""

    def test_normalize_deepseek_full(self):
        assert EvaluatorLLM._normalize_model("deepseek-ai/DeepSeek-V4-Flash") == "deepseek-v4-flash"

    def test_normalize_deepseek_bare(self):
        assert EvaluatorLLM._normalize_model("deepseek-v4-pro") == "deepseek-v4-pro"

    def test_normalize_strips_instruct(self):
        result = EvaluatorLLM._normalize_model("Qwen/Qwen3.5-27B-Instruct")
        assert result.endswith("-27b")

    def test_normalize_strips_beta(self):
        result = EvaluatorLLM._normalize_model("model-beta")
        assert not result.endswith("-beta")

    def test_flavor_flash(self):
        assert EvaluatorLLM._model_flavor("deepseek-ai/DeepSeek-V4-Flash") == "flash"

    def test_flavor_pro(self):
        assert EvaluatorLLM._model_flavor("deepseek-v4-pro") == "pro"

    def test_flavor_ultra(self):
        assert EvaluatorLLM._model_flavor("gpt-4-ultra") == "ultra"

    def test_flavor_unknown(self):
        assert EvaluatorLLM._model_flavor("some-unknown-model") == "unknown"


class TestSalvageTruncatedJson:
    def test_recovers_numeric_fields_before_truncation(self):
        truncated = (
            '{"scores": [{"response_id": "R1", "correctness": 4, '
            '"precision": null, "scientific_depth": 3, "justification": "long text'
        )
        entries = EvaluatorLLM._salvage_scores(truncated)
        assert len(entries) == 1
        assert entries[0]["correctness"] == 4
        assert entries[0]["precision"] is None
        assert entries[0]["scientific_depth"] == 3

    def test_ignores_entry_without_any_score(self):
        assert EvaluatorLLM._salvage_scores('{"response_id": "R1", "justification": "x') == []
