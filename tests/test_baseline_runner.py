"""基线模型 Runner 测试（extra_body 参数透传）。"""

from unittest.mock import MagicMock, patch

from src.config import Config
from src.evaluation.baseline_runner import create_baseline_client
from src.schemas import validate_config


def _minimal_models_yaml(extra_body=None):
    """构建包含可选 extra_body 的最小 models.yaml 结构。"""
    baseline = {
        "provider": "test_p",
        "name": "test-model",
        "model": "test-model-v1",
    }
    if extra_body is not None:
        baseline["extra_body"] = extra_body

    return {
        "providers": [
            {"provider": "test_p", "api_key_env": "TEST_KEY", "base_url": "https://test.com/v1"},
        ],
        "nutrichat": {"provider": "test_p", "model": "test-nutrichat"},
        "evaluator": {"provider": "test_p", "model": "test-eval"},
        "selfcheck": {"temperature": 1.0, "n_samples": 3},
        "baselines": {
            "_defaults": {},
            "models": [baseline],
        },
    }


class TestExtraBodySchema:
    """验证 extra_body 字段能通过 Pydantic schema 校验。"""

    def test_extra_body_accepted_by_schema(self):
        data = _minimal_models_yaml(extra_body={"enable_thinking": False})
        result = validate_config("models", data)
        entry = result["baselines"]["models"][0]
        assert entry["extra_body"] == {"enable_thinking": False}

    def test_extra_body_optional(self):
        data = _minimal_models_yaml(extra_body=None)
        result = validate_config("models", data)
        entry = result["baselines"]["models"][0]
        assert "extra_body" not in entry or entry["extra_body"] is None

    def test_extra_body_multiple_keys(self):
        data = _minimal_models_yaml(extra_body={"key1": "v1", "key2": 42})
        result = validate_config("models", data)
        entry = result["baselines"]["models"][0]
        assert entry["extra_body"] == {"key1": "v1", "key2": 42}

    def test_extra_body_empty_dict(self):
        data = _minimal_models_yaml(extra_body={})
        result = validate_config("models", data)
        entry = result["baselines"]["models"][0]
        assert entry["extra_body"] == {}


class TestExtraBodyConfig:
    """验证 extra_body 从 YAML 加载到 Config.BASELINE_MODELS 的完整链路。"""

    def test_aliyun_qwen3_config_has_extra_body(self):
        """集成测试：检查 models.yaml 中的 aliyun/Qwen3-32B 配置。"""
        for b in Config.BASELINE_MODELS:
            if b["name"] == "aliyun/Qwen3-32B":
                eb = b.get("extra_body")
                # extra_body 可能是 {enable_thinking: False}（默认）
                # 或 None（当该模型在 thinking_baselines 白名单中时）
                assert eb is None or eb == {"enable_thinking": False},                     f"Unexpected extra_body for aliyun/Qwen3-32B: {eb}"
                return
        # 如果没找到该模型，说明 evaluate.yaml 过滤或 key 未配置
        # 此时跳过而非报错
        import pytest
        pytest.skip("aliyun/Qwen3-32B not in active baselines (check API keys)")


class TestExtraBodyPassthrough:
    """验证 create_baseline_client 将 extra_body 传入 API 调用。"""

    def test_extra_body_passed_to_api(self):
        baseline = {
            "provider": "siliconflow",
            "name": "test",
            "model": "test-model",
            "api_key": "test-key",
            "base_url": "https://test.com/v1",
            "extra_body": {"enable_thinking": False},
        }
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )
            invoke = create_baseline_client(baseline)
            result = invoke([{"role": "user", "content": "hello"}])

        assert result["response"] == "ok"
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs.get("extra_body") == {"enable_thinking": False}

    def test_extra_body_omitted_when_not_set(self):
        baseline = {
            "provider": "siliconflow",
            "name": "test",
            "model": "test-model",
            "api_key": "test-key",
            "base_url": "https://test.com/v1",
        }
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            )
            invoke = create_baseline_client(baseline)
            result = invoke([{"role": "user", "content": "hello"}])

        assert result["response"] == "ok"
        _, kwargs = mock_client.chat.completions.create.call_args
        assert "extra_body" not in kwargs
