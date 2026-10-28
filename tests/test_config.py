"""配置加载与 schema 校验测试。"""

import yaml
from src.schemas import validate_config
from src.config import Config


class TestSchemaValidation:
    def test_settings_valid(self):
        result = validate_config("settings", {"species": "broiler"})
        assert result["species"] == "broiler"
        assert result["max_tokens"] == 200  # default

    def test_settings_invalid_species(self):
        import pytest
        with pytest.raises(Exception, match="species must be"):
            validate_config("settings", {"species": "duck"})

    def test_settings_invalid_type(self):
        import pytest
        with pytest.raises(Exception):
            validate_config("settings", {"max_tokens": "not_a_number"})

    def test_embeddings_valid(self):
        result = validate_config("embeddings", {"provider": "local"})
        assert result["provider"] == "local"

    def test_embeddings_invalid_provider(self):
        import pytest
        with pytest.raises(Exception, match="provider must be"):
            validate_config("embeddings", {"provider": "aws"})

    def test_models_valid(self):
        data = {
            "providers": [
                {"provider": "test_p", "api_key_env": "TEST_KEY", "base_url": "https://test.com/v1"},
            ],
            "nutrichat": {"provider": "test_p", "model": "test-model"},
            "evaluator": {"provider": "test_p", "model": "test-eval"},
            "selfcheck": {"temperature": 1.0, "n_samples": 3},
            "baselines": {
                "_defaults": {},
                "models": [{"provider": "test_p", "name": "m1", "model": "model1"}],
            },
        }
        result = validate_config("models", data)
        assert result["baselines"]["models"][0]["name"] == "m1"

    def test_pricing_valid(self):
        data = {"models": {"test/model": {"input": 1.0, "output": 2.0}}}
        result = validate_config("pricing", data)
        assert "test/model" in result["models"]


class TestProviderModelValidation:
    """Provider-model 格式校验。"""

    def _validate(self, provider, model):
        from src.schemas import _validate_model_by_provider
        return _validate_model_by_provider(provider, model)

    def test_aliyun_pass(self):
        assert self._validate("aliyun", "qwen3.6-flash")
        assert self._validate("aliyun", "glm-5")
        # DashScope also serves capitalised / namespaced catalog IDs.
        assert self._validate("aliyun", "MiniMax-M2.5")
        assert self._validate("aliyun", "MiniMax/MiniMax-M2.5")

    def test_aliyun_fail_space(self):
        import pytest
        with pytest.raises(ValueError, match="does not match"):
            self._validate("aliyun", "MiniMax M2.5")

    def test_aliyun_fail_invalid_char(self):
        import pytest
        with pytest.raises(ValueError):
            self._validate("aliyun", "qwen3.6:flash")

    def test_siliconflow_pass(self):
        assert self._validate("siliconflow", "deepseek-ai/DeepSeek-V4-Flash")
        assert self._validate("siliconflow", "Qwen/Qwen3.5-27B")

    def test_siliconflow_fail_no_slash(self):
        import pytest
        with pytest.raises(ValueError):
            self._validate("siliconflow", "deepseek-v4-flash")

    def test_deepseek_pass(self):
        assert self._validate("deepseek", "deepseek-v4-flash")
        assert self._validate("deepseek", "deepseek-v4-pro")

    def test_deepseek_fail_wrong_prefix(self):
        import pytest
        with pytest.raises(ValueError):
            self._validate("deepseek", "qwen3.6-flash")

    def test_unknown_provider_skipped(self):
        """未知 provider 不校验（兼容自定义 API）"""
        assert self._validate("custom", "any-model-name")


class TestConfigData:
    """验证当前 YAML 文件能够通过加载和校验。"""

    def test_config_has_expected_attrs(self):
        assert Config.SPECIES in ("broiler", "layer")
        assert Config.LLM_MODEL
        assert Config.PRICING_TABLE
        assert len(Config.BASELINE_MODELS) >= 1

    def test_pricing_has_models(self):
        assert len(Config.PRICING_TABLE) >= 3
        assert "siliconflow::deepseek-ai/DeepSeek-V4-Flash" in Config.PRICING_TABLE
        entry = Config.PRICING_TABLE["siliconflow::deepseek-ai/DeepSeek-V4-Flash"]
        assert entry["input"] > 0
        assert entry["output"] > 0
