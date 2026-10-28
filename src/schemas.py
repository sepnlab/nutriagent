"""YAML 配置的 Pydantic schema 校验。

加载 YAML 后校验结构，字段拼写错误直接报错（不再静默回退）。
"""
import re
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, field_validator


# ── Provider-Model 格式校验 ────────────────────────────────

_PROVIDER_MODEL_PATTERNS = {
    # DashScope IDs are usually lower-case, but some catalog entries are
    # capitalised (e.g. MiniMax-M2.5) or namespaced (MiniMax/MiniMax-M2.5).
    "aliyun": r"^[A-Za-z0-9][A-Za-z0-9_.\-/]*$",
    "siliconflow": r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$",  # deepseek-ai/DeepSeek-V4-Flash
    "deepseek": r"^deepseek-v?[a-z0-9.-]+$",       # deepseek-v4-flash
    "volcano": r"^[a-z0-9][a-z0-9_.-]+$",          # doubao-seed-2-0-lite-260215
    "tencent": r"^[a-zA-Z][a-zA-Z0-9_.-]+$",       # hy-mt2-pro, hunyuan-2.0-instruct-20251111
    "xiaomi": r"^mimo-[a-z0-9.-]+$",                # mimo-v2.5-pro
    "qianwen": r"^qwen[0-9][a-z0-9._-]*$",          # qwen3.8-flash
}

_PROVIDER_EXAMPLES = {
    "aliyun": "qwen3.6-flash",
    "siliconflow": "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek": "deepseek-v4-flash",
    "volcano": "doubao-seed-2-0-lite-260215",
    "tencent": "hy-mt2-pro",
    "xiaomi": "mimo-v2.5-pro",
    "qianwen": "qwen3.8-flash",
}


def _validate_model_by_provider(provider: str, model: str) -> str:
    """Validate model ID format matches the provider's expected pattern."""
    pattern = _PROVIDER_MODEL_PATTERNS.get(provider.lower())
    if pattern and not re.match(pattern, model):
        example = _PROVIDER_EXAMPLES.get(provider.lower(), "")
        raise ValueError(
            f"Model '{model}' does not match the expected format for provider "
            f"'{provider}'. The model field must be the API model ID (e.g. "
            f"'{example}'), not the display name. "
            f"See config/models.yaml provider comments for each provider's format."
        )
    return model


# ── models.yaml ───────────────────────────────────────────

class ProviderDef(BaseModel):
    """供应商连接信息定义"""
    provider: str
    api_key_env: str = "SILICONFLOW_API_KEY"
    base_url: str = "https://api.siliconflow.cn/v1"


class RoleModelConfig(BaseModel):
    """角色模型配置（nutrichat / evaluator），通过 provider 引用连接信息"""
    provider: str = "siliconflow"
    model: str
    temperature: float = 0.0
    max_tokens: int = 200
    extra_body: Optional[dict] = None

    @field_validator("model")
    @classmethod
    def validate_model_format(cls, v: str, info) -> str:
        provider = info.data.get("provider", "siliconflow")
        return _validate_model_by_provider(provider, v)


class SelfCheckRoleConfig(BaseModel):
    """SelfCheck 模型配置，不指定 provider/model 时继承 nutrichat"""
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 200
    n_samples: int = 5


class BaselineDefaults(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 200


class BaselineEntry(BaseModel):
    provider: str = "siliconflow"
    name: str
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_body: Optional[dict] = None

    @field_validator("model")
    @classmethod
    def validate_model_format(cls, v: str, info) -> str:
        provider = info.data.get("provider", "siliconflow")
        return _validate_model_by_provider(provider, v)


class BaselinesBlock(BaseModel):
    _defaults: BaselineDefaults
    models: List[BaselineEntry]

    @field_validator("models", mode="before")
    @classmethod
    def normalize_models(cls, v):
        """Accept flat list or dict keyed by provider -> normalize to list."""
        if isinstance(v, dict):
            result = []
            for provider, entries in v.items():
                for entry in entries:
                    if isinstance(entry, dict):
                        entry.setdefault("provider", provider)
                    result.append(entry)
            return result
        return v


class ModelsFile(BaseModel):
    providers: List[ProviderDef]
    nutrichat: RoleModelConfig
    evaluator: RoleModelConfig
    selfcheck: SelfCheckRoleConfig
    baselines: BaselinesBlock


# ── settings.yaml ─────────────────────────────────────────

class RetrievalSettings(BaseModel):
    k: int = 1


class RetrySettings(BaseModel):
    max_attempts: int = 3
    min_wait: int = 2
    max_wait: int = 30


class ApiSettings(BaseModel):
    timeout: int = 30
    max_concurrent: int = 6
    retry: RetrySettings = RetrySettings()
    baseline_timeout: Optional[int] = None


class RouterSettings(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 200


class SynthesisSettings(BaseModel):
    temperature: float = 0.0


class AgentSettings(BaseModel):
    router: RouterSettings = RouterSettings()
    synthesis: SynthesisSettings = SynthesisSettings()
    max_iterations: int = 10
    early_stopping: bool = True


class BaselineSettings(BaseModel):
    thinking_max_tokens: int = 4096


class SettingsFile(BaseModel):
    species: str = "broiler"
    max_tokens: int = 200
    retrieval: RetrievalSettings = RetrievalSettings()
    api: ApiSettings = ApiSettings()
    agent: AgentSettings = AgentSettings()
    baselines: BaselineSettings = BaselineSettings()

    @field_validator("species")
    @classmethod
    def validate_species(cls, v: str) -> str:
        if v.lower() not in ("broiler", "layer"):
            raise ValueError(f"species must be 'broiler' or 'layer', got '{v}'")
        return v.lower()


# ── embeddings.yaml ───────────────────────────────────────

class NliConfig(BaseModel):
    model_path: str = "./models/nli-deberta-v3-base"
    device: Optional[str] = None  # None = inherit from embedding device


class LocalConfig(BaseModel):
    model_path: str = "./Qwen3-Embedding-0.6B"
    device: str = "cuda"
    nli: NliConfig = NliConfig()


class SiliconFlowConfig(BaseModel):
    default_model: str = "Qwen/Qwen3-Embedding-8B"


class EmbeddingsFile(BaseModel):
    provider: str = "local"
    local: LocalConfig = LocalConfig()
    siliconflow: SiliconFlowConfig = SiliconFlowConfig()

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v.lower() not in ("local", "siliconflow"):
            raise ValueError(f"provider must be 'local' or 'siliconflow', got '{v}'")
        return v.lower()


# ── pricing.yaml ──────────────────────────────────────────

class PricingEntry(BaseModel):
    input: float
    output: float


class PricingFile(BaseModel):
    models: dict[str, PricingEntry]


# ── ablation.yaml ──────────────────────────────────────────

class DimMeta(BaseModel):
    title: str = ""
    subtitle: str = ""
    note: str = ""


class ArchVariant(BaseModel):
    name: str
    description: str = ""
    skip_selfcheck: bool = False


class SpecialistVariant(BaseModel):
    name: str
    exclude: Optional[str] = None
    description: str = ""
    skip_selfcheck: bool = False


class RouterVariant(BaseModel):
    name: str
    strategy: str = "llm"
    description: str = ""
    skip_selfcheck: bool = False


class RagVariant(BaseModel):
    name: str
    rag_enabled: bool = True
    description: str = ""
    skip_selfcheck: bool = False


class ArchDimension(BaseModel):
    enabled: bool = True
    meta: DimMeta = DimMeta()
    variants: List[ArchVariant] = []


class SpecialistDimension(BaseModel):
    enabled: bool = True
    meta: DimMeta = DimMeta()
    variants: List[SpecialistVariant] = []


class RouterDimension(BaseModel):
    enabled: bool = True
    meta: DimMeta = DimMeta()
    variants: List[RouterVariant] = []


class RagDimension(BaseModel):
    enabled: bool = True
    meta: DimMeta = DimMeta()
    variants: List[RagVariant] = []


class SynthesisVariant(BaseModel):
    name: str
    synthesis_enabled: bool = True
    description: str = ""
    skip_selfcheck: bool = False


class SynthesisDimension(BaseModel):
    enabled: bool = True
    meta: DimMeta = DimMeta()
    variants: List[SynthesisVariant] = []


class LLMBackboneVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    model: str = "deepseek-ai/DeepSeek-V4-Flash"
    model_ref: Optional[str] = None  # models.yaml baselines 的 name（推荐）
    description: str = ""
    skip_selfcheck: bool = False


class LLMBackboneDimension(BaseModel):
    enabled: bool = True
    meta: DimMeta = DimMeta()
    variants: List[LLMBackboneVariant] = []


class LanguageVariant(BaseModel):
    name: str
    query_set: str = "all"
    description: str = ""
    skip_selfcheck: bool = False


class LanguageDimension(BaseModel):
    enabled: bool = True
    meta: DimMeta = DimMeta()
    variants: List[LanguageVariant] = []


class FactorialVariant(BaseModel):
    """A variant of a factorial contrast dimension (E1/E2)."""

    name: str
    description: str = ""
    skip_selfcheck: bool = False


class FactorialDimension(BaseModel):
    enabled: bool = True
    meta: DimMeta = DimMeta()
    variants: List[FactorialVariant] = []


class AblationScoring(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluator_llm: bool = True
    selfcheck: bool = False  # 消融默认跳过（太慢）
    hallucination_ratio: float = 1.0
    n_selfcheck_values: Optional[List[int]] = None
    flesch: bool = True
    tolerance: bool = True
    # --quick 覆盖参数（config.local.yaml 可自定义），不参与 AblationScoring 校验
    quick: Optional[dict] = None

    @field_validator("hallucination_ratio")
    @classmethod
    def validate_hallucination_ratio(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"hallucination_ratio must be between 0 and 1, got {v}")
        return v


class AblationFile(BaseModel):
    scoring: AblationScoring = AblationScoring()
    architecture: ArchDimension = ArchDimension()
    specialist: SpecialistDimension = SpecialistDimension()
    router: RouterDimension = RouterDimension()
    rag: RagDimension = RagDimension()
    synthesis: SynthesisDimension = SynthesisDimension()
    llm_backbone: LLMBackboneDimension = LLMBackboneDimension()
    language: LanguageDimension = LanguageDimension()
    architecture_factorial: FactorialDimension = FactorialDimension()
    synthesis_factorial: FactorialDimension = FactorialDimension()


# ── evaluate.yaml ─────────────────────────────────────────

class EvaluationSettings(BaseModel):
    """评估参数（全局默认，Config 在模块加载时读取）"""
    n_selfcheck_values: List[int] = [5, 10, 20]
    n_selfcheck_gen_samples: Optional[int] = None
    sample_ratio: Optional[float] = None
    hallucination_ratio: float = 1.0

    @field_validator("sample_ratio")
    @classmethod
    def validate_sample_ratio(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError(f"sample_ratio must be >= 0, got {v}")
        return v

    @field_validator("hallucination_ratio")
    @classmethod
    def validate_hallucination_ratio(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"hallucination_ratio must be between 0 and 1, got {v}")
        return v


class ScoringSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluator_llm: bool = True
    selfcheck: bool = True
    n_selfcheck_values: Optional[List[int]] = None
    tolerance: bool = True
    flesch: bool = True
    shuffle_responses: bool = False
    hallucination_ratio: float = 1.0
    # --quick 覆盖参数（config.local.yaml 可自定义），不参与 ScoringSettings 校验
    quick: Optional[dict] = None

    @field_validator("hallucination_ratio")
    @classmethod
    def validate_hallucination_ratio(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"hallucination_ratio must be between 0 and 1, got {v}")
        return v


class EvaluateFile(BaseModel):
    evaluation: EvaluationSettings = EvaluationSettings()
    agent_mode: str = "single"
    query_set: str = "all"
    baselines: Optional[List[str]] = None
    breakdown: bool = False
    scoring: ScoringSettings = ScoringSettings()
    categories: Optional[List[str]] = None
    thinking_baselines: Optional[List[str]] = None

    @field_validator("agent_mode")
    @classmethod
    def validate_agent_mode(cls, v: str) -> str:
        if v not in ("single", "multi", "both"):
            raise ValueError(f"agent_mode must be single/multi/both, got '{v}'")
        return v

    @field_validator("query_set")
    @classmethod
    def validate_query_set(cls, v: str) -> str:
        if v not in ("all", "chinese", "western"):
            raise ValueError(f"query_set must be all/chinese/western, got '{v}'")
        return v


# ── 校验入口 ─────────────────────────────────────────────

def validate_config(name: str, data: dict) -> dict:
    """Validate a raw YAML dict against its schema.

    Args:
        name: Config file identifier ('baselines', 'settings', 'embeddings', 'pricing')
        data: Raw dict loaded from YAML

    Returns:
        Validated dict (same data, with defaults filled in)

    Raises:
        ValueError: On invalid fields or types with a descriptive message
    """
    schema_map = {
        "models": ModelsFile,
        "settings": SettingsFile,
        "embeddings": EmbeddingsFile,
        "pricing": PricingFile,
        "ablation": AblationFile,
        "evaluate": EvaluateFile,
    }

    model_cls = schema_map.get(name)
    if model_cls is None:
        return data

    validated = model_cls(**data)
    # models.yaml: 保留原始字段的显式 null 以区分 unset（用于 thinking 策略判断）
    if name == "models":
        return validated.model_dump(exclude_unset=True)
    return validated.model_dump()
