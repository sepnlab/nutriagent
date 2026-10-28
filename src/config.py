"""Central configuration for NutriCHAT system.

Reads from YAML config files and .env for API keys only.
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_local_override() -> dict:
    """Load config.local.yaml if it exists (used for local experiment overrides)."""
    path = os.path.join(_PROJECT_ROOT, "config", "config.local.yaml")
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data is None:
                return {}
            return data
    except FileNotFoundError:
        return {}
    except ImportError:
        return {}
    except yaml.YAMLError as e:
        import sys
        print(f"[WARNING] config.local.yaml 解析失败，所有本地配置将被忽略！", file=sys.stderr)
        print(f"  YAML 错误: {e}", file=sys.stderr)
        print(f"  路径: {path}", file=sys.stderr)
        return {}


def _load_yaml(*path_parts: str, schema: str = None) -> dict:
    """Load a YAML file from config/ directory, optionally validated against a schema."""
    path = os.path.join(_PROJECT_ROOT, "config", *path_parts)
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if schema:
            from src.schemas import validate_config
            data = validate_config(schema, data)
        return data
    except FileNotFoundError:
        return {}
    except (ImportError, yaml.YAMLError) as e:
        import sys
        print(f"[WARNING] Failed to load config/{os.path.join(*path_parts)}: {e}", file=sys.stderr)
        return {}


# ── Settings ──────────────────────────────────────────────
_settings = _load_yaml("settings.yaml", schema="settings")

# Local experiment overrides — merged into all configs
_LOCAL_OVERRIDE = _load_local_override()
if _LOCAL_OVERRIDE:
    for k, v in _LOCAL_OVERRIDE.items():
        if k not in _settings:
            _settings[k] = v  # new key
        elif isinstance(_settings[k], type(v)):
            # 类型匹配 → 覆盖
            if isinstance(v, dict):
                _settings[k] = deep_merge(_settings[k], v)
            else:
                _settings[k] = v
        else:
            # 类型不匹配 → 跳过并警告
            # 特例：baselines 在 settings.yaml 是 dict（thinking_max_tokens），
            # 在 config.local.yaml 是 list（--compare 过滤），两者独立消费，非真正冲突
            if k != "baselines":
                import sys
                print(f"[WARNING] config.local.yaml: key '{k}' type mismatch "
                      f"(expected {type(_settings[k]).__name__}, got {type(v).__name__}), skipping",
                      file=sys.stderr)

_SPECIES = (os.getenv("SPECIES") or _settings.get("species", "broiler")).strip().lower()
_MAX_TOKENS = int(_settings.get("max_tokens", 200))
_SPECIALIST_MAX_TOKENS = int(_settings.get("specialist_max_tokens", 200))
_SELFCHECK_MAX_TOKENS = int(_settings.get("selfcheck_max_tokens", 200))
_BASELINE_THINKING_MAX_TOKENS: int = int(
    _settings["baselines"].get("thinking_max_tokens", 4096)
) if isinstance(_settings.get("baselines"), dict) else 4096

# Evaluation settings — from config/evaluate.yaml (evaluation: section), overridden by config.local.yaml
_eval_file = _load_yaml("evaluate.yaml")
if _LOCAL_OVERRIDE:
    _eval_file = deep_merge(_eval_file, _LOCAL_OVERRIDE)
    # scoring_evaluate 优先级高于 scoring（实验专用覆盖）
    if "scoring_evaluate" in _LOCAL_OVERRIDE:
        _eval_file["scoring"] = deep_merge(
            _eval_file.get("scoring", {}), _LOCAL_OVERRIDE["scoring_evaluate"]
        )
    # 校验 merged scoring：拒掉 config.local.yaml 中的拼写错误
    if "scoring" in _eval_file:
        from src.schemas import ScoringSettings
        ScoringSettings(**_eval_file["scoring"])
_eval_cfg = _eval_file.get("evaluation", {}) if _eval_file else {}
_N_SELFCHECK_VALUES: list = _eval_cfg.get("n_selfcheck_values", [5, 10, 20])
_N_SELFCHECK_GEN: int = _eval_cfg.get("n_selfcheck_gen_samples") or max(_N_SELFCHECK_VALUES)
_N_SAMPLE_RATIO: Optional[float] = _eval_cfg.get("sample_ratio")
# Priority: scoring.hallucination_ratio（local 覆盖）> evaluation.hallucination_ratio > settings.yaml > 1.0
_h_scoring = _eval_file.get("scoring", {}).get("hallucination_ratio") if _eval_file else None
_HALLUCINATION_RATIO: float = float(
    _h_scoring if _h_scoring is not None
    else _eval_cfg.get("hallucination_ratio") if _eval_cfg.get("hallucination_ratio") is not None
    else _settings.get("evaluation", {}).get("hallucination_ratio", 1.0)
)

# Retrieval settings
_ret_cfg = _settings.get("retrieval", {})
_RAG_K: int = int(_ret_cfg.get("k", 1))

# API settings
_api_cfg = _settings.get("api", {})
_API_TIMEOUT: int = int(_api_cfg.get("timeout", 30))
_API_MAX_CONCURRENT: int = int(_api_cfg.get("max_concurrent", 6))
_retry_cfg = _api_cfg.get("retry", {})
_API_RETRY_MAX: int = int(_retry_cfg.get("max_attempts", 3))
_API_RETRY_MIN_WAIT: int = int(_retry_cfg.get("min_wait", 2))
_API_RETRY_MAX_WAIT: int = int(_retry_cfg.get("max_wait", 30))
_API_BASELINE_TIMEOUT: Optional[int] = _api_cfg.get("baseline_timeout")

# Agent settings
_agent_cfg = _settings.get("agent", {})
_router_cfg = _agent_cfg.get("router", {})
_ROUTER_TEMPERATURE: float = float(_router_cfg.get("temperature", 0.0))
_ROUTER_MAX_TOKENS: int = int(_router_cfg.get("max_tokens", 200))
_synth_cfg = _agent_cfg.get("synthesis", {})
_SYNTHESIS_TEMPERATURE: float = float(_synth_cfg.get("temperature", 0.0))
_MAX_ITERATIONS: int = int(_agent_cfg.get("max_iterations", 10))
_EARLY_STOPPING: bool = bool(_agent_cfg.get("early_stopping", True))

_CORPUS_DIR = os.path.join(_PROJECT_ROOT, "corpus")
_SHARED_DATA = os.path.join(_CORPUS_DIR, "shared")
_SPECIES_DATA = os.path.join(_CORPUS_DIR, _SPECIES)
_QUERIES_DIR = os.path.join(_PROJECT_ROOT, "queries")
_SPECIES_QUERIES = os.path.join(_QUERIES_DIR, _SPECIES)
_VDB_DIR = os.path.join(_PROJECT_ROOT, "vector_db")
_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".cache")

_MODEL_KEYS = {"nutrichat", "evaluator", "selfcheck"}

# ── Models (models.yaml) ──────────────────────────────────
_model_cfg = _load_yaml("models.yaml", schema="models")
if _LOCAL_OVERRIDE:
    # Only merge model-relevant keys into _model_cfg to avoid
    # clashes (e.g. evaluate.yaml's "baselines: list" vs models.yaml's "baselines: dict")
    _model_cfg = deep_merge(_model_cfg, {
        k: v for k, v in _LOCAL_OVERRIDE.items() if k in _MODEL_KEYS
    })
    # Re-validate after local overrides to catch provider-model format mismatches
    from src.schemas import ModelsFile
    _model_cfg = ModelsFile(**_model_cfg).model_dump(exclude_unset=True)

_PROVIDERS_LIST: list = _model_cfg.get("providers", [])


def _get_provider_config(provider_name: str) -> dict:
    """Look up a provider's connection details by name."""
    for p in _PROVIDERS_LIST:
        if p.get("provider") == provider_name:
            return p
    return {}


def _thinking_disabled_extra_body(provider: str) -> Optional[dict]:
    """Return provider-appropriate extra_body to disable deep thinking.

    Returns None for providers that don't support a thinking toggle
    (e.g. openrouter, openai, anthropic — passthrough or no toggle).
    """
    p = provider.lower()
    if p in ("deepseek", "volcano", "siliconflow"):
        return {"thinking": {"type": "disabled"}}
    if p in ("aliyun", "tencent", "qianwen"):
        return {"enable_thinking": False}
    # openrouter / openai / anthropic: no provider-level thinking toggle
    return None


def _extract_model_config(section: str, fallback: dict) -> dict:
    """Extract a role model config from models.yaml, resolve provider connection info.

    If no explicit extra_body is set in YAML, auto-injects a provider-appropriate
    thinking-disabled extra_body so users don't need to remember the format per provider.
    Set ``extra_body: null`` in YAML to keep thinking enabled without a provider default.
    """
    cfg = _model_cfg.get(section, {})
    if cfg and cfg.get("model"):
        provider_name = cfg.get("provider", "")
        provider_cfg = _get_provider_config(provider_name) if provider_name else {}
        env_var = provider_cfg.get("api_key_env", "SILICONFLOW_API_KEY")
        result = {
            "model": cfg["model"],
            "base_url": provider_cfg.get("base_url") or fallback.get("base_url", "https://api.siliconflow.cn/v1"),
            "api_key": os.getenv(env_var) or fallback.get("api_key", ""),
            "temperature": float(cfg.get("temperature", fallback.get("temperature", 0.0))),
            "max_tokens": int(cfg.get("max_tokens", fallback.get("max_tokens", 200))),
        }
        extra_body = cfg.get("extra_body")
        if extra_body is not None:
            result["extra_body"] = extra_body
        elif provider_name and "extra_body" not in cfg:
            # Auto-inject only when extra_body is truly absent (not null, not set)
            auto_eb = _thinking_disabled_extra_body(provider_name)
            if auto_eb:
                result["extra_body"] = auto_eb

        # Normalize to provider format — handles deep_merge residue when user
        # switches provider via config.local.yaml but old extra_body keys linger.
        if provider_name and "extra_body" in result:
            canon = _thinking_disabled_extra_body(provider_name)
            if canon and not canon.items() <= result["extra_body"].items():
                # Current extra_body is missing expected provider keys → replace
                result["extra_body"] = canon

        return result
    return dict(fallback)


_NUTRI_CHAT_MODEL = _extract_model_config("nutrichat", {
    "model": os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
    "base_url": os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1"),
    "api_key": os.getenv("SILICONFLOW_API_KEY", ""),
    "temperature": float(os.getenv("TEMPERATURE", "0.0")),
    "max_tokens": _MAX_TOKENS,
})

_EVALUATOR_MODEL = _extract_model_config("evaluator", {
    "model": _NUTRI_CHAT_MODEL["model"],
    "base_url": _NUTRI_CHAT_MODEL["base_url"],
    "api_key": _NUTRI_CHAT_MODEL["api_key"],
    "max_tokens": 2000,
})

_selfcheck_cfg = _model_cfg.get("selfcheck", {})
if _selfcheck_cfg.get("provider") or _selfcheck_cfg.get("model"):
    sc_provider_name = _selfcheck_cfg.get("provider") or "siliconflow"
    sc_provider_cfg = _get_provider_config(sc_provider_name) or {}
    env_var = sc_provider_cfg.get("api_key_env", "SILICONFLOW_API_KEY")
    _SELFCHECK_CONFIG = {
        "model": _selfcheck_cfg.get("model") or _NUTRI_CHAT_MODEL["model"],
        "base_url": sc_provider_cfg.get("base_url", _NUTRI_CHAT_MODEL["base_url"]),
        "api_key": os.getenv(env_var, _NUTRI_CHAT_MODEL["api_key"]),
        "n_samples": int(_selfcheck_cfg.get("n_samples", 5)),
        "extra_body": _selfcheck_cfg.get("extra_body") or _NUTRI_CHAT_MODEL.get("extra_body"),
    }
else:
    _SELFCHECK_CONFIG = {"model": _NUTRI_CHAT_MODEL["model"], "base_url": _NUTRI_CHAT_MODEL["base_url"],
                         "api_key": _NUTRI_CHAT_MODEL["api_key"], "n_samples": 5,
                         "extra_body": _NUTRI_CHAT_MODEL.get("extra_body")}

# Provider-level API compat: auto-inject extra_body for providers that
# always_disable_thinking (e.g. SiliconFlow returns empty at T>0 without it)
if not _SELFCHECK_CONFIG.get("extra_body"):
    sc_provider = _selfcheck_cfg.get("provider") or _model_cfg.get("nutrichat", {}).get("provider", "")
    sc_provider_cfg = _get_provider_config(sc_provider)
    if sc_provider_cfg.get("always_disable_thinking"):
        _SELFCHECK_CONFIG["extra_body"] = _thinking_disabled_extra_body(sc_provider)

# ── Embedding (embeddings.yaml) ───────────────────────────
_emb_cfg = _load_yaml("embeddings.yaml", schema="embeddings")
_EMBEDDING_PROVIDER = _emb_cfg.get("provider", "local").lower()

_local_emb = _emb_cfg.get("local", {})
_LOCAL_EMBEDDING_PATH = _local_emb.get("model_path", os.path.join(_PROJECT_ROOT, "models", "Qwen3-Embedding-0.6B"))
# Resolve relative path against project root
if _LOCAL_EMBEDDING_PATH and not os.path.isabs(_LOCAL_EMBEDDING_PATH):
    _LOCAL_EMBEDDING_PATH = os.path.normpath(os.path.join(_PROJECT_ROOT, _LOCAL_EMBEDDING_PATH))
_LOCAL_EMBEDDING_DEVICE = _local_emb.get("device", "cuda")

# NLI model config (under local.nli, falls back to cross-encoder/nli-deberta-v3-base)
_nli_cfg = _local_emb.get("nli", {})
_NLI_MODEL_PATH = _nli_cfg.get("model_path", os.path.join(_PROJECT_ROOT, "models", "nli-deberta-v3-base"))
if _NLI_MODEL_PATH and not os.path.isabs(_NLI_MODEL_PATH):
    _NLI_MODEL_PATH = os.path.normpath(os.path.join(_PROJECT_ROOT, _NLI_MODEL_PATH))
_NLI_DEVICE = _nli_cfg.get("device") or _LOCAL_EMBEDDING_DEVICE

_sf_emb = _emb_cfg.get("siliconflow", {})
_SILICONFLOW_EMBEDDING_MODEL = _sf_emb.get("default_model", "Qwen/Qwen3-Embedding-8B")

# ── Pricing (pricing.yaml) ──────────────────────────
_PRICING_TABLE: dict = _load_yaml("pricing.yaml", schema="pricing").get("models", {})

# ── Baselines — merge provider connection info ────────
# Models whose thinking cannot be disabled (dedicated reasoning / thinking-only models).
# These are always excluded from thinking-disable auto-injection, regardless of provider.
_THINKING_INHERENT = {
    "volcano/DeepSeek-R1",      # reasoning model, always thinks
    "tencent/Hy2-Think",        # thinking-specific variant, always outputs <think>
}

def _resolve_baseline_models() -> list:
    """Resolve baseline model configs: merge provider connection info + per-model overrides."""
    bl_cfg = _model_cfg.get("baselines", {})
    defaults = {**bl_cfg.get("_defaults", {})}
    # 基线 max_tokens 默认继承 settings.yaml，确保 NutriAgent 与基线一致
    defaults["max_tokens"] = _MAX_TOKENS
    models_list = bl_cfg.get("models", [])
    _thinking_set = set(_LOCAL_OVERRIDE.get("thinking_baselines") or []) | _THINKING_INHERENT
    resolved = []
    for m in models_list:
        provider_name = m.get("provider") or "siliconflow"
        pc = _get_provider_config(provider_name) or {}
        entry = {
            **defaults,
            "provider": provider_name,
            "api_key_env": pc.get("api_key_env", "SILICONFLOW_API_KEY"),
            "base_url": pc.get("base_url", "https://api.siliconflow.cn/v1"),
            "name": m["name"],
            "model": m["model"],
        }
        if m.get("temperature") is not None:
            entry["temperature"] = m["temperature"]
        if m.get("max_tokens") is not None:
            entry["max_tokens"] = m["max_tokens"]
        if m["name"] in _thinking_set:
            # Keep thinking enabled: don't set extra_body.
            # Thinking models need more output room: use configurable default.
            if entry.get("max_tokens", 200) == 200 and m.get("max_tokens") is None:
                entry["max_tokens"] = _BASELINE_THINKING_MAX_TOKENS
        elif m.get("extra_body") is not None:
            entry["extra_body"] = m["extra_body"]
        else:
            auto_eb = _thinking_disabled_extra_body(provider_name)
            if auto_eb:
                entry["extra_body"] = auto_eb
        entry["api_key"] = os.getenv(entry["api_key_env"]) or ""
        resolved.append(entry)
    return resolved

_BASELINE_MODELS = _resolve_baseline_models()


class Config:
    # Species
    SPECIES: str = _SPECIES

    # LLM — from config/models.yaml (nutrichat: section)
    LLM_API_KEY: str = _NUTRI_CHAT_MODEL["api_key"]
    LLM_BASE_URL: str = _NUTRI_CHAT_MODEL["base_url"]
    LLM_MODEL: str = _NUTRI_CHAT_MODEL["model"]
    TEMPERATURE: float = _NUTRI_CHAT_MODEL["temperature"]
    MAX_TOKENS: int = _NUTRI_CHAT_MODEL["max_tokens"]
    LLM_EXTRA_BODY: Optional[dict] = _NUTRI_CHAT_MODEL.get("extra_body")

    # Embedding — from config/embeddings.yaml
    EMBEDDING_PROVIDER: str = _EMBEDDING_PROVIDER
    LOCAL_EMBEDDING_PATH: str = _LOCAL_EMBEDDING_PATH
    LOCAL_EMBEDDING_DEVICE: str = _LOCAL_EMBEDDING_DEVICE
    SILICONFLOW_EMBEDDING_MODEL: str = _SILICONFLOW_EMBEDDING_MODEL

    # NLI — from config/embeddings.yaml (local.nli section)
    NLI_MODEL_PATH: str = _NLI_MODEL_PATH
    NLI_DEVICE: str = _NLI_DEVICE

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.path.join(_VDB_DIR, f"chroma_{_SPECIES}")
    CHROMA_COLLECTION_NAME: str = f"poultry_db_{_SPECIES}"

    # Paths — corpus (CSV data), queries (test sets), vector_db (ChromaDB)
    CORPUS_DIR: str = _CORPUS_DIR
    SHARED_DATA_DIR: str = _SHARED_DATA
    SPECIES_DATA_DIR: str = _SPECIES_DATA
    QUERIES_DIR: str = _QUERIES_DIR
    VDB_DIR: str = _VDB_DIR
    CACHE_DIR: str = _CACHE_DIR
    FEED_INGREDIENT_BANK_FILE: str = os.path.join(_SHARED_DATA, "feed_ingredient_bank.csv")
    FEED_INGREDIENT_AA_FILE: str = os.path.join(_SHARED_DATA, "feed_ingredient_amino_acids.csv")
    FEED_INGREDIENT_DIGESTIBILITY_FILE: str = os.path.join(_SHARED_DATA, "feed_ingredient_digestibility.csv")
    DEFINITIONS_FILE: str = os.path.join(_SHARED_DATA, "definitions.csv")
    PERFORMANCE_MANAGEMENT_FILE: str = os.path.join(_SPECIES_DATA, "performance_management.csv")
    PERFORMANCE_PROCESSING_FILE: str = os.path.join(_SPECIES_DATA, "performance_processing.csv")
    NUTRIENT_REQUIREMENTS_FILE: str = os.path.join(_SPECIES_DATA, "nutrient_requirements.csv")
    NUTRIENT_VITAMIN_MINERAL_FILE: str = os.path.join(_SPECIES_DATA, "nutrient_vitamin_mineral.csv")
    EVALUATION_QUERIES_FILE: str = os.path.join(_SPECIES_QUERIES, "evaluation_queries_curated.json")

    # EvaluatorLLM — from config/models.yaml (evaluator: section)
    EVALUATOR_MODEL: str = _EVALUATOR_MODEL["model"]
    EVALUATOR_BASE_URL: str = _EVALUATOR_MODEL["base_url"]
    EVALUATOR_API_KEY: str = _EVALUATOR_MODEL["api_key"]
    EVALUATOR_MAX_TOKENS: int = _EVALUATOR_MODEL["max_tokens"]

    # SelfCheckGPT — from config/models.yaml (selfcheck: section)
    SELFCHECK_MODEL: str = _SELFCHECK_CONFIG["model"]
    SELFCHECK_BASE_URL: str = _SELFCHECK_CONFIG["base_url"]
    SELFCHECK_API_KEY: str = _SELFCHECK_CONFIG["api_key"]
    SELFCHECK_N_SAMPLES: int = _SELFCHECK_CONFIG["n_samples"]
    SELFCHECK_EXTRA_BODY: Optional[dict] = _SELFCHECK_CONFIG.get("extra_body")

    # Evaluation — from config/evaluate.yaml (evaluation: section)
    EVAL_N_SELFCHECK_VALUES: list = _N_SELFCHECK_VALUES
    EVAL_N_SELFCHECK_GEN_SAMPLES: int = _N_SELFCHECK_GEN
    EVAL_SAMPLE_RATIO: Optional[float] = _N_SAMPLE_RATIO
    HALLUCINATION_RATIO: float = _HALLUCINATION_RATIO

    # Retrieval — from config/settings.yaml (retrieval: section)
    RAG_K: int = _RAG_K

    # API — from config/settings.yaml (api: section)
    API_TIMEOUT: int = _API_TIMEOUT
    API_MAX_CONCURRENT: int = _API_MAX_CONCURRENT
    API_RETRY_MAX_ATTEMPTS: int = _API_RETRY_MAX
    API_RETRY_MIN_WAIT: int = _API_RETRY_MIN_WAIT
    API_RETRY_MAX_WAIT: int = _API_RETRY_MAX_WAIT
    API_BASELINE_TIMEOUT: Optional[int] = _API_BASELINE_TIMEOUT

    # Agent — from config/settings.yaml (agent: section)
    ROUTER_TEMPERATURE: float = _ROUTER_TEMPERATURE
    ROUTER_MAX_TOKENS: int = _ROUTER_MAX_TOKENS
    SPECIALIST_MAX_TOKENS: int = _SPECIALIST_MAX_TOKENS
    SELFCHECK_MAX_TOKENS: int = _SELFCHECK_MAX_TOKENS
    SYNTHESIS_TEMPERATURE: float = _SYNTHESIS_TEMPERATURE
    MAX_ITERATIONS: int = _MAX_ITERATIONS
    EARLY_STOPPING: bool = _EARLY_STOPPING

    # Pricing — from config/pricing.yaml
    PRICING_TABLE: dict = _PRICING_TABLE

    # Baseline models — from config/models.yaml (baselines: section), provider info merged
    BASELINE_MODELS: list = _BASELINE_MODELS

    # Default max_tokens for thinking baselines (from settings.yaml baselines.thinking_max_tokens)
    BASELINE_THINKING_MAX_TOKENS: int = _BASELINE_THINKING_MAX_TOKENS

    @staticmethod
    def resolve_model_ref(name: str) -> Optional[dict]:
        """Resolve a models.yaml baseline name to {model, base_url, api_key, extra_body}.

        Used by ablation variants to reference baselines by name instead of
        manually specifying provider connection details.
        """
        for m in Config.BASELINE_MODELS:
            if m.get("name") == name:
                return {
                    "model": m["model"],
                    "base_url": m["base_url"],
                    "api_key": m["api_key"],
                    "temperature": float(m.get("temperature", 0.0)),
                    "max_tokens": int(m.get("max_tokens", Config.MAX_TOKENS)),
                    "extra_body": m.get("extra_body"),
                }
        return None

    @staticmethod
    def _check_pricing():
        """Warn if active models are missing from pricing table."""
        import logging
        logger = logging.getLogger(__name__)
        from src.utils.cost_tracker import _resolve_pricing_key
        models_to_check = [Config.LLM_MODEL]
        for m in Config.BASELINE_MODELS:
            models_to_check.append(f"{m['provider']}::{m['model']}")
        for model in models_to_check:
            if _resolve_pricing_key(model) is None:
                logger.warning(
                    "Pricing not found for '%s' in config/pricing.yaml - "
                    "cost will be reported as $0. Add to config/pricing.yaml: %%s", model)

    # Species handler (lazy import to avoid circular)
    @classmethod
    def get_species_handler(cls):
        from src.species import get_handler
        return get_handler(cls.SPECIES)

    # Data sources metadata
    DATA_SOURCES: dict = {
        "Feed Ingredient Bank": "Industry-standard databases, peer-reviewed articles, NRC publications",
        "Performance Management": "Species-specific breed guidelines",
        "Nutrient Requirements": "Species-specific industry nutrient specifications",
        "Definitions": "Poultry science terminology, peer-reviewed literature",
    }

    # Species-specific metadata for prompts
    SPECIES_META: dict = {
        "broiler": {
            "display": "Broiler (meat-type chicken)",
            "breeds": "Ross 308, Ross 708, Cobb 500, Shengze 901, Guangming 2, Wode 188, Qingyuan Partridge, Sanhuang, Crossbred 817",
            "phases": "Starter → Grower → Finisher → Withdrawal (conventional); extended cycles (63-90d) for yellow-feathered",
            "metrics": "Body weight, ADG, FCR, feed intake, mortality, processing yield, breast meat, feather color, meat quality",
        },
        "layer": {
            "display": "Layer (egg-type chicken)",
            "breeds": "Hy-Line W-36, Hy-Line Brown, ISA Brown, Lohmann LSL-Classic, Jingfen 1, Jingfen 2, Dawu Jinfeng, Nongda 3",
            "phases": "Starter → Grower → Developer → Pre-Lay → Lay I → Lay II → Lay III",
            "metrics": "Hen-day production, egg weight, egg mass, feed/dozen, feed/kg egg, body weight, shell quality, mortality",
        },
    }


# ── 实验配置继承 ──────────────────────────────────────────
#
# 评估参数（n_selfcheck_values 等）默认值现已直接定义在
# evaluate.yaml 的 evaluation: 段中，不再从 settings.yaml 继承。

_INHERITANCE_MAP: dict = {
    "evaluate": {
        # evaluate.yaml evaluation.hallucination_ratio → settings.yaml evaluation.hallucination_ratio
        ("evaluation", "hallucination_ratio"): ("evaluation", "hallucination_ratio"),
        # evaluate.yaml evaluation.sample_ratio → settings.yaml evaluation.sample_ratio
        ("evaluation", "sample_ratio"): ("evaluation", "sample_ratio"),
    },
}


def _deep_get(d: dict, path: tuple):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def inherit_experiment_config(exp_cfg: dict, schema: str) -> dict:
    """Fill None fields in experiment config from settings.yaml defaults.

    Args:
        exp_cfg: Validated experiment config dict
        schema: Schema name ("evaluate")

    Returns:
        Config dict with inherited defaults filled in-place.
    """
    inherit = _INHERITANCE_MAP.get(schema, {})
    for exp_path, settings_path in inherit.items():
        if _deep_get(exp_cfg, exp_path) is None:
            val = _deep_get(_settings, settings_path)
            if val is not None:
                target = exp_cfg
                for key in exp_path[:-1]:
                    target = target.setdefault(key, {})
                target[exp_path[-1]] = val
    return exp_cfg
