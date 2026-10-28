"""Baseline model runner for NutriCHAT comparative evaluation.

Loads baseline model configurations from Config.BASELINE_MODELS (populated
from config/models.yaml with shared defaults merged), creates LLM clients,
and runs queries for comparison against NutriCHAT.
"""
import logging
import time
from typing import Dict, List, Optional

from src.config import Config
from src.retry import llm_retry

logger = logging.getLogger(__name__)


def load_baseline_config() -> List[dict]:
    """Load baseline model configs from Config (populated from models.yaml).

    Config already merges _defaults and resolves api_key from env.
    Skips entries whose required api_key is empty.
    """
    configs = Config.BASELINE_MODELS
    resolved = []
    for b in configs:
        if not b.get("api_key"):
            env_var = b.get("api_key_env", "")
            logger.warning("SKIP %s: env var '%s' not set", b["name"], env_var)
            continue
        resolved.append(b)
    return resolved


def create_baseline_client(baseline: dict):
    """Create an LLM client for a baseline model based on its provider.

    Returns a callable: client.invoke(messages) -> str
    """
    provider = baseline.get("provider", "siliconflow").lower()

    if provider in (
        "siliconflow",
        "openai",
        "openrouter",
        "aliyun",
        "volcano",
        "tencent",
        "deepseek",
        "qianwen",
        "xiaomi",
    ):
        from openai import OpenAI

        default_urls = {
            "siliconflow": "https://api.siliconflow.cn/v1",
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "deepseek": "https://api.deepseek.com",
            "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "volcano": "https://ark.cn-beijing.volces.com/api/v3",
            "tencent": "https://tokenhub.tencentmaas.com/v1",
            "qianwen": "https://maas.qianwenaiapi.com/compatible-mode/v1",
            "xiaomi": "https://api.xiaomimimo.com/v1",
        }
        client = OpenAI(
            base_url=baseline.get("base_url", default_urls.get(provider, "https://openrouter.ai/api/v1")),
            api_key=baseline["api_key"],
            timeout=Config.API_BASELINE_TIMEOUT or Config.API_TIMEOUT,
        )

        def invoke(messages: List[dict]) -> dict:
            try:
                @llm_retry
                def _call() -> dict:
                    kwargs = dict(
                        model=baseline["model"],
                        messages=messages,
                        temperature=baseline.get("temperature", 0.0),
                        max_tokens=baseline.get("max_tokens", 200),
                    )
                    extra_body = baseline.get("extra_body")
                    if extra_body:
                        kwargs["extra_body"] = extra_body
                    return client.chat.completions.create(**kwargs)
                resp = _call()
                usage = resp.usage
                _pricing_key = f"{baseline['provider']}::{baseline['model']}"
                return {
                    "response": resp.choices[0].message.content or "",
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "model": baseline["model"],
                    "_pricing_key": _pricing_key,
                }
            except Exception as e:
                _pricing_key = f"{baseline['provider']}::{baseline['model']}"
                return {"response": f"[BaselineError] {e}", "prompt_tokens": 0, "completion_tokens": 0, "model": baseline["model"], "_pricing_key": _pricing_key}

        return invoke

    elif provider == "anthropic":
        from anthropic import Anthropic
        from anthropic import APIError as AnthropicAPIError
        from anthropic import RateLimitError as AnthropicRateLimit

        client = Anthropic(api_key=baseline["api_key"])

        # Build retry decorator for Anthropic exceptions
        from src.retry import _make_retry
        _anthropic_retry = _make_retry(
            max_attempts=Config.API_RETRY_MAX_ATTEMPTS,
            min_wait=Config.API_RETRY_MIN_WAIT,
            max_wait=Config.API_RETRY_MAX_WAIT,
            exceptions=(AnthropicAPIError, AnthropicRateLimit, ConnectionError),
        )

        def invoke(messages: List[dict]) -> dict:
            system = None
            msgs = []
            for m in messages:
                if m.get("role") == "system":
                    system = m["content"]
                else:
                    msgs.append({"role": m["role"], "content": m["content"]})

            @_anthropic_retry
            def _call():
                return client.messages.create(
                    model=baseline["model"],
                    system=system,
                    messages=msgs,
                    temperature=baseline.get("temperature", 0.0),
                    max_tokens=baseline.get("max_tokens", 200),
                )

            response = _call()
            text = response.content[0].text if response.content else ""
            usage = getattr(response, "usage", None)
            return {
                "response": text,
                "prompt_tokens": usage.input_tokens if usage else 0,
                "completion_tokens": usage.output_tokens if usage else 0,
                "model": baseline["model"],
            }

        return invoke

    else:
        raise ValueError(f"Unknown provider: {provider}")


class BaselineRunner:
    """Runs evaluation queries against multiple baseline LLMs.

    Usage:
        runner = BaselineRunner()
        runner.load_baselines()
        runner.run_all(queries)
        runner.print_comparison(nutrichat_results)
    """

    def __init__(self):
        self.baselines: List[dict] = []
        self.clients: Dict[str, callable] = {}
        self.results: Dict[str, List[dict]] = {}

    def load_baselines(self, filter_names: Optional[List[str]] = None):
        """Load and initialize baseline models, optionally filtered by name."""
        configs = load_baseline_config()
        if filter_names:
            name_set = set(filter_names)
            configs = [b for b in configs if b["name"] in name_set]
            skipped = set(name_set) - {b["name"] for b in configs}
            if skipped:
                logger.warning("Baselines not found in config: %s", skipped)
        self.baselines = configs
        if filter_names:
            logger.info("Filtered baselines: %s", [b["name"] for b in configs])
        for b in configs:
            try:
                self.clients[b["name"]] = create_baseline_client(b)
                logger.info("OK   %s (%s:%s)", b["name"], b["provider"], b["model"])
            except Exception as e:
                logger.error("ERR  %s: %s", b["name"], e)

    def run_query(self, baseline_name: str, query: str) -> str:
        """Run a single query against a specific baseline model.

        Returns the response text. Usage info available via .last_usage.
        NOT thread-safe for last_usage — use run_query_with_usage() for parallel calls.
        """
        resp, usage = self.run_query_with_usage(baseline_name, query)
        return resp

    def run_query_with_usage(self, baseline_name: str, query: str) -> tuple:
        """Run a single query, returning (response, usage_dict) directly.

        Thread-safe: usage is returned in the tuple, not stored on self.
        """
        client = self.clients.get(baseline_name)
        if not client:
            empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "model": ""}
            return f"[Error] Baseline '{baseline_name}' not initialized.", empty_usage

        try:
            messages = [
                {"role": "system", "content": "You are a poultry nutrition assistant."},
                {"role": "user", "content": query},
            ]
            result = client(messages)
            usage = {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "model": result.get("model", ""),
                "_pricing_key": result.get("_pricing_key", ""),
            }
            # Also set _last_usage for backward-compatible sequential callers
            self._last_usage = usage
            return result.get("response", ""), usage
        except Exception as e:
            empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "model": ""}
            self._last_usage = empty_usage
            return f"[Error] {e}", empty_usage

    @property
    def last_usage(self) -> dict:
        """Token usage from the last run_query() call."""
        return getattr(self, "_last_usage", {"prompt_tokens": 0, "completion_tokens": 0, "model": ""})

    def run_all(self, queries: List[dict]) -> Dict[str, List[dict]]:
        """Run all queries against all baseline models.

        Args:
            queries: List of query dicts with "id", "query", "category", "type" keys.

        Returns:
            {baseline_name: [{"query_id": ..., "response": ..., ...}, ...]}
        """
        results: Dict[str, List[dict]] = {}
        for b in self.baselines:
            name = b["name"]
            results[name] = []
            logger.info("Baseline: %s", name)
            for i, q in enumerate(queries):
                logger.info("  [%d/%d] %s...", i + 1, len(queries), q['query'][:50])
                try:
                    start = time.time()
                    response = self.run_query(name, q["query"])
                    elapsed = time.time() - start
                    logger.info("(%0.1fs)", elapsed)
                except Exception as e:
                    response = f"[Error] {e}"
                    logger.error("ERR: %s", e)

                results[name].append({
                    "query_id": q.get("id", f"Q{i+1:03d}"),
                    "query": q["query"],
                    "category": q.get("category", ""),
                    "query_type": q.get("type", "unknown"),
                    "response": response,
                })

        self.results = results
        return results

    def print_comparison(self, nutrichat_report: dict):
        """Print a comparison table of NutriCHAT vs all baselines.

        Args:
            nutrichat_report: Output of evaluator.generate_summary_report("NutriCHAT", ...)
        """
        print("\n" + "=" * 70)
        print("  MODEL COMPARISON SUMMARY")
        print("=" * 70)
        print(f"  {'Model':<28} {'Correctness':<14} {'Precision':<14} {'Scientific Depth':<16} {'Overall':<10}")
        print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*16} {'-'*10}")

        nc = nutrichat_report
        print(f"  {'NutriCHAT (ReAct+RAG)':<28} {nc['correctness']['mean']:<14.2f} "
              f"{nc['precision']['mean'] if nc['precision']['mean'] else 0:<14.2f} "
              f"{nc['scientific_depth']['mean']:<16.2f} {nc['overall_weighted_score']:<10.2f}")

        # Note: baseline scoring requires expert or LLM evaluator annotation
        print(f"\n  * Baseline scores require expert or EvaluatorLLM annotation.")
        print(f"  * This runner collects responses only.")
        print(f"  * Run --evaluate to score NutriCHAT, then annotate baseline")
        print(f"    responses using the same evaluation criteria for comparison.")
        print("=" * 70)
