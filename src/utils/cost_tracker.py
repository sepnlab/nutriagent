"""Token 用量追踪与成本计算。

追踪每次 API 调用的 token 用量和耗时，
结合 Config.PRICING_TABLE 计算预估成本。
"""
import time

from src.config import Config


_warned_models = set()


def _warn_missing_pricing(model: str):
    """Log a warning once per model if pricing is missing."""
    if model not in _warned_models:
        import logging
        logging.getLogger(__name__).warning(
            "Pricing not found for '%s' in config/pricing.yaml — cost shown as $0. "
            "Add an entry like:\n  %s:\n    input: 0.xx\n    output: 0.xx",
            model, model,
        )
        _warned_models.add(model)


def _resolve_pricing_key(model: str) -> str | None:
    """查找定价 key，支持 {provider}::{model_id} 格式并回退到 bare model_id。

    1. 精确匹配 model
    2. 遍历 pricing table，查找以 ::{model_id} 结尾的 key
    """
    pricing = Config.PRICING_TABLE
    if model in pricing:
        return model
    # 回退：遍历所有 key，找以 ::{model_id} 结尾的匹配
    suffix = f"::{model}"
    for key in pricing:
        if key.endswith(suffix):
            return key
    return None


def get_model_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """计算单次 API 调用的预估成本（USD）。"""
    pricing = Config.PRICING_TABLE
    key = _resolve_pricing_key(model)
    if key is None:
        if prompt_tokens > 0 or completion_tokens > 0:
            _warn_missing_pricing(model)
        return 0.0
    rates = pricing.get(key, {})
    in_cost = rates.get("input", 0) * prompt_tokens / 1_000_000
    out_cost = rates.get("output", 0) * completion_tokens / 1_000_000
    return in_cost + out_cost


class UsageRecord:
    """单次查询的用量记录。"""

    def __init__(self, model: str = ""):
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.elapsed = 0.0  # 秒

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> float:
        return get_model_cost(self.model, self.prompt_tokens, self.completion_tokens)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_s": round(self.elapsed, 2),
            "cost_usd": round(self.cost, 6),
        }


class CostTracker:
    """多模型成本追踪器。

    使用示例：
        tracker = CostTracker()
        r1 = tracker.start("deepseek-ai/DeepSeek-V4-Flash")
        # ... API call ...
        r1.finish(prompt_tokens=50, completion_tokens=100, elapsed=2.5)

        r2 = tracker.start("Qwen/Qwen3.5-9B")
        # ... API call ...
        r2.finish(prompt_tokens=30, completion_tokens=60, elapsed=1.2)

        summary = tracker.summary()  # 按模型汇总
    """

    def __init__(self):
        self.records: list[UsageRecord] = []

    def record(self, model: str, prompt_tokens: int, completion_tokens: int, elapsed: float):
        """记录一次 API 调用。"""
        r = UsageRecord(model)
        r.prompt_tokens = prompt_tokens
        r.completion_tokens = completion_tokens
        r.elapsed = elapsed
        self.records.append(r)
        return r

    def summary(self) -> dict:
        """按模型汇总统计。"""
        from collections import defaultdict
        by_model: dict = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0,
                                               "completion_tokens": 0, "total_time": 0.0})

        for r in self.records:
            m = r.model
            by_model[m]["calls"] += 1
            by_model[m]["prompt_tokens"] += r.prompt_tokens
            by_model[m]["completion_tokens"] += r.completion_tokens
            by_model[m]["total_time"] += r.elapsed

        result = {}
        total_cost = 0.0
        for model, data in by_model.items():
            total_tok = data["prompt_tokens"] + data["completion_tokens"]
            cost = get_model_cost(model, data["prompt_tokens"], data["completion_tokens"])
            result[model] = {
                "calls": data["calls"],
                "prompt_tokens": data["prompt_tokens"],
                "completion_tokens": data["completion_tokens"],
                "total_tokens": total_tok,
                "total_time_s": round(data["total_time"], 2),
                "avg_time_s": round(data["total_time"] / data["calls"], 2) if data["calls"] else 0,
                "cost_usd": round(cost, 6),
            }
            total_cost += cost

        total_calls = sum(d["calls"] for d in by_model.values())
        total_time = sum(d["total_time"] for d in by_model.values())
        result["_total"] = {
            "calls": total_calls,
            "total_tokens": sum(d["prompt_tokens"] + d["completion_tokens"] for d in by_model.values()),
            "total_time_s": round(total_time, 2),
            "avg_time_s": round(total_time / total_calls, 2) if total_calls else 0,
            "cost_usd": round(total_cost, 6),
        }
        return result
