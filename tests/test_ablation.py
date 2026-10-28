"""消融实验框架测试（不依赖 LLM API）。

测试消融变体的构建逻辑和工具函数，不实际运行 LLM。
"""
import pytest
import json
import os
import tempfile

from src.evaluation.ablation import _ABLATION_RESPONSE_CACHE


class TestAblationUtils:
    """测试消融框架的工具函数。"""

    def test_disk_cache_roundtrip(self):
        """磁盘缓存写入和读取（嵌套格式 {variant: {query: resp}}）。"""
        from src.evaluation.ablation import _save_disk_cache, _load_disk_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            import src.evaluation.ablation as abl
            original_file = abl._ABLATION_CACHE_FILE
            try:
                abl._ABLATION_CACHE_FILE = cache_path
                abl._ABLATION_RESPONSE_CACHE.clear()
                abl._ABLATION_RESPONSE_CACHE["Variant A"] = {"What is CP?": "CP is crude protein"}

                abl._save_disk_cache()
                assert os.path.exists(cache_path)

                # Verify nested format
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                assert data["Variant A"]["What is CP?"] == "CP is crude protein"

                # Load into fresh cache
                abl._ABLATION_RESPONSE_CACHE.clear()
                abl._ABLATION_CACHE_FILE = cache_path
                abl._load_disk_cache()
                assert abl._ABLATION_RESPONSE_CACHE["Variant A"]["What is CP?"] == "CP is crude protein"
            finally:
                abl._ABLATION_CACHE_FILE = original_file
                abl._ABLATION_RESPONSE_CACHE.clear()

    def test_disk_cache_loads_flat_format(self):
        """磁盘缓存能兼容旧的扁平格式 {variant|||query: resp}。"""
        from src.evaluation.ablation import _load_disk_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            # Write old flat format
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"Variant A|||What is CP?": "CP is crude protein"}, f)

            import src.evaluation.ablation as abl
            original_file = abl._ABLATION_CACHE_FILE
            try:
                abl._ABLATION_RESPONSE_CACHE.clear()
                abl._ABLATION_CACHE_FILE = cache_path
                abl._load_disk_cache()
                # Should normalize to nested format
                assert abl._ABLATION_RESPONSE_CACHE["Variant A"]["What is CP?"] == "CP is crude protein"
            finally:
                abl._ABLATION_CACHE_FILE = original_file
                abl._ABLATION_RESPONSE_CACHE.clear()

    def test_val_extraction(self):
        """_val 返回 (mean, std, n) 元组。"""
        from src.evaluation.ablation import _val
        data = {"correctness": {"mean": 4.5, "std": 0.5, "n": 10}}
        assert _val(data, "correctness") == (4.5, 0.5, 10)
        assert _val(data, "nonexistent") == (None, None, 0)
        assert _val({}, "key") == (None, None, 0)

    def test_fmt_pct(self):
        """_fmt_pct 格式化百分比（含 std/n）。"""
        from src.evaluation.ablation import _fmt_pct
        assert _fmt_pct(0.5) == "50.0%"
        assert _fmt_pct(0.5, 0.03, 10) == "50.0%±3.0% (n=10)"
        assert _fmt_pct(0.956) == "95.6%"
        assert _fmt_pct(None) == "N/A"

    def test_fmt_decimal(self):
        """_fmt 格式化小数（含 std/n）。"""
        from src.evaluation.ablation import _fmt
        assert _fmt(4.567) == "4.57"
        assert _fmt(4.567, 0.12, 10, 2) == "4.57±0.12 (n=10)"
        assert _fmt(0.12345, decimals=4) == "0.1235"
        assert _fmt(None) == "N/A"


class TestAblationConfig:
    """测试消融配置加载。"""

    def test_ablation_yaml_loads(self):
        """ablation.yaml 能正确加载并含有预期的维度。"""
        from src.evaluation.ablation import _load_ablation_config
        cfg = _load_ablation_config()
        assert isinstance(cfg, dict)
        for dim in ("architecture", "specialist", "router", "rag"):
            assert dim in cfg, f"Missing dimension: {dim}"

    def test_architecture_variants(self):
        """架构消融维度使用显式 dispatch 的规范变体名。"""
        from src.evaluation.ablation import _load_ablation_config
        cfg = _load_ablation_config()
        variants = cfg["architecture"]["variants"]
        names = [v["name"] for v in variants]
        assert "Single ReAct" in names
        assert "Multi-Agent (full)" in names
        assert "Multi-Agent (single-pass tool call)" in names
        assert "Multi-Agent (no tools)" in names
        # The invalid legacy name must not be used in the config anymore.
        assert "Multi-Agent (no ReAct)" not in names

    def test_specialist_variants(self):
        """Specialist 消融有 exclude 字段。"""
        from src.evaluation.ablation import _load_ablation_config
        cfg = _load_ablation_config()
        variants = cfg["specialist"]["variants"]
        for v in variants:
            if v["name"] == "Full (4 Specialists)":
                assert "exclude" not in v or v.get("exclude") is None
            else:
                assert "exclude" in v, f"{v['name']} missing 'exclude'"

    def test_router_variants(self):
        """路由消融有正确的策略值。"""
        from src.evaluation.ablation import _load_ablation_config
        cfg = _load_ablation_config()
        variants = cfg["router"]["variants"]
        strategies = [v.get("strategy") for v in variants]
        assert "llm" in strategies
        assert "keyword" in strategies
        assert "broadcast" in strategies

    def test_rag_variants(self):
        """RAG 消融有正确的启停配置。"""
        from src.evaluation.ablation import _load_ablation_config
        cfg = _load_ablation_config()
        variants = cfg["rag"]["variants"]
        for v in variants:
            assert "rag_enabled" in v, f"{v['name']} missing rag_enabled"


class TestAblationScoring:
    """测试消融的评分报告函数。"""

    def test_print_table_format(self):
        """_print_table 不会崩溃。"""
        from src.evaluation.ablation import _print_table
        results = {
            "Variant A": {
                "correctness": {"mean": 4.5, "std": 0.5, "n": 10},
                "precision": {"mean": 3.8, "std": 0.8, "n": 8},
                "scientific_depth": {"mean": 4.0, "std": 0.6, "n": 10},
                "flesch_reading_ease": {"mean": 45.0, "std": 10.0, "n": 10},
                "hallucination": {"mean": 0.1, "std": 0.05, "n": 10},
                "faithfulness": {"mean": 0.85, "std": 0.1, "n": 10},
                "faithfulness_semantic": {"mean": 0.72, "std": 0.12, "n": 10},
                "tolerance_p2": {"mean": 0.8, "std": 0.2, "n": 10},
                "tolerance_p5": {"mean": 0.9, "std": 0.1, "n": 10},
            },
        }
        # Should not raise
        _print_table("Test", results,
                     metrics=["correctness", "precision", "faithfulness_semantic"],
                     headers=["Correctness", "Precision", "Faith(Sem)"])

    def test_print_table_empty(self):
        """空结果不崩溃。"""
        from src.evaluation.ablation import _print_table
        _print_table("Empty", {}, metrics=["correctness"], headers=["C"])
