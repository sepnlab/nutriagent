"""工具模块单元测试（cost_tracker, sampling, csv_cache）。"""


class TestUsageRecord:
    def test_defaults(self):
        from src.utils.cost_tracker import UsageRecord
        r = UsageRecord(model="test-model")
        assert r.model == "test-model"
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.total_tokens == 0
        assert r.elapsed == 0.0

    def test_total_tokens(self):
        from src.utils.cost_tracker import UsageRecord
        r = UsageRecord(model="m")
        r.prompt_tokens = 50
        r.completion_tokens = 100
        assert r.total_tokens == 150

    def test_cost_zero_for_unknown_model(self):
        from src.utils.cost_tracker import UsageRecord
        r = UsageRecord(model="unknown-model")
        r.prompt_tokens = 1000
        r.completion_tokens = 500
        assert r.cost == 0.0

    def test_cost_known_model(self):
        from src.utils.cost_tracker import UsageRecord
        # Use a known pricing model
        r = UsageRecord(model="deepseek-ai/DeepSeek-V4-Flash")
        r.prompt_tokens = 1_000_000
        r.completion_tokens = 500_000
        # input=0.14/M, output=0.28/M
        expected = 0.14 * 1 + 0.28 * 0.5
        assert abs(r.cost - expected) < 0.001

    def test_to_dict(self):
        from src.utils.cost_tracker import UsageRecord
        r = UsageRecord(model="m")
        r.prompt_tokens = 10
        r.completion_tokens = 20
        r.elapsed = 1.5
        d = r.to_dict()
        assert d["model"] == "m"
        assert d["total_tokens"] == 30
        assert d["elapsed_s"] == 1.5


class TestCostTracker:
    def test_empty_summary(self):
        from src.utils.cost_tracker import CostTracker
        t = CostTracker()
        s = t.summary()
        assert "_total" in s
        assert s["_total"]["calls"] == 0

    def test_single_record(self):
        from src.utils.cost_tracker import CostTracker
        t = CostTracker()
        t.record("test-model", prompt_tokens=100, completion_tokens=50, elapsed=2.0)
        s = t.summary()
        assert s["_total"]["calls"] == 1
        assert s["_total"]["total_tokens"] == 150
        assert s["_total"]["total_time_s"] == 2.0

    def test_multiple_records_same_model(self):
        from src.utils.cost_tracker import CostTracker
        t = CostTracker()
        t.record("m", 100, 50, 1.0)
        t.record("m", 200, 100, 2.0)
        s = t.summary()
        assert s["m"]["calls"] == 2
        assert s["m"]["total_tokens"] == 450
        assert s["m"]["total_time_s"] == 3.0
        assert s["_total"]["calls"] == 2

    def test_multiple_models(self):
        from src.utils.cost_tracker import CostTracker
        t = CostTracker()
        t.record("model-a", 100, 50, 1.0)
        t.record("model-b", 200, 100, 2.0)
        s = t.summary()
        assert "model-a" in s
        assert "model-b" in s
        assert s["_total"]["calls"] == 2

    def test_cost_usd_key_in_total(self):
        """Regression: _total must have 'cost_usd' key (not 'total_cost_usd')."""
        from src.utils.cost_tracker import CostTracker
        t = CostTracker()
        t.record("unknown", 100, 50, 1.0)
        s = t.summary()
        assert "cost_usd" in s["_total"], f"Missing cost_usd, got keys: {list(s['_total'].keys())}"


class TestStratifiedSample:
    def test_none_ratio_returns_all(self):
        from src.utils.sampling import stratified_sample
        queries = [{"id": "1", "category": "A"}, {"id": "2", "category": "B"}]
        result = stratified_sample(queries, ratio=None)
        assert len(result) == 2

    def test_zero_ratio_returns_empty(self):
        from src.utils.sampling import stratified_sample
        queries = [{"id": "1", "category": "A"}]
        result = stratified_sample(queries, ratio=0)
        assert result == []

    def test_count_mode(self):
        from src.utils.sampling import stratified_sample
        queries = [
            {"id": "1", "category": "A"},
            {"id": "2", "category": "A"},
            {"id": "3", "category": "A"},
            {"id": "4", "category": "B"},
        ]
        result = stratified_sample(queries, ratio=1)
        assert len(result) == 2  # 1 per category

    def test_category_filter(self):
        from src.utils.sampling import stratified_sample
        queries = [
            {"id": "1", "category": "A"},
            {"id": "2", "category": "B"},
            {"id": "3", "category": "C"},
        ]
        result = stratified_sample(queries, ratio=10, categories=["A", "B"])
        categories = {q["category"] for q in result}
        assert "C" not in categories
        assert len(result) == 2

    def test_deterministic_seed(self):
        from src.utils.sampling import stratified_sample
        queries = [{"id": str(i), "category": "A"} for i in range(10)]
        r1 = stratified_sample(queries, ratio=3, seed=42)
        r2 = stratified_sample(queries, ratio=3, seed=42)
        assert [q["id"] for q in r1] == [q["id"] for q in r2]


class TestCsvCache:
    def test_empty_path(self):
        from src.utils.csv_cache import load_csv
        df = load_csv("")
        assert df.empty

    def test_nonexistent_path(self):
        from src.utils.csv_cache import load_csv
        df = load_csv("/nonexistent/path.csv")
        assert df.empty

    def test_normalize_column(self):
        from src.utils.csv_cache import normalize_column
        import pandas as pd
        df = pd.DataFrame({"Name": ["  Alice  ", "  BOB  "]})
        normalize_column(df, "Name")
        assert df.iloc[0]["Name"] == "alice"
        assert df.iloc[1]["Name"] == "bob"

    def test_normalize_missing_column(self):
        from src.utils.csv_cache import normalize_column
        import pandas as pd
        df = pd.DataFrame({"Name": ["Alice"]})
        normalize_column(df, "NonExistent")  # should not raise
        assert len(df) == 1
