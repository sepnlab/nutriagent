"""SelfCheckGPT 模块单元测试（不依赖 NLI 模型 / LLM API）。

测试 sentence splitting、自适应并发等纯文本/结构逻辑。
"""
import time


class TestSelfCheckSentenceSplit:
    def test_simple_sentences(self):
        from src.evaluation.selfcheck import SelfCheckGPT
        sc = SelfCheckGPT(device="cpu")
        text = "The calcium content is 0.96%. The protein level is 23%."
        sents = sc._split_sentences(text)
        assert len(sents) == 2
        assert "calcium" in sents[0]
        assert "protein" in sents[1]

    def test_short_sentences_filtered(self):
        from src.evaluation.selfcheck import SelfCheckGPT
        sc = SelfCheckGPT(device="cpu")
        text = "Yes. No. The calcium content is 0.96%."
        sents = sc._split_sentences(text)
        # "Yes." and "No." are too short (< 10 chars)
        assert len(sents) == 1
        assert "calcium" in sents[0]

    def test_empty_text(self):
        from src.evaluation.selfcheck import SelfCheckGPT
        sc = SelfCheckGPT(device="cpu")
        assert sc._split_sentences("") == []

    def test_single_sentence(self):
        from src.evaluation.selfcheck import SelfCheckGPT
        sc = SelfCheckGPT(device="cpu")
        text = "The calcium content of limestone is approximately 38%."
        sents = sc._split_sentences(text)
        assert len(sents) == 1
        assert "calcium" in sents[0]

    def test_multiple_spaces(self):
        from src.evaluation.selfcheck import SelfCheckGPT
        sc = SelfCheckGPT(device="cpu")
        text = "Ca  is  0.96%.   P  is  0.45%."
        sents = sc._split_sentences(text)
        assert len(sents) == 2
        # No double spaces in output
        assert "  " not in sents[0]


class TestSelfCheckAdaptiveConcurrency:
    """Verify adaptive concurrency logic in _generate_samples."""

    def test_source_contains_adaptive_levels(self):
        """The _concurrency_levels method must define the 3 levels."""
        import inspect
        from src.evaluation.selfcheck import SelfCheckGPT
        src = inspect.getsource(SelfCheckGPT._concurrency_levels)
        for level in ("20", "10", "5"):
            assert f"{level}" in src, (
                f"Missing concurrency level {level} in _concurrency_levels"
            )

    def test_no_hardcoded_max_workers(self):
        """_generate_samples must NOT have the old fixed max_workers=5."""
        import inspect
        from src.evaluation.selfcheck import SelfCheckGPT
        src = inspect.getsource(SelfCheckGPT._generate_samples)
        assert "max_workers=5" not in src, (
            "Fixed max_workers=5 still present (should use adaptive levels)"
        )

    def test_adaptive_retry_small_n(self):
        """When n < 20, levels clamp to n, not to 20."""
        from src.evaluation.selfcheck import SelfCheckGPT
        sc = SelfCheckGPT(device="cpu")
        levels = sc._concurrency_levels(5)
        # For n=5, each level: min(5, 20)=5, min(5, 10)=5, min(5, 5)=5
        assert levels == [5, 5, 5], f"Expected [5,5,5] for n=5, got {levels}"

        levels = sc._concurrency_levels(15)
        # min(15, 20)=15, min(15, 10)=10, min(15, 5)=5
        assert levels == [15, 10, 5], f"Expected [15,10,5] for n=15, got {levels}"

    def test_adaptive_retry_large_n(self):
        """When n >= 20, levels should be [20, 10, 5]."""
        from src.evaluation.selfcheck import SelfCheckGPT
        sc = SelfCheckGPT(device="cpu")
        levels = sc._concurrency_levels(20)
        assert levels == [20, 10, 5], f"Expected [20,10,5] for n=20, got {levels}"

        levels = sc._concurrency_levels(50)
        assert levels == [20, 10, 5], f"Expected [20,10,5] for n=50, got {levels}"
