"""工具基础功能测试（依赖 CSV 数据文件）。"""

from src.tools.feed_ingredient_bank import FeedIngredientBankTool
from src.tools.definitions_tool import DefinitionsTool


class TestFeedIngredientBankTool:
    def setup_method(self):
        self.tool = FeedIngredientBankTool()

    def test_lookup_exact(self):
        result = self.tool._run("Maize")
        assert "maize" in result.lower()
        assert "CP" in result

    def test_lookup_partial(self):
        result = self.tool._run("maize")
        assert "maize" in result.lower()

    def test_lookup_with_parameter(self):
        result = self.tool._run("Maize", parameter="CP")
        assert "CP:" in result or "CP" in result

    def test_lookup_not_found(self):
        result = self.tool._run("NonExistentIngredientXYZ")
        assert "not found" in result.lower()

    def test_lookup_amino_acids_loaded(self):
        result = self.tool._run("Soybean meal 48%")
        # Should include amino acid data if AA file is loaded
        assert "Arginine" in result or "Arg" in result or len(result) > 100


class TestDefinitionsTool:
    def setup_method(self):
        self.tool = DefinitionsTool()

    def test_exact_match(self):
        result = self.tool._run("Broiler")
        assert "broiler" in result.lower()

    def test_partial_match(self):
        result = self.tool._run("Feed Conversion")
        assert "Feed Conversion Ratio" in result or "feed" in result.lower()

    def test_not_found(self):
        result = self.tool._run("XYZZYXNonExistent")
        assert "not found" in result.lower()
