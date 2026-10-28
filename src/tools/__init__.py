"""Tool factory."""
from src.tools.feed_ingredient_bank import FeedIngredientBankTool
from src.tools.performance_management import PerformanceManagementTool
from src.tools.nutrient_requirements import NutrientRequirementsTool
from src.tools.definitions_tool import DefinitionsTool
from src.utils.csv_cache import load_csv, normalize_column

TOOL_DOMAINS = {
    "feed_ingredient_bank": "Feed Ingredient Bank",
    "performance_management": "Performance Management",
    "nutrient_requirements": "Nutrient Requirements",
    "definitions": "Definitions",
}


def get_all_tools():
    return [
        FeedIngredientBankTool(),
        PerformanceManagementTool(),
        NutrientRequirementsTool(),
        DefinitionsTool(),
    ]


def get_tool_by_name(name: str):
    for tool in get_all_tools():
        if tool.name == name:
            return tool
    return None


def get_tool_domain(tool_name: str) -> str:
    return TOOL_DOMAINS.get(tool_name, "Unknown")
