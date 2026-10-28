"""Performance Specialist — handles production performance and growth queries."""

from langchain_core.tools import BaseTool
from typing import List

from src.agent.base_agent import SpecialistAgent
from src.config import Config


class PerformanceSpecialist(SpecialistAgent):
    """Specialist agent focused on production performance metrics."""

    agent_name = "performance"

    def __init__(
        self,
        tools: List[BaseTool],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        verbose: bool = False,
        base_url: str = None,
        api_key: str = None,
        extra_body: dict = None,
        single_pass: bool = False,
    ):
        self._perf_tools = [t for t in tools if t.name == "performance_management"]
        super().__init__(
            tools=self._perf_tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body,
            single_pass=single_pass,
        )

    def _build_prompt(self) -> str:
        handler = Config.get_species_handler()
        return (
            "You are NutriCHAT's Performance Specialist. Your ONLY area of expertise is "
            "poultry production performance: body weight, growth rate, feed conversion, "
            "egg production, egg quality, mortality, and processing yield.\n\n"
            f"Current species: {handler.display_name}\n"
            f"Available breeds: {handler.breeds}\n"
            f"Production phases: {handler.phases}\n\n"
            "RULES:\n"
            "1. ALWAYS use the performance_management tool to retrieve precise performance data.\n"
            "2. Report values with exact numbers and units (g, g/d, %, N, etc.).\n"
            "3. For comparison queries across breeds or ages, retrieve each data point separately.\n"
            "4. For growth curve questions, retrieve multiple age points and summarize the pattern.\n"
            "5. Keep responses concise and focused on performance metrics.\n"
            "6. If the query asks about feed composition, nutrient requirements, or term definitions, "
            "politely state this is outside your specialty.\n\n"
            "Available tools:\n"
            "- performance_management: Query production performance data by breed/genetic line, "
            f"age ({handler.performance_age_desc}), and parameter.\n\n"
            "Answer the user's performance question using the tool above."
        )
