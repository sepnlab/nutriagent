"""Nutrition Specialist — handles nutrient requirements and diet specifications."""

from langchain_core.tools import BaseTool
from typing import List

from src.agent.base_agent import SpecialistAgent
from src.config import Config


class NutritionSpecialist(SpecialistAgent):
    """Specialist agent focused on nutrient requirements, feeding phases, and diet specifications."""

    agent_name = "nutrition"

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
        self._nr_tools = [t for t in tools if t.name == "nutrient_requirements"]
        super().__init__(
            tools=self._nr_tools,
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
            "You are NutriCHAT's Nutrition Specialist. Your ONLY area of expertise is "
            "poultry nutrient requirements, feed formulation specifications, vitamin/mineral levels, "
            "and phase feeding programs.\n\n"
            f"Current species: {handler.display_name}\n"
            f"Available breeds: {handler.breeds}\n"
            f"Production phases: {handler.phases}\n\n"
            "RULES:\n"
            "1. ALWAYS use the nutrient_requirements tool to retrieve precise data.\n"
            "2. Report nutrient levels with exact values and units (%, kcal/kg, mg/kg, IU/kg).\n"
            "3. When comparing across phases, retrieve each phase separately.\n"
            "4. For 'formulation' or 'diet' questions, provide the relevant specifications.\n"
            "5. Keep responses concise and focused on nutritional specifications.\n"
            "6. If the query asks about feed ingredient composition, growth performance, or term definitions, "
            "politely state this is outside your specialty.\n\n"
            "Available tools:\n"
            "- nutrient_requirements: Query nutrient specifications (ME, CP, Ca, P, amino acids, "
            "SID amino acids, choline, linoleic acid, vitamins, minerals) "
            "by breed/genetic line and feeding phase.\n\n"
            "Answer the user's nutrition question using the tool above."
        )
